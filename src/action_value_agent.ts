import { 
    BaseEnvironment, 
    Observation, 
    Reward,
    Action,
    Terminal,
    BaseAgent,
} from "./rl_base";
import { RLGlue } from './rl_glue';
import * as math from 'mathjs';
import { type } from "jquery";
import {Layer, mat_mul, Grads} from './td_nn'
import {BaseOptimizer} from './optimizer'
import * as d3 from 'd3';


function broadcast_add(matrix, vector) {
    return math.matrix(d3.range(matrix.size()[0]).map( i => {
        return (math.add(row(matrix, i), math.flatten(vector)) as math.Matrix).toJSON().data;
    }))
}

function sum_across_col(matrix) {
    return math.matrix(
        [d3.range(matrix.size()[1]).map(i => {
            return math.sum(get_col(matrix, i))
        })])
}

export class ActionValueNetwork {
    state_dim: number
    num_hidden_units: number
    num_actions: number
    layer_sizes: number[]
    weights: Layer[];

    constructor(network_config) {
        this.state_dim = network_config.state_dim;
        this.num_hidden_units = network_config.num_hidden_units;
        this.num_actions = network_config.num_actions;
        this.layer_sizes= [this.state_dim, this.num_hidden_units, this.num_actions]

        this.weights = new Array(this.layer_sizes.length - 1);
        

        for (let i = 0; i < this.weights.length; i++) {
            let inputs = this.layer_sizes[i];
            let outputs = this.layer_sizes[i + 1]
            let W = this.random_matrix(inputs, outputs);
            let b = math.matrix(math.zeros([1, outputs]));
            this.weights[i] = {
                W: W,
                b: b
            }
        }
    }

    random_matrix(rows, cols){
        return  math.matrix(d3.range(rows).map(function(){
            return d3.range(cols).map(function(){return -1 + Math.random()* 2});
          }));
    }

    get_action_values(s: math.Matrix) {
        let layer0 = this.weights[0];
        let layer1 = this.weights[1];
        let dot_prod = mat_mul(s, layer0.W)
        let v = broadcast_add(dot_prod, layer0.b)
        v = math.map(v, function(value) {
            return math.max(value, 0)
          }) 
        v = broadcast_add(mat_mul(v, layer1.W), layer1.b) as math.Matrix;
        return v;
    }

    cloneWeights(weights: Layer[]): Layer[] {
        return weights.map(layer => {
            return {
                W: math.clone(layer.W),
                b: math.clone(layer.b)
            }
    })}

    get_TD_update(s: math.Matrix, delta_mat: math.Matrix): Grads[] {
        let s_dim = s.size()[0]
        let grads: Grads[] = new Array(2);
        let layer0 = this.weights[0];
        let layer1 = this.weights[1];
        let psi = broadcast_add(mat_mul(s, layer0.W), layer0.b) as math.Matrix;
        let x = math.map(psi, function(value) {
            return math.max(value, 0)
          })

        let dx = math.map(x, function(v) { // dx
            return v > 0 ? 1 : 0;
        })

        grads[1] = {
            W:  math.divide(mat_mul(math.transpose(x), delta_mat), s_dim) as math.Matrix,
            b: math.divide(sum_across_col(delta_mat), s_dim) as math.Matrix
        }
        
        let v = math.dotMultiply(mat_mul(delta_mat, math.transpose(layer1.W)), dx) as math.Matrix

        grads[0] = {
            W: math.divide(mat_mul(math.transpose(s), v), s_dim) as math.Matrix,
            b: math.divide(sum_across_col(v), s_dim) as math.Matrix

        }
        return grads;
    };

    get_weights(): Layer[] {
        return this.cloneWeights(this.weights)
    }

    set_weights(weights: Layer[]) {
        this.weights = this.cloneWeights(weights)
    }

    clone(){
        let network_config = {
            state_dim: this.state_dim,
            num_hidden_units: this.num_hidden_units,
            num_actions: this.num_actions
        }
        let new_network = new ActionValueNetwork(network_config)
        new_network.set_weights(this.get_weights())
        return new_network
    }
}

export class AdamOptimizer implements BaseOptimizer {
    layer_sizes: number[]
    step_size: number
    beta_m: number
    beta_v: number
    epsilon: number
    layer_size: number[]
    m: Layer[]
    v: Layer[]
    beta_m_product: number
    beta_v_product: number

    constructor(layer_sizes, optimizer_info) {
        optimizer_info['layer_sizes'] = layer_sizes
        this.optimizer_init(optimizer_info)
    }

    optimizer_init(optimizer_info: any) {
        this.layer_sizes = optimizer_info.layer_sizes;
        this.step_size = optimizer_info.step_size;
        this.beta_m = optimizer_info.beta_m;
        this.beta_v = optimizer_info.beta_v;
        this.epsilon = optimizer_info.epsilon;

        this.m = new Array(this.layer_sizes.length - 1)
        this.v = new Array(this.layer_sizes.length - 1)

        for (let i = 0; i < this.layer_sizes.length - 1; i++) {
            this.m[i] = {
                W: math.zeros(this.layer_sizes[i], this.layer_sizes[i + 1]) as math.Matrix,
                b: math.zeros(1, this.layer_sizes[i + 1]) as math.Matrix
            }
            this.v[i] = {
                W: math.zeros(this.layer_sizes[i], this.layer_sizes[i + 1]) as math.Matrix,
                b: math.zeros(1, this.layer_sizes[i + 1]) as math.Matrix
            }
        }

        this.beta_m_product = this.beta_m
        this.beta_v_product = this.beta_v

    }

    update_weights(weights: Layer[], g: Grads[]) {
        for (let i = 0; i < weights.length; i++){
            this.m[i] = {
                W: math.add(math.multiply(this.beta_m,  this.m[i].W ), math.multiply(1 - this.beta_m, g[i].W)) as math.Matrix,
                b:  math.add(math.multiply(this.beta_m,  this.m[i].b), math.multiply(1 - this.beta_m, g[i].b)) as math.Matrix
            }
            this.v[i] = {
                W: math.add(math.multiply(this.beta_v,  this.v[i].W ), math.multiply(1- this.beta_v, math.dotMultiply(g[i].W, g[i].W))) as math.Matrix,
                b:  math.add(math.multiply(this.beta_v,  this.v[i].b ), math.multiply(1 - this.beta_v, math.dotMultiply(g[i].b, g[i].b))) as math.Matrix
            }
            let m_hatW = math.dotDivide(this.m[i].W, (1 - this.beta_m_product)) as math.Matrix;
            let m_hat_b = math.dotDivide(this.m[i].b, (1 - this.beta_m_product)) as math.Matrix;

            let v_hatW = math.dotDivide(this.v[i].W, (1 - this.beta_v_product)) as math.Matrix;
            let v_hat_b = math.dotDivide(this.v[i].b, (1 - this.beta_v_product)) as math.Matrix;

            weights[i] = {
                W: math.add(
                    weights[i].W,
                    math.dotDivide(math.multiply(this.step_size, m_hatW), math.add(math.sqrt(v_hatW), this.epsilon))
                ) as math.Matrix,
                b: math.add(
                    weights[i].b,
                    math.dotDivide(math.multiply(this.step_size, m_hat_b), math.add(math.sqrt(v_hat_b), this.epsilon))
                ) as math.Matrix
            }
        }
        this.beta_m_product *= this.beta_m;
        this.beta_v_product *= this.beta_v;
        
        return weights
    }
}

export class ReplayBuffer {
    buffer: object[]
    minimatch_size: number
    max_size: number
    constructor(size, minimatch_size) {
        this.buffer = []
        this.minimatch_size = minimatch_size
        this.max_size = size
    }

    append(state, action, reward, terminal, next_state) {
        if (this.buffer.length == this.max_size) {
            this.buffer.shift();
        }
        this.buffer.push([state, action, reward, terminal, next_state])
    }

    sample() {
        let shuffled = d3.shuffle(this.buffer)
        return shuffled.slice(0, this.minimatch_size);
    }

    
    size() {
        return this.buffer.length;
    }
}

function row(matrix, index, flatten=true) {
    var cols = math.size(matrix).valueOf()[1];
    let ans = math.subset(matrix, math.index(index, math.range(0, cols)));
    if (flatten) return math.flatten(ans);
    return ans
  }

function col(matrix, index) {
    var rows = math.size(matrix).valueOf()[0];
    return math.flatten(math.subset(matrix, math.index([0, rows], index)));
  }

export function softmax(action_values: math.Matrix, tau: number = 1.0) {
    let preferences = math.divide(action_values, tau) as math.Matrix
    let action_probas = d3.range(preferences.size()[0]).map(i => {
        let r = row(preferences, i)
        let max_r = math.max(r)
        let exp_r = math.map(r, function(v) {return math.exp(v - max_r)})
        let exp_r_sum = math.sum(exp_r)
        return math.map(exp_r, function(v) {return v / exp_r_sum})
    })
    return math.matrix(action_probas) 
}

export function get_td_error(
    states, 
    next_states, 
    actions, 
    rewards, 
    discount, 
    terminals, 
    network: ActionValueNetwork, 
    current_q: ActionValueNetwork, 
    tau) {
        let q_next_mat = current_q.get_action_values(next_states) as math.Matrix
        let probs_mat = softmax(q_next_mat, tau) as math.Matrix

        let v_next_vec = math.matrix(d3.range(q_next_mat.size()[0]).map(i => {
            let t_index;
            if (terminals.size().length == 1) {
                t_index = [i]
            } else t_index = [i, 0]
            return math.sum(
                math.multiply(row(q_next_mat, i), row(probs_mat, i))
                ) - terminals.get(t_index)
        }))
        rewards = math.flatten(rewards)
        let target_vec = math.add(rewards, math.multiply(discount, v_next_vec))

        let q_mat = network.get_action_values(states)

        let q_vec = math.matrix(d3.range(q_mat.size()[0]).map(i => {
            let t_index;
            if (actions.size().length == 1) {
                t_index = [i]
            } else t_index = [i, 0]
                let action = actions.get(t_index)
                return q_mat.get([i, action])
            }))
        
        let delta_vec = math.subtract(target_vec, q_vec)
        return delta_vec
        }

function get_col(arr, idx: number) {
    return arr.map(a => {
        return a[idx]
    })
}
export function optimize_network(
    experiences,
    discount, 
    optimizer: AdamOptimizer, 
    network: ActionValueNetwork,
    current_q,
    tau,
) {
    let states = math.matrix(get_col(experiences, 0))
    let actions = math.matrix(get_col(experiences, 1))
    let rewards = math.matrix(get_col(experiences, 2))
    let terminals = math.matrix(get_col(experiences, 3))
    let next_states = math.matrix(get_col(experiences, 4))

    let batch_size = experiences.length
    let delta_vec = get_td_error(
        states,
        next_states,
        actions,
        rewards, 
        discount,
        terminals,
        network,
        current_q,
        tau,
    ) as math.Matrix;
    let delta_mat = math.matrix(math.zeros([batch_size, network.num_actions]))
    for (let i = 0; i < batch_size; i ++) {
        let t_index;
        if (terminals.size().length == 1) {
            t_index = [i]
        } else t_index = [i, 0]
        let action_index = actions.get(t_index)
        math.subset(delta_vec, math.index(i, action_index), 1)
    }

    let td_update = network.get_TD_update(states, delta_mat)
    let weights = optimizer.update_weights(network.get_weights(), td_update)
    network.set_weights(weights)
}

export class Agent implements BaseAgent {
    replay_buffer: ReplayBuffer
    network: ActionValueNetwork
    optimizer: AdamOptimizer
    num_replay: number
    discount: number
    tau: number
    last_state: Observation
    last_action: number
    sum_rewards: number
    episode_steps: number
    num_actions: number
    agent_init(agent_info: any) {
        this.replay_buffer = new ReplayBuffer(
            agent_info.replay_buffer_size,
            agent_info.minimatch_size,
        )
        this.network = new ActionValueNetwork(
            agent_info.network_config,
        )
        this.optimizer = new AdamOptimizer(this.network.layer_sizes, agent_info.optimizer_info)
        this.num_actions = this.network.num_actions
        this.num_replay = agent_info.num_replay
        this.discount = agent_info.gamma
        this.tau = agent_info.tau
        this.sum_rewards = 0
        this.episode_steps = 0   
    }

    policy(state) {
        let action_values = this.network.get_action_values(state)
        let probs_batch = softmax(action_values, this.tau)
        return this.random_choice(probs_batch)
    }

    random_choice(probas){
        let num = Math.random();
        let last_index = probas.size()[1] - 1;
        var s = 0;
        for (var i = 0; i < last_index; i++) {
            s += probas.get([0, i]);
            if (num < s) {
                    return i;
            }
        }
        return last_index;
    }   

    agent_start(observation: number | number[]): number {
        let state;
        let saved_observation;
        if (Array.isArray(observation)) {
            state = math.matrix([observation])
            saved_observation = observation
        } else {
            state = math.matrix([[observation]])
            saved_observation = [observation]
        }
        this.sum_rewards = 0
        this.episode_steps = 0
        this.last_state = saved_observation
        this.last_action = this.policy(state)
        return this.last_action
    }
    agent_step(reward: number, observation: number | number[]): number {
        let state;
        let saved_observation;
        if (Array.isArray(observation)) {
            state = math.matrix([observation])
            saved_observation = observation
        } else {
            state = math.matrix([[observation]])
            saved_observation = [observation]
        }
        this.sum_rewards += reward
        this.episode_steps += 1
        let action = this.policy(state)
        this.replay_buffer.append(this.last_state, this.last_action, reward, false, saved_observation)
        if (this.replay_buffer.size() >= this.replay_buffer.minimatch_size) {
            let current_q = this.network.clone()
            for (let i = 0; i < this.num_replay; i ++) {
                let experiences = this.replay_buffer.sample()
                optimize_network(experiences, this.discount, this.optimizer, this.network, current_q, this.tau)
            }
        }
        this.last_state = saved_observation
        this.last_action = action
        return action
    }

    agent_end(reward: number) {
        this.sum_rewards += reward
        this.episode_steps += 1
        let observation;
        if (Array.isArray(this.last_state)) {
            observation = math.matrix(math.zeros(math.matrix(this.last_state).size())).toJSON().data
        } else observation = [0];   

        this.replay_buffer.append(this.last_state, this.last_action, reward, false, observation)
        if (this.replay_buffer.size() > this.replay_buffer.minimatch_size) {
            let current_q = this.network.clone()
            for (let i = 0; i < this.num_replay; i ++) {
                let experiences = this.replay_buffer.sample()
                optimize_network(experiences, this.discount, this.optimizer, this.network, current_q, this.tau)
            }
        }
    }

    agent_cleanup() {
       this.last_action = undefined
       this.last_state = undefined
    }
    agent_message(message: string) {
        if (message == "get_sum_reward") {
            return this.sum_rewards;
        }
    }  
}

