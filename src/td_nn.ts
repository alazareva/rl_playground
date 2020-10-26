import * as math from 'mathjs';
import {BaseOptimizer} from './optimizer'
import { range } from 'mathjs';
import { BaseAgent, Observation } from './rl_base';

export function mat_mul(m1: math.Matrix, m2: math.Matrix): math.Matrix{
    // TODO add the other stuff
    return math.multiply(m1, m2);
};

export interface Layer {
    W: math.Matrix;
    b: math.Matrix;
}

/**
 * Compute value of input s given the weights of a neural network
 * @param s 
 * @param weights 
 */

export function get_value(s: math.Matrix, weights: Layer[]) {
    let layer0 = weights[0];
    let layer1 = weights[1];
    var v = math.add(mat_mul(s, layer0.W), layer0.b) as math.Matrix;
    v = math.map(v, function(value) {
        return math.max(value, 0)
      }) 
    v = math.add(mat_mul(v, layer1.W), layer1.b) as math.Matrix;
    return v;
};

export interface Grads {
    W: math.Matrix
    b: math.Matrix
}

/**
 * Given inputs s and weights, return the gradient of v with respect to the weights
 * @param s 
 * @param weights 
 */
export function get_gradient(s, weights: Layer[]): Grads[] {
    let grads: Grads[] = new Array(2);
    let layer0 = weights[0];
    let layer1 = weights[1];
    let v = math.add(mat_mul(s, layer0.W), layer0.b) as math.Matrix;
    let x = math.map(v, function(value) {
        return math.max(value, 0)
      })

    grads[1] = {
        W: math.transpose(x),
        b: math.matrix([[1]])
    }
    let x_mask = math.map(x, function(v) {
        return v > 0 ? 1 : 0;
    })
    let b0 = math.dotMultiply(math.transpose(layer1.W), x_mask) as math.Matrix;
    grads[0] = {
        W: mat_mul(math.transpose(s), b0),
        b: b0

    }
    return grads;
};


export class SDGOptimizer implements BaseOptimizer {
    step_size: number
    optimizer_init(optimizer_info: any) {
       this.step_size = optimizer_info.step_size;
    }
    update_weights(weights: Layer[], g: Grads[]) {
        let updated: Layer[] = new Array(weights.length);
        for (let i = 0; i < weights.length; i++) {
            let layer = {
                W: math.add(weights[i].W , math.multiply(this.step_size, g[i].W)) as math.Matrix,
                b: math.add(weights[i].b , math.multiply(this.step_size, g[i].b)) as math.Matrix,
            }
            updated[i] = layer;
        }
        return updated;
    }
}

class AdamOptimizer implements BaseOptimizer {
    num_states: number
    num_hidden_layer: number
    num_hidden_units: number
    step_size: number
    beta_m: number
    beta_v: number
    epsilon: number
    layer_size: number[]
    m: Layer[]
    v: Layer[]
    beta_m_product: number
    beta_v_product: number

    optimizer_init(optimizer_info: any) {
        this.num_states = optimizer_info.num_states;
        this.num_hidden_layer = optimizer_info.num_hidden_layer;
        this.num_hidden_units = optimizer_info.num_hidden_units;
        this.step_size = optimizer_info.step_size;
        this.beta_m = optimizer_info.beta_m;
        this.beta_v = optimizer_info.beta_v;
        this.epsilon = optimizer_info.epsilon;

        this.layer_size = [this.num_states, this.num_hidden_units, 1]

        this.m = new Array(this.num_hidden_layer + 1)
        this.v = new Array(this.num_hidden_layer + 1)

        for (let i = 0; i <= this.num_hidden_layer; i++) {
            this.m[i] = {
                W: math.zeros(this.layer_size[i], this.layer_size[i + 1]) as math.Matrix,
                b: math.zeros(1, this.layer_size[i + 1]) as math.Matrix
            }
            this.v[i] = {
                W: math.zeros(this.layer_size[i], this.layer_size[i + 1]) as math.Matrix,
                b: math.zeros(1, this.layer_size[i + 1]) as math.Matrix
            }
        }

        this.beta_m_product = this.beta_m
        this.beta_v_product = this.beta_v

    }

    update_weights(weights: Layer[], g: Grads[]) {
        for (let i = 0; i < weights.length; i++){
            this.m[i] = {
                W: math.add(math.multiply(this.beta_m,  this.m[i].W ), math.multiply(1 - this.beta_m, g[i].W)) as math.Matrix,
                b:  math.add(math.multiply(this.beta_m,  this.m[i].b ), math.multiply(1 - this.beta_m, g[i].b)) as math.Matrix
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

export function one_hot(state, num_states): math.Matrix {
    let one_hot_vector = math.zeros(1, num_states) as math.Matrix;
    one_hot_vector = one_hot_vector.subset(math.index(0, math.floor(state-1)), 1);
    return one_hot_vector;
}

export class TDNNAgent implements BaseAgent {

    num_states: number;
    num_hidden_layer: number;
    num_hidden_units: number;
    discount_factor: number;
    layer_size: number[];
    weights: Layer[];
    optimizer: AdamOptimizer;
    last_state;
    last_action;

    agent_init(agent_info: any) {
        this.num_states = agent_info.num_states;
        this.num_hidden_layer = agent_info.num_hidden_layer;
        this.num_hidden_units = agent_info.num_hidden_units;
        this.discount_factor = agent_info.discount_factor;
        this.layer_size = [this.num_states, this.num_hidden_units]

        this.layer_size = [this.num_states].concat(Array(this.num_hidden_layer).fill(this.num_hidden_units)).concat([1])
        this.weights = new Array(this.num_hidden_layer+1);

        for (let i = 0; i <= this.num_hidden_layer; i++) {
            let inputs = this.layer_size[i];
            let outputs = this.layer_size[i + 1]
            let W = math.matrix(math.random([inputs, outputs], math.sqrt(2/inputs)));
            let b = math.matrix(math.random([1, outputs], math.sqrt(2/inputs)))
            this.weights[i] = {
                W: W,
                b: b
            }

        }
        this.optimizer = new AdamOptimizer();
        let optimizer_info = {
            num_states: this.num_states,
            num_hidden_layer: this.num_hidden_layer,
            num_hidden_units: this.num_hidden_units,
            step_size: agent_info.step_size,
            beta_m: agent_info.beta_m,
            beta_v: agent_info.beta_v,
            epsilon: agent_info.epsilon
        }
        this.optimizer.optimizer_init(optimizer_info)
    }

    agent_start(observation: Observation): number {
        this.last_state = observation;
        this.last_action = this.agent_policy(observation);
        return this.last_action
    }

    agent_step(reward: number, observation: Observation): number {

        let state_one_hot = one_hot(observation, this.num_states);
        let prev_state_one_hot = one_hot(this.last_state, this.num_states);
        let delta = reward + this.discount_factor * get_value(state_one_hot, this.weights).get([0, 0]) - get_value(prev_state_one_hot, this.weights).get([0, 0]);
        let grads = get_gradient(prev_state_one_hot, this.weights)
        let g: Grads[] = new Array(this.num_hidden_layer + 1);
        for (let i = 0; i < this .num_hidden_layer + 1; i ++) {
            let W = this.weights[i].W;
            let b = this.weights[i].b;
            g[i] = {
                W: math.multiply(delta, grads[i].W) as math.Matrix,
                b: math.multiply(delta, grads[i].b) as math.Matrix,
            }
        }
        this.weights = this.optimizer.update_weights(this.weights, g);
        this.last_state = observation;
        this.last_action = this.agent_policy(observation);

        return this.last_action;

    }
    agent_end(reward: number) {
        let prev_state_one_hot = one_hot(this.last_state, this.num_states);
        let delta = reward - get_value(prev_state_one_hot, this.weights).get([0, 0])
        let grads = get_gradient(prev_state_one_hot, this.weights)
        let g: Grads[] = new Array(this.num_hidden_layer + 1);
        for (let i = 0; i < this .num_hidden_layer + 1; i ++) {
            let W = this.weights[i].W;
            let b = this.weights[i].b;
            g[i] = {
                W: math.multiply(delta, grads[i].W) as math.Matrix,
                b: math.multiply(delta, grads[i].b) as math.Matrix,
            }
        }
        this.weights = this.optimizer.update_weights(this.weights, g)
    }

    agent_cleanup() {
        throw new Error("Method not implemented.");
    }
    agent_message(message: string) {
        if (message == 'get state value'){
            let state_value = math.zeros(this.num_states)
            for (let i = 0; i < this.num_states + 1; i ++) {
                let s = one_hot(i, this.num_states);
                state_value[i - 1] = get_value(s, this.weights)
            }
        return state_value
        }
    }

    agent_policy(state) {
        let num = Math.random();
        return  num < 0.5? 1: 0;  
    } 
}
