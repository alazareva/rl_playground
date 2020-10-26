import * as math from 'mathjs';
import { 
    BaseEnvironment, 
    Observation, 
    Reward,
    Action,
    Terminal,
    BaseAgent,
} from "./rl_base";


function agent_policy() {
    return Math.random() < 0.5 ? 1 : 0;
}

export function get_state_feature(num_states_in_group, num_groups, state) {
    let one_hot_vector = new Array(num_groups).fill(0); 
    one_hot_vector[math.floor((state - 1) / num_states_in_group)] = 1
    return one_hot_vector
}

export function one_hot(state, num_states): math.Matrix {
    let one_hot_vector = math.zeros(1, num_states) as math.Matrix;
    one_hot_vector = one_hot_vector.subset(math.index(0, math.floor(state-1)), 1);
    return one_hot_vector;
}

export class TDAgentStateAgg implements BaseAgent {
    discount: number
    step_size: number
    weights: number[]
    num_actions: number
    num_states: number
    last_state: number
    num_groups: number
    all_state_features: number[][]
    last_action: number

    agent_init(agent_info) {
        this.discount = agent_info.discount;
        this.step_size = agent_info.step_size;
        this.num_groups = agent_info.num_groups;
        let num_states_in_group = Math.floor(this.num_states / this.num_groups)
        let all_state_features = Array.from(Array(this.num_states).keys()).map( state => 
           get_state_feature(num_states_in_group, this.num_groups, state + 1))
        this.weights = new Array(this.num_groups).fill(0);
    }
    agent_start(observation: Observation): Action {
        if (Array.isArray(observation)) {
            observation = observation[0]
        }
        this.last_action = agent_policy();
        this.last_state = observation;
        return this.last_action; 
    }

    dot(v1: number[], v2: number[]) {
        let len = math.min(v1.length, v2.length)
        var result = 0;
        for (var i = 0; i < len; i++) {
              result += v1[i] * v2[i];
            }
        return result;
    }

    agent_step(reward: number, observation: Observation): Action {
        if (Array.isArray(observation)) {
            observation = observation[0]
        }
        let current_state_feature = this.all_state_features[observation - 1]
        let last_state_feature = this.all_state_features[this.last_state - 1]
        let current_dot_w = this.dot(current_state_feature, this.weights)
        let last_dot_w = this.dot(last_state_feature, this.weights)
        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] = this.weights[i] + this.step_size * 
            (reward + this.discount * current_dot_w[i] - last_dot_w[i]) * last_state_feature[i]
        }
        this.last_state = observation
        this.last_action = agent_policy()
        return this.last_action
    }

    agent_end(reward: number) {
        let last_state_feature = this.all_state_features[this.last_state - 1]
        let last_dot_w = this.dot(last_state_feature, this.weights)
        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] = this.weights[i] + this.step_size * 
            (reward - last_dot_w[i]) * last_state_feature[i]
        }
    }

    agent_cleanup() {
        this.last_state = undefined;
    }
    agent_message(message: string): any {
        if (message == 'get state value') {
            return this.all_state_features.map(arr => this.dot(arr, this.weights))
        }
    }
}