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
import { get_value } from "./td_nn";

function row(matrix, index) {
    var cols = math.size(matrix).valueOf()[1];
    return math.flatten(math.subset(matrix, math.index(index, math.range(0, cols))));
  }

export class ExpectedSarsaAgent implements BaseAgent {
    num_actions: number
    num_states: number
    epsilon: number
    step_size: number
    discount: number
    prev_observation: number
    prev_action: number

    q: math.Matrix

    agent_init(agent_info: any) {
        this.num_states = agent_info.num_states
        this.num_actions = agent_info.num_actions
        this.epsilon = agent_info.epsilon
        this.step_size = agent_info.step_size
        this.discount = agent_info.discount

        this.q = math.matrix(math.zeros([this.num_states, this.num_actions]))
    }

    argmax(arr) {
        return arr._data.indexOf(math.max(arr))
    }
    agent_start(observation: number): number {
        if (Array.isArray(observation)) {
            observation = observation[0]
        }
        let current_q = row(this.q, observation)
        if (Math.random() < this.epsilon) {
            this.prev_action = Math.floor(Math.random() * this.num_actions)
        }
        else {
            this.prev_action = this.argmax(current_q);
        }
        this.prev_observation = observation
        return this.prev_action;

    }
    agent_step(reward: number, observation: number): number {
        if (Array.isArray(observation)) {
            observation = observation[0]
        }
        let current_q = row(this.q, observation)
        let action;
        if (Math.random() < this.epsilon) {
            action = Math.floor(Math.random() * this.num_actions)
        }
        else {
            action = this.argmax(current_q);
        }

        let max_current_q = math.max(current_q)
        let best_qs = math.map(current_q, (v) => v == max_current_q? 1: 0)
        let prob_non_best = this.epsilon * 1 / this.num_actions
        let prob_best = (1 - this.epsilon) / math.sum(best_qs) + this.epsilon * 1 / this.num_actions
        let prob = Array(this.num_actions)
        for (let i = 0; i < this.num_actions; i++) {
            if (best_qs.get([i]) == 0) {
                prob[i] = prob_non_best
            } else {
                prob[i] = prob_best
            }
        }
        let target = math.multiply(current_q, prob)

        let Q_SA = this.q.get([this.prev_observation, this.prev_action])
        this.q = math.subset(
            this.q, 
            math.index(this.prev_observation, this.prev_action), 
            Q_SA + this.step_size * (reward + this.discount * target - Q_SA))

        this.prev_observation = observation
        this.prev_action = action
        return action;
    }
    agent_end(reward: number) {
        let Q_SA = this.q.get([this.prev_observation, this.prev_action])
        this.q = math.subset(
            this.q, 
            math.index(this.prev_observation, this.prev_action), 
            Q_SA + this.step_size * (reward - Q_SA))
    }

    set_epsilon(eps) {
        this.epsilon = eps;
    }

    set_step_size(step_size) {
        this.step_size = step_size;
    }

    set_discount(discount) {
        this.discount = discount;
    }

    agent_cleanup() {
        throw new Error("Method not implemented.");
    }
    agent_message(message: string) {
        if (message == 'get_Q') return this.get_q()
    }

    get_q() {
        return this.q;
    }

}