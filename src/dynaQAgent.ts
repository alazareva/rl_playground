import { 
    BaseEnvironment, 
    Observation, 
    Reward,
    Action,
    Terminal,
    BaseAgent,
} from "./rl_base";
import { RLGlue } from './rl_glue';
import {row, argmax} from './qlearning';
import * as math from 'mathjs';
import { random, re } from "mathjs";

export class DynaQAgent implements BaseAgent {
    num_states: number;
    num_actions: number;
    gamma: number;
    step_size: number;
    epsilon: number;
    planning_steps: number;
    q: math.Matrix;
    actions: number[];
    past_action: number;
    past_state: number;
    model: object;

    agent_init(agent_info) {
        this.num_states = agent_info.num_states;
        this.num_actions = agent_info.num_actions;
        this.gamma = agent_info.discount;
        this.step_size = agent_info.step_size;
        this.epsilon = agent_info.epsilon;
        this.planning_steps = agent_info.planning_steps || 10;

        this.q = math.matrix(math.zeros([this.num_states, this.num_actions]))
        this.actions = Array.from(new Array(this.num_actions), (x, i) => i);​​​​​​
        this.past_action = -1
        this.past_state = -1
        this.model = {};
    }

    update_model(past_state, past_action, state, reward) {
        let current = (past_state in this.model) ? this.model[past_state] : {};
        current[past_action] = [state, reward]
        this.model[past_state] = current;
    }

    random_key(obj){
        return parseInt(Object.keys(obj)[Math.floor(Math.random() * Object.keys(obj).length)]);
    }

    planning_step() {
        for (let i = 0; i < this.planning_steps; i++) {
            let random_state = this.random_key(this.model);
            let random_action = this.random_key(this.model[random_state]);
            let [next_s, next_r] = this.model[random_state][random_action];

            let Q_SA = this.q.get([random_state, random_action])
            if (next_s != -1) {
                let current_q = row(this.q, next_s)
                let q_max = this.q.get([next_s, argmax(current_q)])
                this.q = math.subset(
                    this.q, 
                    math.index(random_state, random_action), 
                    Q_SA + this.step_size * (next_r + this.gamma * q_max - Q_SA))
            } else {
                this.q = math.subset(
                    this.q, 
                    math.index(random_state, random_action), 
                    Q_SA + this.step_size * (next_r - Q_SA))
            }   
        }
    }

    choose_action_egreedy(observation){
        let action;
        if (Math.random() < this.epsilon) {
            action = Math.floor(Math.random() * this.num_actions)
        }
        else {
            let current_q = row(this.q, observation)
            action = argmax(current_q);
        }
        return action;
    }

    agent_start(observation: Observation): number {
        if (Array.isArray(observation)) {
            observation = observation[0]
        }
        let action = this.choose_action_egreedy(observation)
        this.past_state = observation;
        this.past_action = action;
        return this.past_action;
    }
    agent_step(reward: number, observation: Observation): number {
        if (Array.isArray(observation)) {
            observation = observation[0]
        }
        let current_q = row(this.q, observation)
        let Q_SA = this.q.get([this.past_state, this.past_action])
        let q_max = this.q.get([observation, argmax(current_q)])
        this.q = math.subset(
            this.q, 
            math.index(this.past_state, this.past_action), 
            Q_SA + this.step_size * (reward + this.gamma * q_max - Q_SA))
        this.update_model(this.past_state, this.past_action, observation, reward);
        this.planning_step();
        let action = this.choose_action_egreedy(observation);
        this.past_state = observation;
        this.past_action = action;
        return this.past_action;
    }
    agent_end(reward: number) {
        let Q_SA = this.q.get([this.past_state, this.past_action]);
        this.q = math.subset(
            this.q, 
            math.index(this.past_state, this.past_action), 
            Q_SA + this.step_size * (reward - Q_SA))
        this.update_model(this.past_state, this.past_action, -1, reward)
        this.planning_step();
    }
    agent_cleanup() {
        this.past_state = -1;
        this.past_action = -1;
    }
    agent_message(message: string) {
        if (message == 'get_Q') return this.get_q()
    }

    get_q() {
        return this.q;
    }

    set_epsilon(eps) {
        this.epsilon = eps;
    }

    set_step_size(step_size) {
        this.step_size = step_size;
    }

    set_discount(discount) {
        this.gamma = discount;
    }
}