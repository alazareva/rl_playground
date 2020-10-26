import {TDAgent} from '../src/cliffwalking';

//import * as tf from '@tensorflow/tfjs';
//require('@tensorflow/tfjs-node');


describe("CliffWalkingEnvironment", () => {
    it("correcly pick random action", () => {
        let agent = new TDAgent();
        let policy = [
            [9, 0],
            [0, 10],
        ];
        agent.agent_init({policy: policy, discount: 0.5, step_size: 2});
        expect(agent.random_choice([0])).toEqual(0);
        expect(agent.random_choice([1])).toEqual(1);
    });

    it('should correctly do a td update', () => {
        let agent = new TDAgent()
        let policy = [
            [1, 0],
            [1, 0],
        ];
        agent.agent_init({
            policy: policy,
            discount: 0.99,
            step_size: 0.1,
        });
        agent.values = [0, 1];
        agent.agent_start([0]);
        let reward = -1;
        let next_state = [1];
        agent.agent_step(reward, next_state);
        expect(agent.values[0]).toBeCloseTo(-0.001)
        expect(agent.values[1]).toBeCloseTo(1)
    });

    it('should correctly do a td update when terminal', () => {
        let agent = new TDAgent()
        let policy = [
            [1, 0],
            [1, 0],
        ];
        agent.agent_init({
            policy: policy,
            discount: 0.99,
            step_size: 0.1,
        });
        agent.values = [0];
        agent.agent_start([0]);
        let reward = -100;
        let next_state = [0];
        agent.agent_step(reward, next_state);
        expect(agent.values[0]).toBeCloseTo(-10)
    });
});