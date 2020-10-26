import {ExpectedSarsaAgent} from '../src/expectedSarsa';
const assert = require('assert')
import * as math from 'mathjs';

describe("QLearningAgent", () => {
    it("correctly start agent", () => {
        let agent_info = {
            num_actions: 4,
            num_states: 3,
            epsilon: 0.1,
            step_size: 0.1,
            discount: 1.0,
        }

        let agent = new ExpectedSarsaAgent();
        agent.agent_init(agent_info);
        let action = agent.agent_start(0);
        assert.deepStrictEqual(math.matrix(math.zeros([3, 4])), agent.q);
    });

    it("correctly perform agent step", () => {
        let agent_info = {
            num_actions: 4,
            num_states: 3,
            epsilon: 0.1,
            step_size: 0.1,
            discount: 1.0,
        }

        let agent = new ExpectedSarsaAgent();
        let expected = math.matrix([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        agent.agent_init(agent_info);
        let action1 = agent.agent_start(0);
        let action2 = agent.agent_step(2, 1);
        let action3 = agent.agent_step(0, 0);

        expected = math.subset(
            expected, 
            math.index(0, action1), 
            0.2
            )
        expected = math.subset(
                expected, 
                math.index(1, action2), 
                0.0185
            )

        assert.deepStrictEqual(math.round(agent.q, 5), expected);
    });

    it("correctly perform agent end", () => {
        let agent_info = {
            num_actions: 4,
            num_states: 3,
            epsilon: 0.1,
            step_size: 0.1,
            discount: 1.0,
        }

        let agent = new ExpectedSarsaAgent();
        let expected = math.matrix([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        agent.agent_init(agent_info);
        let action1 = agent.agent_start(0);
        let action2 = agent.agent_step(2, 1);
        agent.agent_end(1);

        expected = math.subset(
            expected, 
            math.index(0, action1), 
            0.2
        )
        expected = math.subset(
            expected, 
            math.index(1, action2), 
            0.1
        )

        assert.deepStrictEqual(math.round(agent.q, 5), expected);
    });

});