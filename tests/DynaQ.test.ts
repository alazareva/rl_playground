const assert = require('assert')
import * as math from 'mathjs';
import {DynaQAgent} from '../src/dynaQAgent';


describe("QLearningAgent", () => {
    it("correctly update model", () => {
        let agent_info = {
            num_actions: 4,
            num_states: 3,
            epsilon: 0.1,
            step_size: 0.1,
            discount: 1.0,
        }

        let agent = new DynaQAgent();
        agent.agent_init(agent_info);
        agent.update_model(0, 2, 0, 1);
        agent.update_model(2, 0, 1, 1);
        agent.update_model(0, 3, 1, 2);
        let expected = {
            0: {
                2: [0, 1],
                3: [1, 2]
            },
            2: {
                0: [1, 1]
            }
        }
        expect(agent.model).toEqual(expected);
    });

    it("correctly run the planning step", () => {
        let agent_info = {
            num_actions: 4,
            num_states: 3,
            epsilon: 0.1,
            step_size: 0.1,
            discount: 1.0,
            planning_steps: 4,
        }

        let agent = new DynaQAgent();
        agent.agent_init(agent_info);
        agent.update_model(0, 2, 1, 1);
        agent.update_model(2, 0, 1, 1);
        agent.update_model(0, 3, 0, 1);
        agent.update_model(0, 1, -1, 1);
        agent.planning_step()
        let expected = {
            0: {
                2: [1, 1],
                3: [0, 1],
                1: [-1, 1]
            },
            2: {
                0: [1, 1]
            }
        }
        expect(agent.model).toEqual(expected);
    });
        it("correctly start", () => {
        let agent_info = {
            num_actions: 4,
            num_states: 3,
            epsilon: 0.1,
            step_size: 0.1,
            discount: 1.0,
            planning_steps: 10,
        }

        let agent = new DynaQAgent();
        agent.agent_init(agent_info);
        let action = agent.agent_start(0);
        let expected_model = {}
        expect(agent.model).toEqual(expected_model);
        assert.deepStrictEqual(math.matrix(math.zeros([3, 4])), agent.q);
    });

    it("correctly perform agent step", () => {
        let agent_info = {
            num_actions: 4,
            num_states: 3,
            epsilon: 0.1,
            step_size: 0.1,
            discount: 1.0,
            planning_steps: 2,
        }

        let agent = new DynaQAgent();
        agent.agent_init(agent_info);
        agent.agent_start(0);
        agent.agent_step(1, 2);
        agent.agent_step(0, 1);

        expect(math.sum(agent.q)).toBeGreaterThan(0)
    });

    it("correctly perform agent end", () => {
        let agent_info = {
            num_actions: 4,
            num_states: 3,
            epsilon: 0.1,
            step_size: 0.1,
            discount: 1.0,
            planning_steps: 2,
        }

        let agent = new DynaQAgent();
        agent.agent_init(agent_info);
        agent.agent_start(0);
        agent.agent_step(1, 2);
        agent.agent_step(0, 1);
        agent.agent_end(1)

        expect(math.sum(agent.q)).toBeGreaterThan(0)
    });
});