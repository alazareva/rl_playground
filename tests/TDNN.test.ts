import {get_value, get_gradient, SDGOptimizer, one_hot, TDNNAgent} from '../src/td_nn';
import * as math from 'mathjs';
const assert = require('assert')
const test_data = require('./test_data/td_nn_agent/asserts.json');

describe("TDNNAgent", () => {
    it("correctly one hot encodes", () => {
        let state = 2;
        let num_states = 5;
        let actual = one_hot(state, num_states);
        let expected = math.matrix([[0, 1, 0, 0, 0]]);
        assert.deepStrictEqual(expected, actual);
    });

    it("correctly get value", () => {
        let s = math.matrix([[0, 0, 0, 1, 0]])
        let layers = [
            {
                W: math.matrix(
                    [
                        [1, 1], 
                        [2, 2], 
                        [3, 4], 
                        [4, 4], 
                        [5, 5]
                ]
                    ),
                b: math.matrix(
                    [[1, 1]]
                    )
            },
            {
                W: math.matrix([
                    [1], 
                    [2]
                ]),
                b: math.matrix([[1]])
            }
        ]
        let estimated_value = get_value(s, layers);
        expect(estimated_value.get([0, 0])).toBeCloseTo(16)

    });

    it("correctly get the gradients", () => {
        let s = math.matrix([[0, 0, 0, 1, 0]])
        let layers = [
            {
                W: math.matrix(
                    [
                        [1, 1], 
                        [2, 2], 
                        [3, 4], 
                        [4, 4], 
                        [5, 5]
                ]
                    ),
                b: math.matrix(
                    [[1, 1]]
                    )
            },
            {
                W: math.matrix([
                    [1], 
                    [2]
                ]),
                b: math.matrix([[1]])
            }
        ]
        let gradients = get_gradient(s, layers);
        let expected_W0 = math.matrix([
            [ 0.,  0.],
            [ 0.,  0.],
            [ 0.,  0.],
            [ 1.,  2.],
            [ 0.,  0.]
        ]);
        let expected_b0 = math.matrix([[1, 2]]);
        let expected_W1 = math.matrix([[5], [5]]);
        let expected_b1 = math.matrix([[1]]);
        assert.deepStrictEqual(gradients[0].W ,expected_W0);
        assert.deepStrictEqual(gradients[0].b ,expected_b0);
        assert.deepStrictEqual(gradients[1].W ,expected_W1);
        assert.deepStrictEqual(gradients[1].b ,expected_b1);
    });

    it("correctly init the agent", () => {
        let agent_info = {
            num_states: 5,
            num_hidden_layer: 1,
            num_hidden_units: 2,
            step_size: 0.25,
            discount_factor: 0.9,
            beta_m: 0.9,
            beta_v: 0.99,
            epsilon: 0.0001,
       }
       let test_agent = new TDNNAgent()
       test_agent.agent_init(agent_info)
       assert.deepStrictEqual(test_agent.layer_size, [agent_info.num_states, agent_info.num_hidden_units, 1]);
       assert.deepStrictEqual(test_agent.weights[0].W.size(), [agent_info.num_states, agent_info.num_hidden_units])
       assert.deepStrictEqual(test_agent.weights[0].b.size(), [1, agent_info.num_hidden_units])
       assert.deepStrictEqual(test_agent.weights[1].W.size(), [agent_info.num_hidden_units, 1])
       assert.deepStrictEqual(test_agent.weights[1].b.size(), [1, 1])

    });

    it("correctly start the agent", () => {
        let agent_info = {
            num_states: 500,
            num_hidden_layer: 1,
            num_hidden_units: 100,
            step_size: 0.1,
            discount_factor: 1.0,
            beta_m: 0.9,
            beta_v: 0.99,
            epsilon: 0.0001,
       }
       let test_agent = new TDNNAgent()
       test_agent.agent_init(agent_info)

       let state = 250;
       test_agent.agent_start(state)
       expect(test_agent.last_state).toBe(250)
    });

    it("correctly perform agent step", () => {
        let agent_info = {
            num_states: 5,
            num_hidden_layer: 1,
            num_hidden_units: 2,
            step_size: 0.1,
            discount_factor: 1.0,
            beta_m: 0.9,
            beta_v: 0.99,
            epsilon: 0.0001,
       }
       let test_agent = new TDNNAgent()
       test_agent.agent_init(agent_info)
       let agent_initial_weight = test_data['agent_step_initial_weights'];
       test_agent.weights[0].W = math.matrix(agent_initial_weight["W0"]);
       test_agent.weights[0].b = math.matrix(agent_initial_weight["b0"]);
       test_agent.weights[1].W = math.matrix(agent_initial_weight["W1"]);
       test_agent.weights[1].b = math.matrix(agent_initial_weight["b1"]);

       let m_data = test_data['agent_step_initial_m']
       test_agent.optimizer.m[0]["W"] = math.matrix(m_data["W0"]);
       test_agent.optimizer.m[0]["b"] = math.matrix(m_data["b0"]);
       test_agent.optimizer.m[1]["W"] = math.matrix(m_data["W1"]);
       test_agent.optimizer.m[1]["b"] = math.matrix(m_data["b1"]);

       let v_data = test_data['agent_step_initial_v']
       test_agent.optimizer.v[0]["W"] = math.matrix(v_data["W0"])
       test_agent.optimizer.v[0]["b"] = math.matrix(v_data["b0"])
       test_agent.optimizer.v[1]["W"] = math.matrix(v_data["W1"])
       test_agent.optimizer.v[1]["b"] = math.matrix(v_data["b1"])

       let start_state = 3
       test_agent.agent_start(start_state)
       let reward = 10.0
       let next_state = 1
       test_agent.agent_step(reward, next_state)
       
       let agent_updated_weight_answer = test_data['agent_step_updated_weights']
       assert.deepStrictEqual(test_agent.weights[0].W, math.matrix(agent_updated_weight_answer["W0"]))
       assert.deepStrictEqual(test_agent.weights[0].b, math.matrix(agent_updated_weight_answer["b0"]))
       assert.deepStrictEqual(test_agent.weights[1].W, math.matrix(agent_updated_weight_answer["W1"]))
       assert.deepStrictEqual(test_agent.weights[1].b, math.matrix(agent_updated_weight_answer["b1"]))

       expect(test_agent.last_state).toBe(1)
    });

    it("correctly perform agent end", () => {
        let agent_info = {
            num_states: 5,
            num_hidden_layer: 1,
            num_hidden_units: 2,
            step_size: 0.1,
            discount_factor: 1.0,
            beta_m: 0.9,
            beta_v: 0.99,
            epsilon: 0.0001,
       }
       let test_agent = new TDNNAgent()
       test_agent.agent_init(agent_info)
       let agent_initial_weight = test_data['agent_end_initial_weights'];
       test_agent.weights[0].W = math.matrix(agent_initial_weight["W0"]);
       test_agent.weights[0].b = math.matrix(agent_initial_weight["b0"]);
       test_agent.weights[1].W = math.matrix(agent_initial_weight["W1"]);
       test_agent.weights[1].b = math.matrix(agent_initial_weight["b1"]);

       let m_data = test_data['agent_step_initial_m']
       test_agent.optimizer.m[0]["W"] = math.matrix(m_data["W0"]);
       test_agent.optimizer.m[0]["b"] = math.matrix(m_data["b0"]);
       test_agent.optimizer.m[1]["W"] = math.matrix(m_data["W1"]);
       test_agent.optimizer.m[1]["b"] = math.matrix(m_data["b1"]);

       let v_data = test_data['agent_step_initial_v']
       test_agent.optimizer.v[0]["W"] = math.matrix(v_data["W0"])
       test_agent.optimizer.v[0]["b"] = math.matrix(v_data["b0"])
       test_agent.optimizer.v[1]["W"] = math.matrix(v_data["W1"])
       test_agent.optimizer.v[1]["b"] = math.matrix(v_data["b1"])

       let start_state = 4
       test_agent.agent_start(start_state)
       let reward = 10.0
       test_agent.agent_end(reward)
       
       let agent_updated_weight_answer = test_data['agent_end_updated_weights']
       assert.deepStrictEqual(test_agent.weights[0].W, math.matrix(agent_updated_weight_answer["W0"]))
       assert.deepStrictEqual(test_agent.weights[0].b, math.matrix(agent_updated_weight_answer["b0"]))
       assert.deepStrictEqual(test_agent.weights[1].W, math.matrix(agent_updated_weight_answer["W1"]))
       assert.deepStrictEqual(test_agent.weights[1].b, math.matrix(agent_updated_weight_answer["b1"]))

    });


    it("correctly correctly use SGD optimizer", () => {
        let s = math.matrix([[0, 0, 0, 1, 0]])
        let layers = [
            {
                W: math.matrix(
                    [
                        [1, 1], 
                        [2, 2], 
                        [3, 4], 
                        [4, 4], 
                        [5, 5]
                ]
                    ),
                b: math.matrix(
                    [[1, 1]]
                    )
            },
            {
                W: math.matrix([
                    [1], 
                    [2]
                ]),
                b: math.matrix([[1]])
            }
        ]

        let grads = [
            {
                W: math.matrix(
                    [
                    [0.01, 0.01], 
                    [-0.01, -0.01], 
                    [-0.01, -0.01],
                    [0.01,  0.01], 
                    [0.01,  0.01]
                ]),
                b: math.matrix(
                    [[0.11, 0.11]]
                    )
            },
            {
                W: math.matrix([
                    [0.1], 
                    [0.2]
                ]),
                b: math.matrix([[0.1]])
            },
        ]
        let test_sgd = new SDGOptimizer()
        let optimizer_info = {step_size: 0.3}
        test_sgd.optimizer_init(optimizer_info)
        let updated_weights = test_sgd.update_weights(layers, grads);
        let expected_W0 = math.matrix([
            [ 1.003, 1.003],
            [ 1.997, 1.997],
            [ 2.997, 3.997],
            [ 4.003, 4.003],
            [ 5.003, 5.003],
        ]);
        let expected_b0 = math.matrix([[1.033, 1.033]]);
        let expected_W1 = math.matrix([[1.03], [2.06]]);
        let expected_b1 = math.matrix([[1.03]]);
        assert.deepStrictEqual(updated_weights[0].W ,expected_W0);
        assert.deepStrictEqual(updated_weights[0].b ,expected_b0);
        assert.deepStrictEqual(updated_weights[1].W ,expected_W1);
        assert.deepStrictEqual(updated_weights[1].b ,expected_b1);
    });
});
