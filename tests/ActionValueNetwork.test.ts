import {
    ActionValueNetwork, 
    ReplayBuffer,
    Agent,
    get_td_error,
    optimize_network,
    softmax,
    AdamOptimizer,
} from '../src/action_value_agent';
import {Layer, mat_mul, Grads} from '../src/td_nn'
import * as math from 'mathjs';
const assert = require('assert')
const test_data = require('./test_data/action_value_network/asserts.json');

describe("ActionValueNetwork", () => {
    it("correctly init network", () => {
        let network_config = {
            state_dim: 5,
            num_hidden_units: 20,
            num_actions: 3,
        }
        let test_network = new ActionValueNetwork(network_config)
        expect(test_network.layer_sizes).toEqual([5, 20, 3])
    })

    it("correctly init Adam optimizer", () => {
        let network_config = {
            state_dim: 5,
            num_hidden_units: 2,
            num_actions: 3,
        }
        let optimizer_info = {
            step_size: 0.1,
            beta_m: 0.99,
            beta_v: 0.999,
            epsilon: 0.0001,
        }
        let test_network = new ActionValueNetwork(network_config)
        let test_adam = new AdamOptimizer(test_network.layer_sizes,  optimizer_info);
        expect(test_adam.m[0].W.size()).toEqual([5, 2])
        expect(test_adam.m[0].b.size()).toEqual([1, 2])
        expect(test_adam.m[1].W.size()).toEqual([2, 3])
        expect(test_adam.m[1].b.size()).toEqual([1, 3])
        
        expect(test_adam.v[0].W.size()).toEqual([5, 2])
        expect(test_adam.v[0].b.size()).toEqual([1, 2])
        expect(test_adam.v[1].W.size()).toEqual([2, 3])
        expect(test_adam.v[1].b.size()).toEqual([1, 3])
    })

    it("correctly do an Adam update", () => {
        let network_config = {
            state_dim: 5,
            num_hidden_units: 2,
            num_actions: 3,
        }
        let optimizer_info = {
            step_size: 0.1,
            beta_m: 0.99,
            beta_v: 0.999,
            epsilon: 0.0001,
        }
        let test_network = new ActionValueNetwork(network_config)
        let test_adam = new AdamOptimizer(test_network.layer_sizes,  optimizer_info);
        let m: Layer[] = [
            {
                b: math.matrix([[ 0.14404357,  1.45427351]]),
                W:  math.matrix([
                    [ 1.76405235,  0.40015721],
                    [ 0.97873798,  2.2408932 ],
                    [ 1.86755799, -0.97727788],
                    [ 0.95008842, -0.15135721],
                    [-0.10321885,  0.4105985 ]
                ])
            },
            {
                b: math.matrix([[ 0.3130677 , -0.85409574, -2.55298982]]),
                W: math.matrix([
                    [ 0.76103773,  0.12167502,  0.44386323],
                    [ 0.33367433,  1.49407907, -0.20515826]
                ])
            }
        ]
        let v: Layer[] = [
            {
                b: math.matrix([[ 0.37816252,  0.88778575]]),
                W: math.matrix([
                    [ 0.6536186 ,  0.8644362 ],
                    [ 0.74216502,  2.26975462],
                    [ 1.45436567,  0.04575852],
                    [ 0.18718385,  1.53277921],
                    [ 1.46935877,  0.15494743]])
            },
            {
                b: math.matrix([[ 0.30230275,  1.04855297,  1.42001794]]),
                W: math.matrix([
                    [ 1.98079647,  0.34791215,  0.15634897],
                    [ 1.23029068,  1.20237985,  0.38732682]])
            }
        ]
        let weights = [
            {
                b: math.matrix([[-0.51080514, -1.18063218]]),
                W: math.matrix([
                    [-1.70627019,  1.9507754 ],
                    [-0.50965218, -0.4380743 ],
                    [-1.25279536,  0.77749036],
                    [-1.61389785, -0.21274028],
                    [-0.89546656,  0.3869025 ]])
            },
            {
                b: math.matrix([[-0.67246045, -0.35955316, -0.81314628]]),
                W: math.matrix([
                    [-0.02818223,  0.42833187,  0.06651722],
                    [ 0.3024719 , -0.63432209, -0.36274117]])
            }
        ]
        let g = [
            {
                b: math.matrix([[-1.23482582,  0.40234164]]),
                W: math.matrix([[
                    -1.7262826 ,  0.17742614],
                    [-0.40178094, -1.63019835],
                    [ 0.46278226, -0.90729836],
                    [ 0.0519454 ,  0.72909056],
                    [ 0.12898291,  1.13940068]])
            },
            {
                b: math.matrix([[ 0.90082649,  0.46566244, -1.53624369]]),
                W: math.matrix([
                    [-0.68481009, -0.87079715, -0.57884966],
                    [-0.31155253,  0.05616534, -1.16514984]])
            }
        ]
        test_adam.m = m;
        test_adam.v = v;
        let updated_weights = test_adam.update_weights(weights, g);
        let updated_weights_answer = test_data['update_weights'];
        let delta =  math.subtract(updated_weights[0].W, math.matrix(updated_weights_answer["W0"])) as math.Matrix
        let size = updated_weights[0].W.size()
        expect(math.sum(math.abs(delta))).toBeLessThan(size[0] * size[1] * 0.000001)

        delta =  math.subtract(updated_weights[0].b, math.matrix(updated_weights_answer["b0"])) as math.Matrix
        size = updated_weights[0].b.size()
        expect(math.sum(math.abs(delta))).toBeLessThan(size[0] * size[1] * 0.000001)

        delta =  math.subtract(updated_weights[1].W, math.matrix(updated_weights_answer["W1"])) as math.Matrix
        size = updated_weights[1].W.size()
        expect(math.sum(math.abs(delta))).toBeLessThan(size[0] * size[1] * 0.000001)

        delta =  math.subtract(updated_weights[1].b, math.matrix(updated_weights_answer["b1"])) as math.Matrix
        size = updated_weights[1].b.size()
        expect(math.sum(math.abs(delta))).toBeLessThan(size[0] * size[1] * 0.000001)
    })
    it("correctly compute softmax", () => {
        let action_values = math.matrix([
            [ 1.76405235,  0.40015721,  0.97873798,  2.2408932 ],
            [ 1.86755799, -0.97727788,  0.95008842, -0.15135721]])
        let tau = 0.5;
        let action_probs = softmax(action_values, tau)
        let expected = math.matrix([
            [0.25849645, 0.01689625, 0.05374514, 0.67086216],
            [0.84699852, 0.00286345, 0.13520063, 0.01493741]
        ])
        let delta =  math.subtract(action_probs, expected) as math.Matrix
        let size = expected.size()
        expect(math.sum(math.abs(delta))).toBeLessThan(size[0] * size[1] * 0.000001)

    })

    it("correctly compute td error", () => {
        let data = test_data['get_td_error_1'];
        let states = math.matrix(data["states"])
        let next_states = math.matrix(data["next_states"])
        let actions = math.matrix(data["actions"])
        let rewards = math.matrix(data["rewards"])
        let discount = data["discount"][0]
        let terminals = math.matrix(data["terminals"])
        let tau = 0.001

        let network_config = {
            state_dim: 8,
            num_hidden_units: 512,
            num_actions: 4,
        }

        let network = new ActionValueNetwork(network_config)
        let weight_arrays = data['network_weights']
        let layers = [
            {
                W: math.matrix(weight_arrays[0]['W']),
                b: math.matrix(weight_arrays[0]['b'])
            },
            {
                W: math.matrix(weight_arrays[1]['W']),
                b: math.matrix(weight_arrays[1]['b'])
            }
        ]
        network.set_weights(layers);
        let current_q = new ActionValueNetwork(network_config);
        weight_arrays = data['current_q_weights']
        let q_layers = [
            {
                W: math.matrix(weight_arrays[0]['W']),
                b: math.matrix(weight_arrays[0]['b'])
            },
            {
                W: math.matrix(weight_arrays[1]['W']),
                b: math.matrix(weight_arrays[1]['b'])
            }
        ]
        current_q.set_weights(q_layers);
        let actual = get_td_error(
            states,
            next_states,
            actions,
            rewards,
            discount,
            terminals,
            network,
            current_q,
            tau,
        )

        let expected = math.matrix(data['delta_vec'])
        let delta =  math.subtract(actual, expected) as math.Matrix
        let size = expected.size()
        expect(math.sum(math.abs(delta))).toBeLessThan(size[0] * 0.000001)
    })

    it("correctly optimize network", () => {
        let data = test_data['optimize_network_input_1'];
        let experiences = data['experiences'].map(arr => arr.map(function(a){
            if (Array.isArray(a)) return math.flatten(math.matrix(a))
            return math.matrix([a])
        }))
        let discount = data['discount'][0]
        let tau = 0.001

        let network_config = {
            state_dim: 8,
            num_hidden_units: 512,
            num_actions: 4
        }
        let network = new ActionValueNetwork(network_config)
        let weight_arrays = data['network_weights']
        let layers = get_layers(weight_arrays)
        network.set_weights(layers)

        let current_q = new ActionValueNetwork(network_config);
        weight_arrays = data['current_q_weights']
        let q_layers = get_layers(weight_arrays);
        current_q.set_weights(q_layers);

        let optimizer_config = {
            step_size: 3e-5,
            beta_m: 0.9,
            beta_v: 0.999,
            epsilon: 1e-8,
        }

        let optimizer = new AdamOptimizer(network.layer_sizes, optimizer_config)
        optimizer.m = get_layers(data['optimizer_m']);
        optimizer.v = get_layers(data['optimizer_v']);
        optimizer.beta_m_product = data['optimizer_beta_m_product'][0];
        optimizer.beta_v_product = data["optimizer_beta_v_product"][0];
        optimize_network(experiences, discount, optimizer, network, current_q, tau)
        let actual = network.get_weights()
        let expected = get_layers(test_data['optimize_network_output_1']['updated_weights'])
        compare_matrix(expected[0].W, actual[0].W, 0.01)
        compare_matrix(expected[0].b, actual[0].b, 0.01)
        compare_matrix(expected[1].W, actual[1].W, 0.01)
        compare_matrix(expected[1].b, actual[1].b, 0.01) 
    })

    it("correctly choose action", () => {
        let agent_info = {
            'network_config': {
                'state_dim': 8,
                'num_hidden_units': 256,
                'num_hidden_layers': 1,
                'num_actions': 4
            },
            'optimizer_info': {
                'step_size': 3e-5, 
                'beta_m': 0.9, 
                'beta_v': 0.999,
                'epsilon': 1e-8
            },
            'replay_buffer_size': 32,
            'minimatch_size': 32,
            'num_replay': 4,
            'gamma': 0.99,
            'tau': 1000.0,
            'seed': 0
        }

        let test_agent = new Agent()
        let probas = math.matrix([[1, 0]])
        let action = test_agent.random_choice(probas);
        expect(action).toEqual(0)
        probas = math.matrix([[0, 1]])
        action = test_agent.random_choice(probas);
        expect(action).toEqual(1)
        probas = math.matrix([[0, 0, 1, 0]])
        action = test_agent.random_choice(probas);
        expect(action).toEqual(2)
    })

    it("correctly perform agent step", () => {
        let agent_info = {
            'network_config': {
                'state_dim': 8,
                'num_hidden_units': 256,
                'num_hidden_layers': 1,
                'num_actions': 4
            },
            'optimizer_info': {
                'step_size': 3e-5, 
                'beta_m': 0.9, 
                'beta_v': 0.999,
                'epsilon': 1e-8
            },
            'replay_buffer_size': 32,
            'minimatch_size': 32,
            'num_replay': 4,
            'gamma': 0.99,
            'tau': 1000.0,
            'seed': 0
        }

        let test_agent = new Agent()
        test_agent.agent_init(agent_info)
        let data = test_data['agent_input_1']
        let layers = get_layers(data['network_weights'])
        test_agent.network.set_weights(layers)

        test_agent.optimizer.m = get_layers(data['optimizer_m'])
        test_agent.optimizer.v = get_layers(data['optimizer_v'])
        test_agent.optimizer.beta_m_product = data["optimizer_beta_m_product"]
        test_agent.optimizer.beta_v_product = data["optimizer_beta_v_product"]
        let experiences = data['replay_buffer']
        experiences.forEach(ex => {
            test_agent.replay_buffer.append(ex[0][0], ex[1], ex[2], ex[3], ex[4][0])
        });

        let last_state_array = data["last_state_array"]
        let last_action_array = data["last_action_array"]
        let state_array = data["state_array"]
        let reward_array = data["reward_array"]

        for (let i = 0; i < 5; i++) {
            test_agent.last_state = last_state_array[i][0];
            test_agent.last_action = last_action_array[i];
            let state = state_array[i];
            let reward = reward_array[i]

            test_agent.agent_step(reward, state)
            let output_data = test_data["agent_step_output_" + i]
            compare_matrix(observation_to_matrix([test_agent.last_state]), math.matrix(output_data['last_state']))

        }
    })

    it("correctly perform agent end", () => {
        let agent_info = {
            'network_config': {
                'state_dim': 8,
                'num_hidden_units': 256,
                'num_hidden_layers': 1,
                'num_actions': 4
            },
            'optimizer_info': {
                'step_size': 3e-5, 
                'beta_m': 0.9, 
                'beta_v': 0.999,
                'epsilon': 1e-8
            },
            'replay_buffer_size': 32,
            'minimatch_size': 32,
            'num_replay': 4,
            'gamma': 0.99,
            'tau': 1000.0,
            'seed': 0
        }

        let test_agent = new Agent()
        test_agent.agent_init(agent_info)
        let data = test_data['agent_input_1']
        let layers = get_layers(data['network_weights'])
        test_agent.network.set_weights(layers)

        test_agent.optimizer.m = get_layers(data['optimizer_m'])
        test_agent.optimizer.v = get_layers(data['optimizer_v'])
        test_agent.optimizer.beta_m_product = data["optimizer_beta_m_product"]
        test_agent.optimizer.beta_v_product = data["optimizer_beta_v_product"]
        let experiences = data['replay_buffer']
        experiences.forEach(ex => {
            test_agent.replay_buffer.append(ex[0][0], ex[1], ex[2], ex[3], ex[4][0])
        });

        let last_state_array = data["last_state_array"]
        let last_action_array = data["last_action_array"]
        let state_array = data["state_array"]
        let reward_array = data["reward_array"]

        test_agent.last_state = last_state_array[0][0];
        test_agent.last_action = last_action_array[0];
        let reward = reward_array[0]
        let pre_end =  test_agent.sum_rewards;
        test_agent.agent_end(reward)
        expect(test_agent.sum_rewards).toEqual(pre_end + reward);
    })

})

function observation_to_matrix(obs) {
    if (Array.isArray(obs)) return math.matrix(obs)
    return math.matrix([obs])
}

function compare_matrix(expected, actual, eps=0.000001){
    let delta =  math.subtract(actual, expected) as math.Matrix
    let size = expected.size()
    expect(math.sum(math.abs(delta))).toBeLessThan(size[0] * eps)
}

function get_layers(arr){
    return [
        {
            W: math.matrix(arr[0]['W']),
            b: math.matrix(arr[0]['b'])
        },
        {
            W: math.matrix(arr[1]['W']),
            b: math.matrix(arr[1]['b'])
        }
    ]
}