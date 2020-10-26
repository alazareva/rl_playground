import {TDAgentStateAgg, get_state_feature} from '../src/tdStateAgg';

describe("TDAgentStateAgg", () => {

    it("correctly get state features", () => {
        let num_states = 10;
        let num_groups = 5;
        let num_states_in_group = 2;
        let state = 1
        let expected = [1.0, 0.0, 0.0, 0.0,  0.0]
        expect(get_state_feature(num_states_in_group, num_groups, 1)).toEqual(expected);

        state = 3
        expected = [0.0, 1.0, 0.0, 0.0,  0.0]
        expect(get_state_feature(num_states_in_group, num_groups, 1)).toEqual(expected);

        state = 6
        expected = [0.0, 0.0, 1.0, 0.0,  0.0]
        expect(get_state_feature(num_states_in_group, num_groups, 1)).toEqual(expected);

        state = 7
        expected = [0.0, 0.0, 0.0, 1.0,  0.0]
        expect(get_state_feature(num_states_in_group, num_groups, 1)).toEqual(expected);

        state = 10
        expected = [0.0, 0.0, 0.0, 0.0,  1.0]
        expect(get_state_feature(num_states_in_group, num_groups, 1)).toEqual(expected);
    })

})