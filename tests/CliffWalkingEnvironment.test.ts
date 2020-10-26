import {CliffWalkingEnvironment, run_experiment} from '../src/cliffwalking'

describe("CliffWalkingEnvironment", () => {
    it("correctly return state", () => {
        let env = new CliffWalkingEnvironment();
        env.env_init({grid_height: 4, grid_width: 12});
        let coords_to_test: [number, number][] = [[0, 0], [0, 11], [1, 5], [3, 0], [3, 9], [3, 11]];
        let true_states = [0, 11, 17, 36, 45, 47]
        for (let i = 0; i < coords_to_test.length; i++) {
            expect(env.state(coords_to_test[i])).toEqual(true_states[i]);
          }
    });
    it("correcly moves UP", () => {
      let env = new CliffWalkingEnvironment();
      env.env_init({grid_height: 4, grid_width: 12})
      env.agent_loc = [0, 0];
      env.env_step(env.UP)
      expect(env.agent_loc).toEqual([0, 0])

      env.agent_loc = [1, 0];
      env.env_step(env.UP)
      expect(env.agent_loc).toEqual([0, 0])
    });
    it("correcly rewards", () => {
      let env = new CliffWalkingEnvironment();
      env.env_init({grid_height: 4, grid_width: 12})
      env.agent_loc = [0, 0];
      let reward_state_term = env.env_step(env.UP)
      expect(reward_state_term[0]).toEqual(-1)
      expect(reward_state_term[1][0]).toEqual(env.state([0, 0]))
      expect(reward_state_term[2]).toBe(false)

      env.agent_loc = [3, 1];
      reward_state_term = env.env_step(env.DOWN)
      expect(reward_state_term[0]).toEqual(-100)
      expect(reward_state_term[1][0]).toEqual(env.state([3, 0]))
      expect(reward_state_term[2]).toBe(false)

      env.agent_loc = [2, 11];
      reward_state_term = env.env_step(env.DOWN)
      expect(reward_state_term[0]).toEqual(-1)
      expect(reward_state_term[1][0]).toEqual(env.state([3, 11]))
      expect(reward_state_term[2]).toBe(true)
    });

    it('can run experiment', () => {
      
      let env_info = {grid_height: 4, grid_width: 12}
      let agent_info = {discount: 1, step_size: 0.01}


      let arr = Array(env_info.grid_width * env_info.grid_height).fill([0.25, 0.25, 0.25, 0.25])
      arr[36] =  [1, 0, 0, 0];
      for (let i = 24; i < 36; i ++){
        arr[i] = [0, 0, 0, 1]
      }
      arr[35] = [0, 0, 1, 0]
      agent_info["policy"] = arr;


        run_experiment(
        env_info, 
        agent_info,
        2,
        1,
        );
          expect(1).toEqual(1)  
          });
        
  });