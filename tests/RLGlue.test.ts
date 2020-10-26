import {RLGlue} from '../src/rl_glue';
import {BaseAgent, BaseEnvironment, Action, Reward, Terminal, Observation}  from '../src/rl_base';


class MockAgent implements BaseAgent {
      step: number;
      action: Action;
      total_reward: number;
      constructor(){
          this.step = 0;
          this.action = 1;
          this.total_reward = 0;
      }
      agent_init(agent_info: any) {
          return;
      }
      agent_start(observation: number[]) {
          return this.action;
      }
      agent_step(reward: number, observation: number[]) {
            this.total_reward += reward;
            return this.action;
      }
      agent_end(reward: number) {
          this.total_reward += reward;
      }
      agent_cleanup() {
          this.step = 0;
          this.total_reward = 0;
      }
      agent_message(message: any) {
          return this.total_reward;
      }

  }

  class MockEnvironment implements BaseEnvironment {
      observation: Observation;
      terminal: Terminal;
      reward: Reward;
      obs: Observation;

      env_init(env_info) {
         this.terminal = false;
         this.reward = 1;
         this.obs = [1, 2];
         return this.obs;
      }
      env_start(): Observation {
         return this.obs;
      }
      env_step(action: Action): [Reward, Observation, Terminal] {        
        return [this.reward, this.obs, this.terminal];
      }
      env_cleanup() {
          return;
      }
      env_message(message: string) {
          return this.terminal;
      }

  }

  test('test_env_start', () => {
      let env = new MockEnvironment();
      let agent = new MockAgent();
      let lr_glue = new RLGlue(env, agent);
      lr_glue.rl_init(1, 2);
      expect(lr_glue.rl_env_start()).toEqual([1, 2]); 
  });
