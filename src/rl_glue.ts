import {
    BaseEnvironment, 
    BaseAgent,
    Observation,
    Reward,
    Action,
    Terminal,

} from './rl_base';

export class RLGlue {
    environment: BaseEnvironment;
    agent: BaseAgent;
    
    total_reward: number;
    last_action?: Action;
    num_steps: number;
    num_episodes: number ;
    prev_episode_reward: number

    constructor(env_class:  BaseEnvironment, agent_class: BaseAgent) {
        this.environment = env_class;
        this.agent = agent_class;
        
        this.total_reward = 0.0;
        this.num_steps = 0;
        this.num_episodes = 0;

        this.prev_episode_reward = 0;

    }

    // Initial method called when RLGlue experiment is created 
    rl_init(agent_init_info, env_init_info) {
        this.environment.env_init(env_init_info);
        this.agent.agent_init(agent_init_info);
    }

    //  Starts RLGlue experiment
    rl_start(): [Observation, Action] {
        this.total_reward = 0.0;
        this.num_steps = 1;
        let last_state = this.environment.env_start();
        this.last_action = this.agent.agent_start(last_state);
        return [last_state, this.last_action];
    }

    // """Starts the agent.
    rl_agent_start(observation: Observation): Action {
        return this.agent.agent_start(observation);
    }
    // """Step taken by the agent
    rl_agent_step(reward: Reward, observation: Observation): Action {
        return this.agent.agent_step(reward, observation);
    }

    // """Run when the agent terminates
    rl_agent_end(reward: Reward){
        return this.agent.agent_end(reward);
    }

    // Starts RL-Glue environment.
    rl_env_start(){
        this.total_reward = 0.0;
        this.num_steps = 1;
        return this.environment.env_start();
    }

    rl_env_fast_forward(){
        this.total_reward = 0.0;
        this.num_episodes += 1;
        return this.environment.env_start();
    }

    // Step taken by the environment based on action from agent
    rl_env_step(action: Action): [Reward, Observation, Terminal]{
        let ro = this.environment.env_step(action)
        let [this_reward, last_state, terminal] = ro
        this.total_reward += this_reward
        if (terminal) {
            this.num_episodes += 1
        }
        else {
            this.num_steps += 1
        }
    
        return ro
    }

    /**
    *  Step taken by RLGlue, takes environment step and either step orend by agent.  
    */
    rl_step(): [Reward, Observation, Action, Terminal] {
        let [this_reward, last_state, terminal] = this.environment.env_step(this.last_action)
        this.total_reward += this_reward;

        let ret;
        
        if (terminal){
            this.num_steps += 1;
            this.agent.agent_end(this_reward);
            ret = [this_reward, last_state, undefined, terminal];
            this.num_episodes += 1
            this.prev_episode_reward = this.total_reward;
        }
        else {
            this.num_steps += 1;
            this.last_action = this.agent.agent_step(this_reward, last_state);
            ret = [this_reward, last_state, this.last_action, terminal];
        }
        return ret;
    }

    // """Cleanup done at end of experiment."""
    rl_cleanup(){
        this.environment.env_cleanup();
        this.agent.agent_cleanup();
    }
    
    // """Message passed to communicate with agent during experiment
    rl_agent_message(message: string): any {
        return this.agent.agent_message(message);
    }

    //"""Message passed to communicate with environment during experiment
    rl_env_message(message: string): any {
        return this.environment.env_message(message);
    }

    // """Runs an RLGlue episode
    rl_episode(max_steps_this_episode: number): Terminal {
        let is_terminal = false
        this.rl_start();
        while (!is_terminal && (max_steps_this_episode == 0) || this.num_steps < max_steps_this_episode) {
            let rl_step_results = this.rl_step();
            is_terminal = rl_step_results[3];
        }
        return is_terminal;
    }

    // The total reward
    rl_return(): number {
        return this.total_reward;
    }
    // The total number of steps taken
    rl_num_steps(): number {
        return this.num_steps;
    }
    // The number of episodes
    rl_num_episodes(): number {
        return this.num_episodes;
    }

    rl_prev_reward(): number {
        return this.prev_episode_reward;
    }
}
