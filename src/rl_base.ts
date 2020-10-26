export type Observation = number | number[];
export type Reward = number;
export type Terminal = boolean;
export type Action = number;


export interface BaseEnvironment {
    reward: Reward
    observation: Observation
    terminal: Terminal

    /**
     * Setup for the environment called when the experiment first starts.
     * @param env_info 
     */
    env_init(env_info)

    /**
     * A step taken by the environment.
     * @param action 
     */
    env_step(action: Action): [Reward, Observation, Terminal]

    /**
     * The first method called when the experiment starts, called before the agent starts.
     */
    env_start(): Observation

    /**
    * """Cleanup done after the environment ends"""
    */
    env_cleanup()

    /**
     * 
     * @param message A message asking the environment for information
     */
    env_message(message: string): any
}

export interface BaseAgent {
    /*
    * Implements the agent for an RL-Glue environment.
    */
    agent_init(agent_info)

    /*
    * Setup for the agent called when the experiment first starts."""
    */
    agent_start(observation: Observation): Action

    /**
     * A step taken by the agent.
     * @param reward 
     * @param observation 
     */
    agent_step(reward: Reward, observation: Observation): Action

    /**
     * Run when the agent terminates
     * @param reward 
     */

    agent_end(reward: Reward)

    // Cleanup done after the agent ends.
    agent_cleanup()

    /**
     * A function used to pass information from the agent to the experiment.
     * @param message 
     */
    agent_message(message: string): any
}
