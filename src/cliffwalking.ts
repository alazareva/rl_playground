import { 
    BaseEnvironment, 
    Observation, 
    Reward,
    Action,
    Terminal,
    BaseAgent,
} from "./rl_base";
import { color } from 'd3';
import { RLGlue } from './rl_glue';
import * as d3 from 'd3';


export function compare_tuples(t1: [number, number], t2: [number, number]): boolean {
    return t1[0] == t2[0] && t1[1] == t2[1];
}

export class CliffWalkingEnvironment implements BaseEnvironment {
    reward: number;
    observation: Observation;
    terminal: boolean;
    grid_h: number;
    grid_w: number;
    start_loc: [number, number];
    goal_loc: [number, number];
    cliff: [number, number][];
    agent_loc: [number, number];
    reward_state_term: [Reward, Observation, Terminal]

    UP = 0;
    LEFT = 1;
    DOWN = 2;
    RIGHT = 3;

    env_init(env_info: { grid_height:number, grid_width :number}) {
        this.grid_h = env_info.grid_height || 4;
        this.grid_w = env_info.grid_width || 12;
        this.start_loc = [this.grid_h - 1, 0]
        this.goal_loc = [this.grid_h - 1, this.grid_w - 1];
        this.reward_state_term = [this.reward, this.observation, this.terminal]
        this.cliff = [];
        for (var _i = 1; _i < this.grid_w - 1; _i++) {
            this.cliff.push([this.grid_h - 1, _i]);
        }
    }
    
    within_bounds(possible_next_loc: [number, number]): boolean {
        let [h, w] = possible_next_loc;
        return h >= 0 && h < this.grid_h && w >= 0 && w < this.grid_w;
    }
    env_step(action: Action): [Reward, Observation, Terminal] {
        let possible_next_loc: [number, number];
        if (action == this.UP) {
            possible_next_loc  = [this.agent_loc[0] - 1, this.agent_loc[1]];
        } else if (action == this.DOWN) {
            possible_next_loc = [this.agent_loc[0] + 1, this.agent_loc[1]];

        } else if (action == this.LEFT) {
            possible_next_loc = [this.agent_loc[0], this.agent_loc[1] - 1];
        } else {
            possible_next_loc = [this.agent_loc[0], this.agent_loc[1] + 1];
        }
        if (this.within_bounds(possible_next_loc)) {
            this.agent_loc = possible_next_loc;
        }
        let reward = -1;
        let terminal = false;

        if (compare_tuples(this.agent_loc, this.goal_loc)) {
            terminal = true;
        } else if (this.includes(this.cliff, this.agent_loc)){
            reward = - 100;
            this.agent_loc = this.start_loc;   
        }

        this.reward_state_term = [reward, [this.state(this.agent_loc)], terminal];
        return this.reward_state_term;
    }

    includes(collection: [number, number][], tup: [number, number]): boolean {
        return collection.some(function(a){return compare_tuples(a, tup)});
    }

    env_start(): Observation {
        let reward = 0;
        this.agent_loc = this.start_loc;
        let state = this.state(this.agent_loc);
        let termination = false;
        this.reward_state_term = [reward, [state], termination];
        return this.reward_state_term[1];
    }
    env_cleanup() {
        this.agent_loc = this.start_loc;
    }
    env_message(message: string): any {
        throw new Error("Method not implemented.");
    }

    state(loc: [number, number]): number {
        let [i, j] = loc;
        return i * this.grid_w + j;
    }
}

export class TDAgent implements BaseAgent {
    policy: number[][]
    discount: number
    step_size: number
    values: number[]
    num_actions: number
    num_states: number
    last_state: Observation

    agent_init(agent_info: {policy: number[][], discount: number, step_size: number}) {
        this.policy = agent_info.policy;
        this.discount = agent_info.discount;
        this.step_size = agent_info.step_size;
        this.values = new Array(this.policy.length).fill(0);
    }
    agent_start(observation: Observation): Action {
        let action = this.random_choice(observation);
        this.last_state = observation;
        return action; 
    }

    agent_step(reward: number, observation: Observation): Action {
        let target = reward + this.discount * this.values[observation[0]];
        let v_st = this.values[this.last_state[0]];
        this.values[this.last_state[0]] = v_st + this.step_size * (target - v_st);
        let action = this.random_choice(observation);
        this.last_state = observation;
        return action;
    }

    agent_end(reward: number) {
        let target = reward;
        let v_st = this.values[this.last_state[0]];
        this.values[this.last_state[0]] = v_st + this.step_size * (target - v_st);
    }
    agent_cleanup() {
        this.last_state = undefined;
    }
    agent_message(message: string): any {
        if (message == 'get_values') {
            return this.values;
        }
    }

    random_choice(obaservation: Observation){
        let weights = this.policy[obaservation[0]]
        let num = Math.random();
        let last_index = weights.length - 1;
        var s = 0;
        for (var i = 0; i < last_index; i++) {
            s += weights[i];
            if (num < s) {
                return i;
            }
        }
        return last_index;

    }
}


export class CliffWalkingEnvironmentDisplay {
    agent_vis;
    X;
    Y;
    display(env: CliffWalkingEnvironment, agent: BaseAgent) {
        let height = env.grid_h * 50;
        let width = env.grid_w * 50;
        var stage = d3.select("#rl_playground");
        var container = stage.append("div").attr("class", "visualization");
        var svg = container
		.append("svg")
		.style("width", "100%")
		.style("min-height", height)
        .style("user-select", "none");
        var grid = svg.append("g").attr("class", "grid");

        var cell_data = [];
        for (let x = 0; x < env.grid_w; x++) {
            for (let y = 0; y < env.grid_h; y++) {
                let cliff = env.includes(env.cliff, [y, x])
                cell_data.push({ x: x, y: y, cliff: cliff})
            }
        }
        var cell = grid.selectAll("g").data(cell_data);

        var X = d3.scale
		.linear()
		.domain([0, env.grid_w])
		.range([0, width]);
	var Y = d3.scale
		.linear()
		.domain([0, env.grid_h])
        .range([0, height]);

    this.X = X;
    this.Y = Y;

    	var S = d3.scale
		.linear()
		.domain([0, env.grid_w])
		.range([0, width]);
        
        var cell_enter = cell
		.enter()
		.append("g")
		.attr("class", d => "cell pos-" + d.x + "-" + d.y)
        .attr("transform", d => "translate(" + X(d.x) + "," + Y(d.y) + ")");
        
       
    	cell_enter
		.append("rect")
		.attr("class", d => d.cliff? "cliff": "background")
		.attr("width", S(0.95))
        .attr("height", S(0.95)); 
        
        var goal = grid
		.append("rect")
		.attr("class", "goal")
		.attr("width", S(0.95))
		.attr("height", S(0.95))
		.attr("data-x", env.goal_loc[1])
		.attr("data-y", env.goal_loc[0])
        .attr("transform", d => "translate(" + X(env.goal_loc[1] + 0.0) + "," + Y(env.goal_loc[0] + 0.0) + ")");

        cell.exit().remove();

        var agent_layer = grid.append("g");
        var agent_vis = agent_layer
            .append("circle")
            .attr("class", "agent")
            .attr("r", S(0.4))
            .attr("data-x", env.agent_loc[0])
            .attr("cx", X(env.agent_loc[1] + 0.475))
            .attr("data-y", env.agent_loc[1])
            .attr("cy", Y(env.agent_loc[0] + 0.475));
        this.agent_vis = agent_vis;
    }

    update(env: CliffWalkingEnvironment, agent: BaseAgent) {
        this.agent_vis
        .attr("data-x", env.agent_loc[0])
        .attr("cx", this.X(env.agent_loc[1] + 0.475))
        .attr("data-y", env.agent_loc[1])
        .attr("cy", this.Y(env.agent_loc[0] + 0.475));
    }
}

export function run_experiment(
    env_info,
    agent_info,
    num_episodes,
    plot_freq,
){
    let env = new CliffWalkingEnvironment()
    let agent = new TDAgent();
    let rl_glue = new RLGlue(env, agent);
    rl_glue.rl_init(agent_info, env_info);
    for (let i = 0; i < num_episodes; i++) {
        rl_glue.rl_episode(0)
        if (i % plot_freq == 0) {
            let values = rl_glue.agent.agent_message('get_values');
            console.log('Episode ' + i);
            console.log(values);
        }
    }
    console.log('rl_num_steps '  + rl_glue.rl_num_steps())
}
