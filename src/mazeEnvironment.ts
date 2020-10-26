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
import * as math from 'mathjs';


const UP = 0;
const RIGHT = 1;
const DOWN = 2;
const LEFT = 3;

export class MazeEnvironment implements BaseEnvironment {
    maze_dims: number[];
    obstacles: number[][];
    start_state: number[];
    end_state: number[];
    current_state: number[];
    reward: number;
    observation: Observation;
    terminal: boolean;
    reward_obs_term: [number, Observation, boolean];
    redraw_maze: boolean;
    visited

    env_init(env_info: any) {
        this.maze_dims = env_info.maze_dims || [6, 9];
        this.obstacles = env_info.obstacles ||  [[1, 2], [2, 2], [3, 2], [4, 5], [0, 7], [1, 7], [2, 7]];
        this.start_state = env_info.start_state || [2, 0];
        this.end_state = env_info.end_state || [0, 8];
        this.current_state = new Array(2);

        this.reward = 0.0;
        this.terminal = false;
        this.redraw_maze = false;

        this.reward_obs_term = [this.reward, this.observation, this.terminal]
        this.visited = [...Array(this.maze_dims[0])].map(e => Array(this.maze_dims[1]).fill(0));
    }

    env_step(action: number): [number, Observation, boolean] {
        let reward = 0.0;
        let is_terminal = false;

        let row = this.current_state[0];
        let col = this.current_state[1];

        let next_row;
        let next_col;

        if (action == UP) {
            next_row = row - 1
            next_col = col;
        } else if (action == RIGHT) {
            next_row = row,
            next_col = col + 1;
        } else if (action == DOWN) {
            next_row = row + 1;
            next_col = col;
        } else {
            next_row = row;
            next_col = col - 1;
        }

        if (!(this.out_of_bounds(next_row, next_col) || this.is_obstacle(next_row, next_col))) {
            this.current_state = [next_row, next_col]
            this.visited[next_row][next_col] += 1
        } else {
            reward = -0.01
        }
        if (this.current_state[0] == this.end_state[0] && this.current_state[1] == this.end_state[1]) {
            reward = 10.0;
            is_terminal = true;
        }
        this.reward = reward;
        this.terminal = is_terminal;
        this.observation = this.get_observation(this.current_state),
        this.reward_obs_term = [this.reward, this.observation, this.terminal]
        return this.reward_obs_term;
    }
    env_start(): Observation {
        this.current_state = this.start_state;
        this.observation = this.get_observation(this.current_state)
        this.reward_obs_term[1] = this.observation;
        return this.observation;
    }
    env_cleanup() {
        this.current_state = this.start_state;
    }
    env_message(message: string) {
        if (message == 'current reward') {
            return this.reward_obs_term[0];
        }
    }

    get_observation(state) {
        return state[0] * this.maze_dims[1] + state[1];
    }

    out_of_bounds(row, col) {
        return (
            row < 0 || row > this.maze_dims[0] - 1
            || col < 0 || col > this.maze_dims[1] - 1
            )
    }

    is_obstacle(row: number, col: number): boolean {
        return this.obstacles.some((tup) => tup[0] == row && tup[1] == col);
    }
};

class ShortcutMazeEnvironment extends MazeEnvironment {
    change_at_n: number;
    timesteps: number;
    shortcut: number[];

    env_init(env_info) {
        super.env_init(env_info)
        this.timesteps = 0;
        this.change_at_n = env_info.change_at_n
    }

    env_step(action) {
        this.timesteps += 1;
        if (this.timesteps == this.change_at_n) {
            this.shortcut = this.obstacles.pop()
            this.redraw_maze = true;
        }
        return super.env_step(action)
    }
    env_cleanup() {
        this.current_state = this.start_state;
        this.obstacles.push(this.shortcut);
    }
}

export class MazeEnvironmentDisplay {
    agent_vis;
    cell_vis;
    Q_vis;
    X;
    Y;
    showQ = true;
    showVisits = true;
    display(env: MazeEnvironment, agent: BaseAgent) {
        let height = env.maze_dims[0] * 50;
        let width = env.maze_dims[1] * 50;
        var stage = d3.select("#rl_playground");
        var container = stage.append("div").attr("class", "visualization");
        var svg = container
		.append("svg")
		.style("width", "100%")
		.style("min-height", height)
        .style("user-select", "none");
        var grid = svg.append("g").attr("class", "grid");

        var cell_data = [];
        for (let x = 0; x < env.maze_dims[1]; x++) {
            for (let y = 0; y < env.maze_dims[0]; y++) {
                let wall = env.is_obstacle(y, x)
                cell_data.push({ x: x, y: y, wall: wall})
            }
        }
        var cell = grid.selectAll("g").data(cell_data);

        var X = d3.scale
		.linear()
		.domain([0, env.maze_dims[1]])
		.range([0, width]);
	var Y = d3.scale
		.linear()
		.domain([0, env.maze_dims[0]])
        .range([0, height]);

    this.X = X;
    this.Y = Y;

    	var S = d3.scale
		.linear()
		.domain([0, env.maze_dims[1]])
		.range([0, width]);
        
        var cell_enter = cell
		.enter()
		.append("g")
		.attr("class", d => "cell pos-" + d.x + "-" + d.y)
        .attr("transform", d => "translate(" + X(d.x) + "," + Y(d.y) + ")");
        
       
    	cell_enter
		.append("rect")
        //.attr("class", d => d.wall? "wall": "background")
        .attr("fill", d => d.wall? "#E69F00": "white")
		.attr("width", S(0.95))
        .attr("height", S(0.95)); 

        this.cell_vis = cell_enter;
        
        var goal = grid
		.append("rect")
		.attr("class", "goal")
		.attr("width", S(0.95))
		.attr("height", S(0.95))
		.attr("data-x", env.end_state[1])
		.attr("data-y", env.end_state[0])
        .attr("transform", d => "translate(" + X(env.end_state[1] + 0.0) + "," + Y(env.end_state[0] + 0.0) + ")");

        cell.exit().remove();

        var agent_layer = grid.append("g");
        var agent_vis = agent_layer
            .append("circle")
            .attr("class", "agent")
            .attr("r", S(0.4))
            .attr("data-x", env.current_state[0])
            .attr("cx", X(env.current_state[1] + 0.475))
            .attr("data-y", env.current_state[1])
            .attr("cy", Y(env.current_state[0] + 0.475));
        this.agent_vis = agent_vis;

        let Q = agent.agent_message('get_Q')

        let scaleq = (q, v) => this.scale_q(q, v) 
        if (Q) {
            this.Q_vis = this.cell_vis.append("g").attr("class", "Q")

            this.Q_vis
            .append("path")
            .attr("class", "Q-UP")
            .style("visibility", function (d) { 
                if (d.wall) return "hidden"
                return "visible"
            })
            .attr("transform", "translate(" + X(0.475) + "," + Y(0.475 - 0.2) + ")")
            .attr("d", d3.svg.symbol().type("triangle-up"))
            .style("fill",  d => scaleq(Q, Q.get([env.get_observation([d.y, d.x]), 0])));

            this.Q_vis 
            .append("path")
            .attr("class", "Q-DOWN")
            .style("visibility", function (d) { 
                if (d.wall) return "hidden"
                return "visible"
            })
            .attr("transform", "translate(" + X(0.475) + "," + Y(0.475 + 0.2) + ")")
            .attr("d", d3.svg.symbol().type("triangle-down"))
            .style("fill",  d => scaleq(Q, Q.get([env.get_observation([d.y, d.x]), 2])));

            this.Q_vis 
            .append("path")
            .attr("class", "Q-LEFT")
            .style("visibility", function (d) { 
                if (d.wall) return "hidden"
                return "visible"
            })
            .attr("transform", "translate(" + X(0.475  - 0.2) + "," + Y(0.475) + ") rotate(-90)")
            .attr("d", d3.svg.symbol().type("triangle-up"))
            .style("fill",  d => scaleq(Q, Q.get([env.get_observation([d.y, d.x]), 3])));


            this.Q_vis 
            .append("path")
            .attr("class", "Q-RIGHT")
            .style("visibility", function (d) { 
                if (d.wall) return "hidden"
                return "visible"
            })
            .attr("transform", "translate(" + X(0.475 + 0.2) + "," + Y(0.475) + ") rotate(90)")
            .attr("d", d3.svg.symbol().type("triangle-up"))
            .style("fill",  d => scaleq(Q, Q.get([env.get_observation([d.y, d.x]), 1])));
        }

    }

    update(env: MazeEnvironment, agent: BaseAgent) {

        if (env.redraw_maze) {
            // TODO update
        }
        this.agent_vis
        .attr("data-x", env.current_state[0])
        .attr("cx", this.X(env.current_state[1] + 0.475))
        .attr("data-y", env.current_state[1])
        .attr("cy", this.Y(env.current_state[0] + 0.475));


        let visited =  env.visited;

        var getColor = d3.scale.linear<string>().domain([math.min(visited), math.max(visited)])
            .range(['white', 'gray']);

        if (this.showVisits) {
            this.cell_vis.selectAll('rect').attr("fill", function (d) { 
                if (d.wall) return "#E69F00"
                return getColor(visited[d.y][d.x]); 
        });
    } else {
        this.cell_vis.selectAll('rect').attr("fill", d => d.wall? "#E69F00": "white")
    }
        let Q = agent.agent_message('get_Q')
        let scaleq = (q, v) => this.scale_q(q, v) 
        if (Q && this.showQ) {
            this.Q_vis.attr("class", "Q")
            this.Q_vis.select(".Q-UP")
            .style("fill",  d => scaleq(Q, Q.get([env.get_observation([d.y, d.x]), 0])));
            this.Q_vis.select(".Q-RIGHT")
            .style("fill",  d => scaleq(Q, Q.get([env.get_observation([d.y, d.x]), 1])));
            this.Q_vis.select(".Q-DOWN")
            .style("fill",  d => scaleq(Q, Q.get([env.get_observation([d.y, d.x]), 2])));
            this.Q_vis.select(".Q-LEFT")
            .style("fill",  d => scaleq(Q, Q.get([env.get_observation([d.y, d.x]), 3])));
    } else if (!this.showQ){
        this.Q_vis.attr("class", "hide-q")
    }
}
    scale_q(Q, v) {
        if (v < 0)
        return d3.scale.linear<string>().domain([0, -math.min(Q)]).range(['white', '#E69F00'])(-v);
        else 
        return d3.scale.linear<string>().domain([0, math.max(Q)]).range(['white', '#0072B2'])(v);
    }
}