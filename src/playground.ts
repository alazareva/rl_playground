import {
  State,
  getKeyFromValue,
  agents,
} from "./state";
import * as d3 from 'd3';
import $ = require("jquery");
import { RLGlue } from './rl_glue';
import {QLearningAgent} from './qlearning'
import {ExpectedSarsaAgent} from './expectedSarsa'
import {DynaQAgent} from './dynaQAgent';
import * as maze from './mazeEnvironment';
const katex = require('katex');

const htmlTest = katex.renderToString("\\gamma V(s_{t+1})", {
  throwOnError: false
});

let eps = katex.renderToString("\\varepsilon")
let gamma = katex.renderToString("\\gamma")
let alpha = katex.renderToString("\\alpha") 

d3.select('#label-epsilon').html(`Epsilon (${eps})`);
d3.select('#label-discount').html(`Discount Factor (${gamma})`);
d3.select('#label-learningRate').html(`Learning Rate (${alpha})`);

d3.select('#s_t').html(katex.renderToString("s_t"));
d3.select('#a_t').html(katex.renderToString("a_t"));
d3.select('#pi').html(katex.renderToString("\\pi"));
d3.select('#r_t').html(katex.renderToString("r_t"));

let sarsaInfo = `
<h4>
<span>Expected SARSA Agent</span>
</h4>
<p> The agent works by blah blah </p><p> ${htmlTest} </p>`;

let qAgentInfo = `
<h4>
<span>Q-Learning Agent</span>
</h4>
<p> The agent works by blah blah </p>`;

let dynaQInfo = `
<h4>
<span>Dyna-Q  Agent</span>
</h4>
<p> The agent works by blah blah </p>`;

let agent_details = {
  "expectedSarsa": sarsaInfo,
  "qLearning": qAgentInfo,
  "dynaQ": dynaQInfo,
}
var episode = 0;

let env_info = {}
let env = new maze.MazeEnvironment();
var agent;
var rl_glue;
let rl_display;

let state = State.deserializeState();
rl_display = new maze.MazeEnvironmentDisplay();
reset_agent();
rl_display.display(env, agent)

function reset_agent() {
  let agent_text = agent_details[getKeyFromValue(agents, state.agentType)];
  let element = d3.select('#agent-details').html(agent_text);

  episode = 0;
  let agent_info =  {
    num_actions: 4, 
    num_states: 6 * 9, 
    epsilon: state.epsilon, 
    step_size: state.learningRate, 
    discount: state.discount,
  }
  if (state.agentType == agents.expectedSarsa) agent = new ExpectedSarsaAgent()
  else if (state.agentType == agents.qLearning) agent = new QLearningAgent()
  else agent = new DynaQAgent()
  rl_glue = new RLGlue(env, agent);
  rl_glue.rl_init(agent_info, env_info)
  rl_glue.rl_start()
}

// let mainWidth;

// More scrolling
d3.select(".more button").on("click", function() {
  let position = 800;
  d3.transition()
    .duration(1000)
    .tween("scroll", scrollTween(position));
});

function scrollTween(offset) {
  return function() {
    let i = d3.interpolateNumber(window.pageYOffset ||
        document.documentElement.scrollTop, offset);
    return function(t) { scrollTo(0, i(t)); };
  };
}

class Player {
  private timerIndex = 0;
  private isPlaying = false;
  private callback: (isPlaying: boolean) => void = null;

  /** Plays/pauses the player. */
  playOrPause() {
    if (this.isPlaying) {
      this.isPlaying = false;
      this.pause();
    } else {
      this.isPlaying = true;
      if (iter === 0) {
        simulationStarted();
      }
      this.play();
    }
  }

  onPlayPause(callback: (isPlaying: boolean) => void) {
    this.callback = callback;
  }

  play() {
    this.pause();
    this.isPlaying = true;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
    this.start(this.timerIndex);
  }

  pause() {
    this.timerIndex++;
    this.isPlaying = false;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
  }

  private start(localTimerIndex: number) {
    d3.timer(() => {
      if (localTimerIndex < this.timerIndex) {
        return true;  // Done.
      }
      oneStep();
      let rl_step_results = rl_glue.rl_step();
      let is_terminal = rl_step_results[3];
      rl_display.update(env, agent)
      if (is_terminal) {
        rl_glue.rl_start()
      }
      episode = rl_glue.rl_num_episodes();
      return false;  // Not done.
    }, 0);
  }
}

let player = new Player();
let iter = 0;


function makeGUI() {
  d3.select("#reset-button").on("click", () => {
    reset_agent();
  });

  d3.select("#play-pause-button").on("click", function () {
    // Change the button's content.
    player.playOrPause();
  });

  player.onPlayPause(isPlaying => {
    d3.select("#play-pause-button").classed("playing", isPlaying);
  });


  d3.select("#next-step-button").on("click", () => {
    rl_glue.rl_env_fast_forward();
  });

  let learningRate = d3.select("#learningRate").on("change", function() {
    state.learningRate = this.value;
    state.serialize();
    agent.set_step_size(state.learningRate)
  });
  learningRate.property("value", state.learningRate);

  let epsilon = d3.select("#epsilon").on("change", function() {
    state.epsilon = this.value;
    state.serialize();
    agent.set_epsilon(state.epsilon)
  });
  epsilon.property("value", state.epsilon);

  let discount = d3.select("#discount").on("change", function() {
    state.discount = this.value;
    state.serialize();
    agent.set_discount(state.discount)
  });
  discount.property("value", state.discount);


  let agentDropdown = d3.select("#agentType").on("change",
      function() {
    state.agentType = agents[this.value];
    reset_agent();
  });
  agentDropdown.property("value", getKeyFromValue(agents, state.agentType));

  let showVisits = d3.select("#show-visits").on("change", function() {
    state.showVisits = this.checked;
    state.serialize();
    rl_display.showVisits = state.showVisits;
  });
  // Check/uncheck the checbox according to the current state.
  showVisits.property("checked", state.showVisits);

  let showQ= d3.select("#show-q").on("change", function() {
    state.showQ = this.checked;
    state.serialize();
    rl_display.showQ = state.showQ;
  });
  // Check/uncheck the checbox according to the current state.
  showQ.property("checked", state.showQ);
}

  function zeroPad(n: number): string {
    let pad = "000000";
    return (pad + n).slice(-pad.length);
  }

  function addCommas(s: string): string {
    return s.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

function oneStep(): void {
  iter++;
  d3.select("#iter-number").text(addCommas(zeroPad(episode)));
}

function reset(onStartup=false) {
  state.serialize();
  if (!onStartup) {
  }
  player.pause();
};

makeGUI();
d3.select("#iter-number").text(addCommas(zeroPad(episode)));
reset(true);

function simulationStarted() {
  ga('send', {
    hitType: 'event',
    eventCategory: 'Starting Simulation',
  });
}
