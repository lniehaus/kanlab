/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as kan from "./kan";
import { HeatMap, reduceMatrix } from "./heatmap";
import { SplineChart } from "./splinechart";
import {
  State,
  datasets,
  regDatasets,
  problems,
  getKeyFromValue,
  Problem
} from "./state";
import { Example2D, shuffle, generateSymbolicDataset, SymbolicDatasetResult } from "./dataset";
import { AppendingLineChart } from "./linechart";
import { NetworkLayoutManager } from "./layout";
import * as d3 from 'd3';

let mainWidth;

// More scrolling
d3.select(".more button").on("click", function () {
  let position = 800;
  d3.transition()
    .duration(1000)
    .tween("scroll", scrollTween(position));
});

function scrollTween(offset) {
  return function () {
    let i = d3.interpolateNumber(window.pageYOffset ||
      document.documentElement.scrollTop, offset);
    return function (t) { scrollTo(0, i(t)); };
  };
}

const RECT_SIZE = 30;
const NUM_SAMPLES_CLASSIFY = 500;
const NUM_SAMPLES_REGRESS = 1200;
const NUM_SAMPLES_SYMBOLIC = 800;
const DENSITY = 100;
const SPLINE_CHART_SIZE_X = 30;
const SPLINE_CHART_SIZE_Y = 30;
const NODE_SPACING = 25;
const SYMBOLIC_PREVIEW_SAMPLES = 200;
const SYMBOLIC_ALLOWED_INPUTS: { [key: string]: boolean } = {
  "x": true
};
const SYMBOLIC_COMPOSITION_NODE_RATIO = 0.03;
const SYMBOLIC_COMPOSITION_MIN_RMS = 1e-3;
const SYMBOLIC_COMPOSITION_COEFF_EPS = 1e-3;

// Helper: populate numControlPoints options based on degree
function updateNumControlPointsOptionsForDegree(degreeVal: number, currentNumControlPoints?: number) {
  // B-spline rule: number of control points >= degree + 1
  const minControlPoints = Math.max(2, Math.floor(degreeVal) + 2);
  const maxControlPoints = 20;

  const values: number[] = d3.range(minControlPoints, maxControlPoints + 1).sort((a: number, b: number) => a - b);
  const cpSel = d3.select("#numControlPoints");

  const opts = cpSel.selectAll("option").data(values, (d: any) => d);
  opts.enter()
    .append("option")
    .attr("value", (d: any) => d)
    .text((d: any) => d);
  opts.exit().remove();

  // enforce DOM order
  cpSel.selectAll("option").sort((a: number, b: number) => a - b);

  // Ensure a valid selection
  const desired = currentNumControlPoints != null ? Math.max(minControlPoints, currentNumControlPoints) : minControlPoints;
  cpSel.property("value", desired);

  // If changed programmatically, dispatch change so the app reacts.
  const el = cpSel.node() as HTMLSelectElement;
  const effectiveCurrent = (currentNumControlPoints != null ? currentNumControlPoints : minControlPoints);
  if (el && +el.value !== effectiveCurrent) {
    const evt = new Event("change", { bubbles: true });
    el.dispatchEvent(evt);
  }
}

enum HoverType {
  WEIGHT
}

interface InputFeature {
  f: (x: number, y: number) => number;
  label?: string;
}

let INPUTS: { [name: string]: InputFeature } = {
  "x": { f: (x, y) => x, label: "X_1" },
  "y": { f: (x, y) => y, label: "X_2" },
  "xSquared": { f: (x, y) => x * x, label: "X_1^2" },
  "ySquared": { f: (x, y) => y * y, label: "X_2^2" },
  "xTimesY": { f: (x, y) => x * y, label: "X_1X_2" },
  "sinX": { f: (x, y) => Math.sin(x), label: "sin(X_1)" },
  "sinY": { f: (x, y) => Math.sin(y), label: "sin(X_2)" },
};

type SymbolicCurvePoint = {
  x: number;
  y: number;
};

type SymbolicLibraryEntry = {
  id: string;
  label: string;
  expression: string;
  evaluator: (x: number) => number;
};

type SymbolicLibraryBasisResult = {
  entry: SymbolicLibraryEntry;
  coefficient: number;
  samples: SymbolicCurvePoint[];
  color: string;
};

type SymbolicLibraryFit = {
  combined: SymbolicCurvePoint[];
  basis: SymbolicLibraryBasisResult[];
  rmse: number;
};

type SymbolicBestBasisSummary = {
  entry: SymbolicLibraryEntry;
  coefficient: number;
  samples: SymbolicCurvePoint[];
  color: string;
  rmse: number;
};

const SYMBOLIC_LIBRARY_COLORS = [
  "#4caf50",
  "#8e24aa",
  "#00897b",
  "#ff7043",
  "#5c6bc0",
  "#f06292",
  "#26c6da",
  "#7cb342"
];

const SYMBOLIC_LIBRARY_COMBINED_COLOR = "#388e3c";
const SYMBOLIC_LIBRARY_REGULARIZATION = 1e-6;

const SYMBOLIC_LIBRARY_ENTRIES: SymbolicLibraryEntry[] = [
  { id: "const", label: "1", expression: "1", evaluator: () => 1 },
  { id: "x", label: "x", expression: "x", evaluator: x => x },
  { id: "x2", label: "x²", expression: "x^2", evaluator: x => x * x },
  { id: "x3", label: "x³", expression: "x^3", evaluator: x => x * x * x },
  { id: "sin", label: "sin(x)", expression: "sin(x)", evaluator: x => Math.sin(x) },
  { id: "cos", label: "cos(x)", expression: "cos(x)", evaluator: x => Math.cos(x) },
  { id: "exp", label: "exp(x)", expression: "exp(x)", evaluator: x => Math.exp(x) },
  { id: "abs", label: "|x|", expression: "|x|", evaluator: x => Math.abs(x) }
];

let symbolicLibraryFit: SymbolicLibraryFit | null = null;
let symbolicBestBasis: SymbolicBestBasisSummary | null = null;

let HIDABLE_CONTROLS = [
  ["Show test data", "showTestData"],
  ["Discretize output", "discretize"],
  ["Play button", "playButton"],
  ["Step button", "stepButton"],
  ["Reset button", "resetButton"],
  ["Learning rate", "learningRate"],
  ["Problem type", "problem"],
  ["Which dataset", "dataset"],
  ["Ratio train data", "percTrainData"],
  ["Noise level", "noise"],
  ["Batch size", "batchSize"],
  ["# of hidden layers", "numHiddenLayers"],
  ["Control points", "numControlPoints"],
  ["Spline degree", "degree"],
  ["Init noise", "initNoise"],
  ["Symbolic expression", "symbolicExpression"],
];

type SymbolicSample = {
  x: number;
  target: number;
  targetNormalized: number;
  predicted: number;
};

type SymbolicNetworkTrace = {
  sampleXs: number[];
  nodeOutputs: { [nodeId: string]: number[] };
  edgeInputs: { [edgeId: string]: number[] };
  edgeOutputs: { [edgeId: string]: number[] };
};

type SymbolicEdgeMatch = {
  edgeId: string;
  sourceId: string;
  destId: string;
  entry: SymbolicLibraryEntry;
  coefficient: number;
  rmse: number;
  contributionRms: number;
  color: string;
  pruned: boolean;
  sourceLabel: string;
  renderedExpression?: string;
  renderedFull?: string;
  sourceExpression?: string;
  termSign?: "+" | "-";
};

type SymbolicNodeComposition = {
  nodeId: string;
  label: string;
  layerIndex: number;
  expression: string;
  contributions: SymbolicEdgeMatch[];
  isOutput: boolean;
};

type SymbolicCompositionLayer = {
  title: string;
  layerIndex: number;
  nodes: SymbolicNodeComposition[];
};

type SymbolicCompositionSummary = {
  expression: string;
  layers: SymbolicCompositionLayer[];
};

class SymbolicPlot {
  private container: d3.Selection<any>;
  private svg: d3.Selection<any>;
  private width: number;
  private height: number;
  private margin = { top: 12, right: 12, bottom: 36, left: 46 };
  private xScale: d3.scale.Linear<number, number>;
  private yScale: d3.scale.Linear<number, number>;
  private xAxis: d3.svg.Axis;
  private yAxis: d3.svg.Axis;
  private targetPath: d3.Selection<any>;
  private predictedPath: d3.Selection<any>;
  private libraryPath: d3.Selection<any>;
  private basisGroup: d3.Selection<any>;
  private trainGroup: d3.Selection<any>;
  private testGroup: d3.Selection<any>;

  constructor(container: d3.Selection<any>) {
    this.container = container;
    let node = container.node() as HTMLElement;
    this.width = Math.max(360, (node && node.offsetWidth) ? node.offsetWidth - this.margin.left - this.margin.right : 360);
    this.height = 220;

    this.xScale = d3.scale.linear()
      .domain([-1, 1])
      .range([0, this.width]);

    this.yScale = d3.scale.linear()
      .domain([-1, 1])
      .range([this.height, 0]);

    this.xAxis = d3.svg.axis()
      .scale(this.xScale)
      .orient("bottom");

    this.yAxis = d3.svg.axis()
      .scale(this.yScale)
      .orient("left");

    this.svg = container.append("svg")
      .attr("width", this.width + this.margin.left + this.margin.right)
      .attr("height", this.height + this.margin.top + this.margin.bottom)
      .append("g")
      .attr("transform", `translate(${this.margin.left},${this.margin.top})`);

    this.svg.append("g")
      .attr("class", "x axis")
      .attr("transform", `translate(0,${this.height})`)
      .call(this.xAxis);

    this.svg.append("g")
      .attr("class", "y axis")
      .call(this.yAxis);

    this.svg.append("text")
      .attr("class", "symbolic-title")
      .attr("x", this.width / 2)
      .attr("y", -6)
      .style("text-anchor", "middle")
      .style("font-size", "13px")
      .style("font-weight", "500")
      .text("Symbolic regression response");

    this.svg.append("text")
      .attr("class", "symbolic-x-label")
      .attr("x", this.width / 2)
      .attr("y", this.height + this.margin.bottom - 8)
      .style("text-anchor", "middle")
      .style("font-size", "11px")
      .text("x");

    this.svg.append("text")
      .attr("class", "symbolic-y-label")
      .attr("transform", "rotate(-90)")
      .attr("y", -this.margin.left + 12)
      .attr("x", -this.height / 2)
      .style("text-anchor", "middle")
      .style("font-size", "11px")
      .text("f(x)");

    let legend = this.svg.append("g")
      .attr("class", "symbolic-legend")
      .attr("transform", `translate(${Math.max(0, this.width - 160)}, ${this.height + 6})`);

    let legendItems = [
      { label: "Ground truth", color: "#0877bd", dash: "" },
      { label: "Prediction", color: "#f59322", dash: "6,3" },
      { label: "Library fit", color: SYMBOLIC_LIBRARY_COMBINED_COLOR, dash: "4,2" }
    ];

    legendItems.forEach((item, idx) => {
      let group = legend.append("g")
        .attr("transform", `translate(0, ${idx * 14})`);
      group.append("line")
        .attr("x1", 0)
        .attr("x2", 18)
        .attr("y1", 6)
        .attr("y2", 6)
        .style("stroke", item.color)
        .style("stroke-width", "3px")
        .style("stroke-dasharray", item.dash);
      group.append("text")
        .attr("x", 22)
        .attr("y", 9)
        .style("font-size", "10px")
        .text(item.label);
    });

    this.targetPath = this.svg.append("path")
      .attr("class", "symbolic-target")
      .style("fill", "none")
      .style("stroke", "#0877bd")
      .style("stroke-width", "2px");

    this.predictedPath = this.svg.append("path")
      .attr("class", "symbolic-prediction")
      .style("fill", "none")
      .style("stroke", "#f59322")
      .style("stroke-width", "2px")
      .style("stroke-dasharray", "6,3");

    this.basisGroup = this.svg.append("g")
      .attr("class", "symbolic-library-basis");

    this.libraryPath = this.svg.append("path")
      .attr("class", "symbolic-library-line")
      .style("fill", "none")
      .style("stroke", SYMBOLIC_LIBRARY_COMBINED_COLOR)
      .style("stroke-width", "2px")
      .style("stroke-dasharray", "4,2")
      .style("opacity", 0);

    this.trainGroup = this.svg.append("g")
      .attr("class", "symbolic-train");

    this.testGroup = this.svg.append("g")
      .attr("class", "symbolic-test");
  }

  setVisible(visible: boolean): void {
    this.container.style("display", visible ? null : "none");
  }

  clear(): void {
    this.targetPath.attr("d", "");
    this.predictedPath.attr("d", "");
    this.libraryPath.attr("d", "").style("opacity", 0);
    this.basisGroup.selectAll("path").remove();
    this.trainGroup.selectAll("circle").remove();
    this.testGroup.selectAll("circle").remove();
  }

  update(samples: SymbolicSample[], train: Example2D[], test: Example2D[], scale: number, fit?: SymbolicLibraryFit | null): void {
    if (!samples || samples.length === 0) {
      this.clear();
      return;
    }

    let yValues: number[] = [];
    samples.forEach(sample => {
      yValues.push(sample.target);
      yValues.push(sample.predicted);
    });
    if (fit && fit.combined && fit.combined.length) {
      fit.combined.forEach(point => yValues.push(point.y));
      fit.basis.forEach(b => b.samples.forEach(point => yValues.push(point.y)));
    }
    if (train) {
      train.forEach(point => yValues.push(point.label * scale));
    }
    if (test) {
      test.forEach(point => yValues.push(point.label * scale));
    }

    let yMin = d3.min(yValues);
    let yMax = d3.max(yValues);
    if (yMin == null || yMax == null) {
      yMin = -1;
      yMax = 1;
    }
    if (yMax === yMin) {
      yMax += 1;
      yMin -= 1;
    }
    let padding = Math.max(0.1, (yMax - yMin) * 0.1);
    this.yScale.domain([yMin - padding, yMax + padding]);
    this.xScale.domain([-1, 1]);

    this.svg.select(".x.axis").call(this.xAxis);
    this.svg.select(".y.axis").call(this.yAxis);

    let targetLine = d3.svg.line<SymbolicSample>()
      .x(d => this.xScale(d.x))
      .y(d => this.yScale(d.target))
      .interpolate("basis");

    let predictedLine = d3.svg.line<SymbolicSample>()
      .x(d => this.xScale(d.x))
      .y(d => this.yScale(d.predicted))
      .interpolate("basis");

    this.targetPath.datum(samples).attr("d", targetLine);
    this.predictedPath.datum(samples).attr("d", predictedLine);

    if (fit && fit.combined && fit.combined.length) {
      let combinedLine = d3.svg.line<SymbolicCurvePoint>()
        .x(d => this.xScale(d.x))
        .y(d => this.yScale(d.y))
        .interpolate("basis");
      this.libraryPath
        .datum(fit.combined)
        .attr("d", combinedLine)
        .style("opacity", 1);

      let basisLine = d3.svg.line<SymbolicCurvePoint>()
        .x(d => this.xScale(d.x))
        .y(d => this.yScale(d.y))
        .interpolate("basis");

      let basisSelection = this.basisGroup.selectAll("path")
        .data(fit.basis, (d: any) => d.entry.id);

      basisSelection.enter().append("path")
        .attr("class", "symbolic-library-basis-line")
        .style("fill", "none")
        .style("stroke-width", "1.5px")
        .style("stroke-dasharray", "2,2")
        .style("opacity", 0.7);

      basisSelection
        .style("stroke", (d: SymbolicLibraryBasisResult) => d.color)
        .attr("d", (d: SymbolicLibraryBasisResult) => basisLine(d.samples));

      basisSelection.exit().remove();
    } else {
      this.libraryPath.attr("d", "").style("opacity", 0);
      this.basisGroup.selectAll("path").remove();
    }

    let trainSelection = this.trainGroup.selectAll("circle").data(train || []);
    trainSelection.enter().append("circle")
      .attr("r", 2.5)
      .style("fill", "#555")
      .style("opacity", 0.7);
    trainSelection
      .attr("cx", d => this.xScale(d.x))
      .attr("cy", d => this.yScale(d.label * scale));
    trainSelection.exit().remove();

    let testSelection = this.testGroup.selectAll("circle").data(test || []);
    testSelection.enter().append("circle")
      .attr("r", 2.5)
      .style("fill", "#999")
      .style("opacity", 0.7);
    testSelection
      .attr("cx", d => this.xScale(d.x))
      .attr("cy", d => this.yScale(d.label * scale));
    testSelection.exit().remove();
  }
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
      return false;  // Not done.
    }, 0);
  }
}

let state = State.deserializeState();

// // Backward-compat: migrate gridSize -> numControlPoints if needed
// if ((state as any).numControlPoints == null) {
//   const gs = (state as any).gridSize != null ? Math.floor((state as any).gridSize) : 5;
//   const deg = Math.floor((state as any).degree != null ? (state as any).degree : 3);
//   (state as any).numControlPoints = Math.max(gs + 1, deg + 1);
// }

// Filter out inputs that are hidden.
state.getHiddenProps().forEach(prop => {
  if (prop in INPUTS) {
    delete INPUTS[prop];
  }
});

let boundary: { [id: string]: number[][] } = {};
let selectedNodeId: string = null;
// Plot the heatmap.
let xDomain: [number, number] = [-1, 1];
let heatMap =
  new HeatMap(300, DENSITY, xDomain, xDomain, d3.select("#heatmap"),
    { showAxes: true });
let linkWidthScale = d3.scale.linear()
  .domain([0, 1])
  .range([0, 5]) // 0, 10
  .clamp(true);
let colorScale = d3.scale.linear<string, number>()
  .domain([-1, 0, 1])
  .range(["#f59322", "#e8eaeb", "#0877bd"])
  .clamp(true);

let linkColorScale = d3.scale.linear<string, number>()
  .domain([-1, 0, 1])
  .range(["#ffffff", "#e8eaeb", "#6B6B6B"]) // #1B998B #7D2E68 #5A5A5A
  .clamp(true);

let iter = 0;
let trainData: Example2D[] = [];
let testData: Example2D[] = [];
let network: kan.KANNode[][] = null;
let lossTrain = 0;
let lossTest = 0;
let player = new Player();
let lineChart = new AppendingLineChart(d3.select("#linechart"),
  ["#777", "black"]);
// Add spline chart variable
let splineChart: SplineChart = null;
let edgeSplineCharts: { [edgeId: string]: SplineChart } = {};
// Hover card spline chart
let hoverCardSplineChart: SplineChart = null;
let hoverCardLibraryInfo: d3.Selection<any> | null = null;
// Current edge being displayed in hover card
let currentHoverCardEdge: kan.KANEdge = null;
let symbolicResult: SymbolicDatasetResult | null = null;
let symbolicPlot: SymbolicPlot | null = null;
let symbolicComposition: SymbolicCompositionSummary | null = null;
// Layout manager for coordinating network positions
let layoutManager: NetworkLayoutManager | null = null;
// Timeout for hiding hover card with delay
let hoverCardHideTimeout: number | null = null;
// Track if user is currently dragging a control point
let isDraggingControlPoint = false;
// Track cursor position during interactions
let lastMousePosition = { x: 0, y: 0 };

function makeGUI() {
  d3.select("#reset-button").on("click", () => {
    reset();
    userHasInteracted();
    d3.select("#play-pause-button");
  });

  d3.select("#play-pause-button").on("click", function () {
    // Change the button's content.
    userHasInteracted();
    player.playOrPause();
  });

  player.onPlayPause(isPlaying => {
    d3.select("#play-pause-button").classed("playing", isPlaying);
  });

  d3.select("#next-step-button").on("click", () => {
    player.pause();
    userHasInteracted();
    if (iter === 0) {
      simulationStarted();
    }
    oneStep();
  });

  d3.select("#data-regen-button").on("click", () => {
    if (generateData()) {
      parametersChanged = true;
    }
  });

  let dataThumbnails = d3.selectAll("canvas[data-dataset]");
  dataThumbnails.on("click", function (this: HTMLCanvasElement) {
    const key = this.dataset["dataset"];
    if (!key) {
      return;
    }
    let newDataset = datasets[key];
    if (newDataset === state.dataset) {
      return; // No-op.
    }
    state.dataset = newDataset;
    dataThumbnails.classed("selected", false);
    d3.select(this).classed("selected", true);
    if (generateData()) {
      parametersChanged = true;
      reset();
    }
  });

  let datasetKey = getKeyFromValue(datasets, state.dataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-dataset=${datasetKey}]`)
    .classed("selected", true);

  let regDataThumbnails = d3.selectAll("canvas[data-regDataset]");
  regDataThumbnails.on("click", function (this: HTMLCanvasElement) {
    const key = this.dataset["regdataset"];
    if (!key) {
      return;
    }
    let newDataset = regDatasets[key];
    if (newDataset === state.regDataset) {
      return; // No-op.
    }
    state.regDataset = newDataset;
    regDataThumbnails.classed("selected", false);
    d3.select(this).classed("selected", true);
    if (generateData()) {
      parametersChanged = true;
      reset();
    }
  });

  let regDatasetKey = getKeyFromValue(regDatasets, state.regDataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-regDataset=${regDatasetKey}]`)
    .classed("selected", true);

  d3.select("#add-layers").on("click", () => {
    if (state.numHiddenLayers >= 5) {
      return;
    }
    state.networkShape[state.numHiddenLayers] = 1;
    state.numHiddenLayers++;
    parametersChanged = true;
    reset();
  });

  d3.select("#remove-layers").on("click", () => {
    if (state.numHiddenLayers <= 0) {
      return;
    }
    state.numHiddenLayers--;
    state.networkShape.splice(state.numHiddenLayers);
    parametersChanged = true;
    reset();
  });

  let showTestData = d3.select("#show-test-data").on("change", function (this: HTMLInputElement) {
    state.showTestData = this.checked;
    state.serialize();
    userHasInteracted();
    heatMap.updateTestPoints(state.showTestData ? testData : []);
  });
  // Check/uncheck the checkbox according to the current state.
  showTestData.property("checked", state.showTestData);

  let discretize = d3.select("#discretize").on("change", function (this: HTMLInputElement) {
    state.discretize = this.checked;
    state.serialize();
    userHasInteracted();
    updateUI();
  });
  // Check/uncheck the checbox according to the current state.
  discretize.property("checked", state.discretize);

  let percTrain = d3.select("#percTrainData").on("input", function (this: HTMLInputElement) {
    state.percTrainData = +this.value;
    d3.select("label[for='percTrainData'] .value").text(this.value);
    if (generateData()) {
      parametersChanged = true;
      reset();
    }
  });
  percTrain.property("value", state.percTrainData);
  d3.select("label[for='percTrainData'] .value").text(state.percTrainData);

  let noise = d3.select("#noise").on("input", function (this: HTMLInputElement) {
    state.noise = +this.value;
    d3.select("label[for='noise'] .value").text(this.value);
    if (generateData()) {
      parametersChanged = true;
      reset();
    }
  });
  let currentMax = parseInt(noise.property("max"));
  if (state.noise > currentMax) {
    if (state.noise <= 80) {
      noise.property("max", state.noise);
    } else {
      state.noise = 50;
    }
  } else if (state.noise < 0) {
    state.noise = 0;
  }
  noise.property("value", state.noise);
  d3.select("label[for='noise'] .value").text(state.noise);

  let batchSize = d3.select("#batchSize").on("input", function (this: HTMLInputElement) {
    state.batchSize = +this.value;
    d3.select("label[for='batchSize'] .value").text(this.value);
    parametersChanged = true;
    reset();
  });
  batchSize.property("value", state.batchSize);
  d3.select("label[for='batchSize'] .value").text(state.batchSize);

  let problemDropdown = d3.select("#problem").on("change", function (this: HTMLSelectElement) {
    state.problem = problems[this.value];
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
    updateProblemSpecificUI();
    const ok = generateData(false, { allowSymbolicFallback: true });
    drawDatasetThumbnails();
    if (ok) {
      reset();
    }
  });
  problemDropdown.property("value",
    getKeyFromValue(problems, state.problem));

  let learningRate = d3.select("#learningRate").on("change", function (this: HTMLSelectElement) {
    state.learningRate = +this.value;
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
    // reset(); not needed
  });
  learningRate.property("value", state.learningRate);

  let symbolicExpressionInput = d3.select("#symbolicExpression");
  if (!symbolicExpressionInput.empty()) {
    symbolicExpressionInput.on("input", function (this: HTMLInputElement) {
      state.symbolicExpression = this.value;
      const ok = generateData(false, { allowSymbolicFallback: false });
      state.serialize();
      drawDatasetThumbnails();
      if (ok) {
        userHasInteracted();
        parametersChanged = true;
        reset();
      }
    });
    symbolicExpressionInput.property("value", state.symbolicExpression || "");
  }

  initSymbolicLibraryUI();
  initSymbolicCompositionToggle();
  updateSymbolicLibraryFit();

  // numControlPoints UI
  let numControlPointsSel = d3.select("#numControlPoints").on("change", function (this: HTMLSelectElement) {
    state.numControlPoints = +this.value;
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
    reset();
  });
  numControlPointsSel.property("value", state.numControlPoints);

  let degree = d3.select("#degree").on("change", function (this: HTMLSelectElement) {
    state.degree = +this.value;
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
    reset();
  });
  degree.property("value", state.degree);

  // Initialize numControlPoints options based on current degree and selection
  updateNumControlPointsOptionsForDegree(+degree.property("value"), state.numControlPoints);

  // Keep options in sync when degree changes
  d3.select("#degree").on("change.gridOptions", function () {
    const degVal = +(this as HTMLSelectElement).value;
    const currentCP = +d3.select("#numControlPoints").property("value") || state.numControlPoints;
    updateNumControlPointsOptionsForDegree(degVal, currentCP);
  });

  // Add initNoise control
  const initNoiseSel = d3.select("#initNoise");

  let initNoise = initNoiseSel.on("change", function () {
    const raw = (this as HTMLSelectElement).value;
    const parsed = parseFloat(raw);
    (state as any).initNoise = isNaN(parsed) ? (raw as any) : parsed;
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
    reset();
  });
  // Ensure the select reflects current state (stringify numbers)
  initNoise.property("value", String(state.initNoise));

  // Add scale to the gradient color map.
  let x = d3.scale.linear().domain([-1, 1]).range([0, 144]);
  let xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom")
    .tickValues([-1, 0, 1])
    .tickFormat(d3.format("d"));
  d3.select("#colormap g.core").append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0,10)")
    .call(xAxis);

  // Listen for css-responsive changes and redraw the svg network.
  window.addEventListener("resize", () => {
    let newWidth = document.querySelector("#main-part")
      .getBoundingClientRect().width;
    if (newWidth !== mainWidth) {
      mainWidth = newWidth;
      drawNetwork(network);
      updateUI(true);
    }
  });

  // Hide the text below the visualization depending on the URL.
  if (state.hideText) {
    d3.select("#article-text").style("display", "none");
    d3.select("div.more").style("display", "none");
    d3.select("header").style("display", "none");
  }

  updateProblemSpecificUI();
}

function updateWeightsUI(network: kan.KANNode[][], container) {
  // Use fixed boundaries for activation std (reasonable values based on typical activation ranges)
  const inputStdBoundary = 0.6;  // Typical std for inputs in [-1, 1] range
  const outputStdBoundary = 0.6; // Typical std for outputs in [-1, 1] range

  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];

    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputEdges.length; j++) {
        let edge = node.inputEdges[j];
        let edgeId = `${edge.sourceNode.id}-${edge.destNode.id}`;

        // Use absolute std values clamped to [0, 1] range based on boundaries
        let inputStd = edge.getInputActivationStd();
        let outputStd = edge.getOutputActivationStd();
        let normalizedInputStd = Math.min(1, inputStd / inputStdBoundary);
        let normalizedOutputStd = Math.min(1, outputStd / outputStdBoundary);

        // If edge is inactive, make links very faint
        const opacity = edge.isActive ? 1.0 : 0.2;

        // Update the first link (source to spline chart) - use input activation std
        container.select(`#link${edgeId}-part1`)
          .style({
            "stroke-dashoffset": -iter / 3,
            "stroke-width": linkWidthScale(normalizedInputStd),
            "stroke": linkColorScale(normalizedInputStd),
            "opacity": opacity
          });

        // Update the second link (spline chart to destination) - use output activation std
        container.select(`#link${edgeId}-part2`)
          .style({
            "stroke-dashoffset": -iter / 3,
            "stroke-width": linkWidthScale(normalizedOutputStd),
            "stroke": linkColorScale(normalizedOutputStd),
            "opacity": opacity
          });

        // Update the spline chart
        if (edgeSplineCharts[edgeId]) {
          edgeSplineCharts[edgeId].updateFunction(edge.learnableFunction);
        }

        // Update hover card spline chart if it's showing this edge
        if (currentHoverCardEdge && hoverCardSplineChart &&
          currentHoverCardEdge === edge) {
          const inputHistogramData = edge.getNormalizedHistogram();
          const outputHistogramData = edge.getNormalizedOutputHistogram();
          hoverCardSplineChart.updateFunction(edge.learnableFunction, inputHistogramData, outputHistogramData);
          updateHoverCardSymbolicOverlay();
        }
      }
    }
  }
}

function drawNode(cx: number, cy: number, nodeId: string, isInput: boolean,
  container, node?: kan.KANNode, isOutput: boolean = false) {
  let x = cx - RECT_SIZE / 2;
  let y = cy - RECT_SIZE / 2;

  let nodeGroup = container.append("g")
    .attr({
      "class": "node",
      "id": `node${nodeId}`,
      "transform": `translate(${x},${y})`
    });

  // Draw the main rectangle.
  nodeGroup.append("rect")
    .attr({
      x: 0,
      y: 0,
      width: RECT_SIZE,
      height: RECT_SIZE,
    });
  let activeOrNotClass = isInputFeatureActive(nodeId) ? "active" : "inactive";
  if (isInput) {
    let label = INPUTS[nodeId].label != null ?
      INPUTS[nodeId].label : nodeId;
    // Draw the input label.
    let text = nodeGroup.append("text").attr({
      class: "main-label",
      x: -10,
      y: RECT_SIZE / 2, "text-anchor": "end"
    });
    if (/[_^]/.test(label)) {
      let myRe = /(.*?)([_^])(.)/g;
      let myArray;
      let lastIndex;
      while ((myArray = myRe.exec(label)) != null) {
        lastIndex = myRe.lastIndex;
        let prefix = myArray[1];
        let sep = myArray[2];
        let suffix = myArray[3];
        if (prefix) {
          text.append("tspan").text(prefix);
        }
        text.append("tspan")
          .attr("baseline-shift", sep === "_" ? "sub" : "super")
          .style("font-size", "9px")
          .text(suffix);
      }
      if (label.substring(lastIndex)) {
        text.append("tspan").text(label.substring(lastIndex));
      }
    } else {
      text.append("tspan").text(label);
    }
    nodeGroup.classed(activeOrNotClass, true);
  }

  // Draw the node's canvas.
  let div = d3.select("#network").insert("div", ":first-child")
    .attr({
      "id": `canvas-${nodeId}`,
      "class": "canvas"
    })
    .style({
      position: "absolute",
      left: `${x + 3}px`,
      top: `${y + 3}px`,
      display: isOutput ? "none" : null
    })
    .on("mouseenter", function () {
      selectedNodeId = nodeId;
      div.classed("hovered", true);
      nodeGroup.classed("hovered", true);
      updateDecisionBoundary(network, false);
      heatMap.updateBackground(boundary[nodeId], state.discretize);
    })
    .on("mouseleave", function () {
      selectedNodeId = null;
      div.classed("hovered", false);
      nodeGroup.classed("hovered", false);
      updateDecisionBoundary(network, false);
      heatMap.updateBackground(boundary[kan.getKANOutputNode(network).id],
        state.discretize);
    });
  if (isInput) {
    let selectable = state.problem !== Problem.SYMBOLIC || !!SYMBOLIC_ALLOWED_INPUTS[nodeId];
    if (selectable) {
      div.on("click", function () {
        state[nodeId] = !state[nodeId];
        parametersChanged = true;
        reset();
      });
      div.style("cursor", "pointer");
    } else {
      state[nodeId] = !!SYMBOLIC_ALLOWED_INPUTS[nodeId];
      div.on("click", null);
      div.classed("disabled", true);
      div.style("cursor", "not-allowed");
    }
  }
  if (isInput) {
    div.classed("active", isInputFeatureActive(nodeId));
    div.classed("inactive", !isInputFeatureActive(nodeId));
  }
  let nodeHeatMap = new HeatMap(RECT_SIZE, DENSITY / 10, xDomain,
    xDomain, div, { noSvg: true });
  div.datum({ heatmap: nodeHeatMap, id: nodeId });

}

// Draw network
function drawNetwork(network: kan.KANNode[][]): void {
  let svg = d3.select("#svg");
  // Remove all svg elements.
  svg.select("g.core").remove();
  // Remove all div elements.
  d3.select("#network").selectAll("div.canvas").remove();
  d3.select("#network").selectAll("div.plus-minus-neurons").remove();
  d3.select("#network").selectAll("div.spline-chart").remove();

  // Clear edge spline charts
  edgeSplineCharts = {};

  // Get the width of the svg container.
  let padding = 3;
  let co = d3.select(".column.output").node() as HTMLDivElement;
  let cf = d3.select(".column.features").node() as HTMLDivElement;
  let width = co.offsetLeft - cf.offsetLeft;
  svg.attr("width", width);

  const nodeIds = Object.keys(INPUTS);

  layoutManager = new NetworkLayoutManager(
    RECT_SIZE,
    NODE_SPACING,
    padding,
    width,
    600,
    SPLINE_CHART_SIZE_X,
    SPLINE_CHART_SIZE_Y
  );
  const manager = layoutManager;

  let maxY = manager.calculateRequiredHeight(network, nodeIds.length, nodeIds);
  svg.attr("height", maxY);
  manager.updateDimensions(width, maxY);

  const { layers, node2coord, edgesByLayer } = manager.getCompleteNetworkLayout(network, nodeIds);

  let container = svg.append("g")
    .classed("core", true)
    .attr("transform", `translate(${padding},${padding})`);

  let numLayers = network.length;
  let calloutThumb = d3.select(".callout.thumbnail").style("display", "none");
  let calloutWeights = d3.select(".callout.weights").style("display", "none");
  let idWithCallout = null;
  let targetIdWithCallout = null;

  layers.forEach((layer, layerIdx) => {
    const isInputLayer = layerIdx === 0;
    const isOutputLayer = layerIdx === numLayers - 1;
    const isHiddenLayer = !isInputLayer && !isOutputLayer;

    if (isHiddenLayer) {
      const layerX = manager.calculateLayerX(layerIdx, numLayers);
      addPlusMinusControl(layerX - RECT_SIZE / 2, layerIdx);
    }

    layer.nodePositions.forEach((pos, nodeIdx) => {
      let nodeId: string;
      let node: kan.KANNode | undefined;

      if (isInputLayer) {
        nodeId = nodeIds[nodeIdx];
      } else {
        node = network[layerIdx][nodeIdx];
        nodeId = node.id;
      }

      drawNode(pos.cx, pos.cy, nodeId, isInputLayer, container, node, isOutputLayer);

      if (isHiddenLayer && node) {
        const numNodes = network[layerIdx].length;
        const nextNumNodes = network[layerIdx + 1].length;
        if (idWithCallout == null &&
          nodeIdx === numNodes - 1 &&
          nextNumNodes <= numNodes) {
          calloutThumb.style({
            display: null,
            top: `${20 + padding + pos.cy}px`,
            left: `${layer.cx}px`
          });
          idWithCallout = node.id;
        }
      }
    });
  });

  for (let layerIdx = 1; layerIdx < numLayers; layerIdx++) {
    const layerEdges = edgesByLayer[layerIdx];
    if (!layerEdges || layerEdges.length === 0) {
      continue;
    }
    const currentLayer = network[layerIdx];

    layerEdges.forEach((edgeLayout, globalEdgeIdx) => {
      const { edge, splinePosition, linkPath, edgeIndex, totalEdgesForNode } = edgeLayout;

      drawLinkWithSplineChart(
        edge,
        node2coord,
        network,
        container,
        edgeIndex === 0,
        edgeIndex,
        totalEdgesForNode,
        splinePosition,
        linkPath
      );

      if (targetIdWithCallout == null && globalEdgeIdx === layerEdges.length - 1) {
        const prevLayer = network[layerIdx - 1];
        const lastNodePrevLayer = prevLayer[prevLayer.length - 1];
        if (edge.sourceNode.id === lastNodePrevLayer.id &&
          (edge.sourceNode.id !== idWithCallout || numLayers <= 5) &&
          edge.destNode.id !== idWithCallout &&
          prevLayer.length >= currentLayer.length) {
          calloutWeights.style({
            display: null,
            top: `${splinePosition.y + SPLINE_CHART_SIZE_Y / 2 + 5}px`,
            left: `${splinePosition.x + 3}px`
          });
          targetIdWithCallout = edge.destNode.id;
        }
      }
    });
  }

  let height = Math.max(
    getRelativeHeight(calloutThumb),
    getRelativeHeight(calloutWeights),
    getRelativeHeight(d3.select("#network"))
  );
  d3.select(".column.features").style("height", height + "px");

  // Position the #heatmap div to align with the output layer node
  const outputNode = kan.getKANOutputNode(network);
  const outputNodePos = node2coord[outputNode.id];
  if (outputNodePos) {
    // Calculate the top position: output node's cy position minus half the heatmap size
    // The node canvas is positioned at cy - RECT_SIZE/2, so we align the #heatmap with that
    const heatmapTop = outputNodePos.cy - RECT_SIZE / 2 + padding;
    d3.select("#heatmap").style("margin-top", `${heatmapTop}px`);
  }
}

function drawLinkWithSplineChart(
  edge: kan.KANEdge,
  node2coord: { [id: string]: { cx: number, cy: number } },
  network: kan.KANNode[][],
  container,
  isFirst: boolean,
  index: number,
  length: number,
  splinePosition: { x: number, y: number },
  linkPath?: { sourceToSpline: any, splineToTarget: any }
) {

  let source = node2coord[edge.sourceNode.id];
  let dest = node2coord[edge.destNode.id];
  let edgeId = `${edge.sourceNode.id}-${edge.destNode.id}`;

  // Get SVG dimensions and padding
  let svg = d3.select("#svg");
  let svgWidth = parseInt(svg.attr("width"));
  let svgHeight = parseInt(svg.attr("height"));
  let padding = 3;

  // Use the globally calculated spline position
  let splineX = splinePosition.x;
  let splineY = splinePosition.y;

  if (layoutManager) {
    const constrained = layoutManager.constrainToSVGBounds(
      splineX,
      splineY,
      SPLINE_CHART_SIZE_X,
      SPLINE_CHART_SIZE_Y
    );
    splineX = constrained.x;
    splineY = constrained.y;
  } else {
    let minX = padding + SPLINE_CHART_SIZE_X / 2;
    let maxX = svgWidth - padding - SPLINE_CHART_SIZE_X / 2;
    let minY = padding + SPLINE_CHART_SIZE_Y / 2;
    let maxY = svgHeight - padding - SPLINE_CHART_SIZE_Y / 2;
    splineX = Math.max(minX, Math.min(maxX, splineX));
    splineY = Math.max(minY, Math.min(maxY, splineY));
  }

  // Create spline chart div
  let splineDiv = d3.select("#network").append("div")
    .attr("class", "spline-chart")
    .attr("id", `spline-${edgeId}`)
    .style({
      position: "absolute",
      left: `${splineX - SPLINE_CHART_SIZE_X / 2 + 1}px`,
      top: `${splineY - SPLINE_CHART_SIZE_Y / 2}px`,
      width: `${SPLINE_CHART_SIZE_X}px`,
      height: `${SPLINE_CHART_SIZE_Y}px`,
      "z-index": "10",
      "pointer-events": "auto"
    });

  // Create spline chart
  let splineChart = new SplineChart(splineDiv, {
    width: SPLINE_CHART_SIZE_X,
    height: SPLINE_CHART_SIZE_Y,
    title: "",
    showControlPoints: false,
    showOldControlPaths: false,
    showKnots: false,
    showGrid: false,
    showXAxisLabels: false,
    showYAxisLabels: false,
    showXAxisValues: false,
    showYAxisValues: false,
    showBorder: true
  });

  // Store the spline chart for updates
  edgeSplineCharts[edgeId] = splineChart;

  // Update spline chart with the learnable function
  splineChart.updateFunction(edge.learnableFunction);

  // Apply initial active/inactive styling
  splineDiv.classed("inactive", !edge.isActive);

  // Add click handler to toggle active state
  splineDiv.on("click", function () {
    edge.isActive = !edge.isActive;
    splineDiv.classed("inactive", !edge.isActive);

    // Reset histograms for ALL edges in the network, not just this one
    // This ensures recalculation from scratch with the new network state
    for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
      let currentLayer = network[layerIdx];
      for (let i = 0; i < currentLayer.length; i++) {
        let node = currentLayer[i];
        for (let j = 0; j < node.inputEdges.length; j++) {
          node.inputEdges[j].resetHistogram();
        }
      }
    }

    // Repopulate histograms with current network state
    trainData.forEach((point) => {
      let input = constructInput(point.x, point.y);
      kan.kanForwardProp(network, input, true);
    });

    // Update visualizations
    updateWeightsUI(network, d3.select("g.core"));
    updateDecisionBoundary(network, false);

    let selectedId = selectedNodeId != null ?
      selectedNodeId : kan.getKANOutputNode(network).id;
    heatMap.updateBackground(boundary[selectedId], state.discretize);

    d3.select("#network").selectAll("div.canvas")
      .each(function (data: { heatmap: HeatMap, id: string }) {
        data.heatmap.updateBackground(reduceMatrix(boundary[data.id], 10),
          state.discretize);
      });

    // Update hover card if it's showing this edge
    if (currentHoverCardEdge === edge && hoverCardSplineChart) {
      const inputHistogramData = edge.getNormalizedHistogram();
      const outputHistogramData = edge.getNormalizedOutputHistogram();
      hoverCardSplineChart.updateFunction(edge.learnableFunction, inputHistogramData, outputHistogramData);
      updateHoverCardSymbolicOverlay();
    }

    // Prevent event from bubbling to hover handlers
    const clickEvent = d3.event as Event;
    if (clickEvent.stopPropagation) {
      clickEvent.stopPropagation();
    }
  });

  // Add hover functionality to spline chart div
  splineDiv.on("mouseenter", function () {
    updateHoverCard(HoverType.WEIGHT, edge, [splineX, splineY]);
  }).on("mouseleave", function () {
    // Don't hide if user is dragging a control point
    if (isDraggingControlPoint) {
      return;
    }
    // Delay hiding to allow mouse to move to hovercard
    hoverCardHideTimeout = window.setTimeout(() => {
      updateHoverCard(null);
    }, 100);
  }).on("mousemove", function () {
    // Track cursor position
    const event = d3.event as MouseEvent;
    lastMousePosition = { x: event.pageX, y: event.pageY };
  });

  // Draw first link: source node to spline chart
  let line1 = container.insert("path", ":first-child");
  let datum1;
  if (linkPath) {
    datum1 = {
      source: { y: linkPath.sourceToSpline.source.y, x: linkPath.sourceToSpline.source.x },
      target: { y: linkPath.sourceToSpline.target.y, x: linkPath.sourceToSpline.target.x }
    };
  } else {
    datum1 = {
      source: {
        y: source.cx + RECT_SIZE / 2 + 2,
        x: source.cy
      },
      target: {
        y: splineX - SPLINE_CHART_SIZE_X / 2,
        x: splineY
      }
    };
  }
  let diagonal = d3.svg.diagonal().projection(d => [d.y, d.x]);
  line1.attr({
    "marker-start": "url(#markerArrow)",
    class: "link",
    id: `link${edgeId}-part1`,
    d: diagonal(datum1, 0)
  });

  // Draw second link: spline chart to destination node
  let line2 = container.insert("path", ":first-child");
  let datum2;
  if (linkPath) {
    datum2 = {
      source: { y: linkPath.splineToTarget.source.y, x: linkPath.splineToTarget.source.x },
      target: { y: linkPath.splineToTarget.target.y, x: linkPath.splineToTarget.target.x }
    };
  } else {
    datum2 = {
      source: {
        y: splineX + SPLINE_CHART_SIZE_X / 2,
        x: splineY
      },
      target: {
        y: dest.cx - RECT_SIZE / 2,
        x: dest.cy + ((index - (length - 1) / 2) / length) * 12
      }
    };
  }
  line2.attr({
    "marker-start": "url(#markerArrow)",
    class: "link",
    id: `link${edgeId}-part2`,
    d: diagonal(datum2, 0)
  });

  // Add invisible thick links for hover detection (without hover card functionality)
  container.append("path")
    .attr("d", diagonal(datum1, 0))
    .attr("class", "link-hover");

  container.append("path")
    .attr("d", diagonal(datum2, 0))
    .attr("class", "link-hover");

  return [line1, line2];
}

/**
 * Given a KAN network, it asks the network for the output (prediction)
 * of every node in the network using inputs sampled on a square grid.
 * It returns a map where each key is the node ID and the value is a square
 * matrix of the outputs of the network for each input in the grid respectively.
 */
function updateDecisionBoundary(network: kan.KANNode[][], firstTime: boolean) {
  if (firstTime) {
    boundary = {};
    kan.forEachKANNode(network, true, node => {
      boundary[node.id] = new Array(DENSITY);
    });
    // Go through all predefined inputs.
    for (let nodeId in INPUTS) {
      boundary[nodeId] = new Array(DENSITY);
    }
  }
  let xScale = d3.scale.linear().domain([0, DENSITY - 1]).range(xDomain);
  let yScale = d3.scale.linear().domain([DENSITY - 1, 0]).range(xDomain);

  let i = 0, j = 0;
  for (i = 0; i < DENSITY; i++) {
    if (firstTime) {
      kan.forEachKANNode(network, true, node => {
        boundary[node.id][i] = new Array(DENSITY);
      });
      // Go through all predefined inputs.
      for (let nodeId in INPUTS) {
        boundary[nodeId][i] = new Array(DENSITY);
      }
    }
    for (j = 0; j < DENSITY; j++) {
      // 1 for points inside the circle, and 0 for points outside the circle.
      let x = xScale(i);
      let y = yScale(j);
      let input = constructInput(x, y);
      kan.kanForwardProp(network, input, false);
      kan.forEachKANNode(network, true, node => {
        boundary[node.id][i][j] = node.output;
      });
      if (firstTime) {
        // Go through all predefined inputs.
        for (let nodeId in INPUTS) {
          boundary[nodeId][i][j] = INPUTS[nodeId].f(x, y);
        }
      }
    }
  }
}

function getLoss(network: kan.KANNode[][], dataPoints: Example2D[]): number {
  let loss = 0;
  for (let i = 0; i < dataPoints.length; i++) {
    let dataPoint = dataPoints[i];
    let input = constructInput(dataPoint.x, dataPoint.y);
    let output = kan.kanForwardProp(network, input, false);
    loss += kan.Errors.SQUARE.error(output, dataPoint.label);
  }
  return loss / dataPoints.length;
}

function updateUI(firstStep = false) {
  // Update the links visually.
  updateWeightsUI(network, d3.select("g.core"));
  // Get the decision boundary of the network.
  updateDecisionBoundary(network, firstStep);
  let selectedId = selectedNodeId != null ?
    selectedNodeId : kan.getKANOutputNode(network).id;
  heatMap.updateBackground(boundary[selectedId], state.discretize);

  // Update all decision boundaries.
  d3.select("#network").selectAll("div.canvas")
    .each(function (data: { heatmap: HeatMap, id: string }) {
      data.heatmap.updateBackground(reduceMatrix(boundary[data.id], 10),
        state.discretize);
    });

  function zeroPad(n: number): string {
    let pad = "000000";
    return (pad + n).slice(-pad.length);
  }

  function addCommas(s: string): string {
    return s.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  function humanReadable(n: number): string {
    return n.toFixed(3);
  }

  // Update loss and iteration number.
  d3.select("#loss-train").text(humanReadable(lossTrain));
  d3.select("#loss-test").text(humanReadable(lossTest));
  d3.select("#iter-number").text(addCommas(zeroPad(iter)));
  lineChart.addDataPoint([lossTrain, lossTest]);
  updateSymbolicPlot();
}

function isInputFeatureActive(inputName: string): boolean {
  if (!state[inputName]) {
    return false;
  }
  if (state.problem === Problem.SYMBOLIC) {
    return !!SYMBOLIC_ALLOWED_INPUTS[inputName];
  }
  return true;
}

function constructInputIds(): string[] {
  let result: string[] = [];
  for (let inputName in INPUTS) {
    if (isInputFeatureActive(inputName)) {
      result.push(inputName);
    }
  }
  return result;
}

function constructInput(x: number, y: number): number[] {
  let input: number[] = [];
  for (let inputName in INPUTS) {
    if (isInputFeatureActive(inputName)) {
      input.push(INPUTS[inputName].f(x, y));
    }
  }
  return input;
}

function oneStep(): void {
  iter++;
  trainData.forEach((point, i) => {
    let input = constructInput(point.x, point.y);
    kan.kanForwardProp(network, input);
    kan.kanBackProp(network, point.label, kan.Errors.SQUARE);
    if ((i + 1) % state.batchSize === 0) {
      kan.updateKANWeights(network, state.learningRate);
    }
  });
  // Compute the loss.
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  updateUI();
}

export function getOutputWeights(network: kan.KANNode[][]): number[] {
  let weights: number[] = [];
  for (let layerIdx = 0; layerIdx < network.length - 1; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.outputEdges.length; j++) {
        let edge = node.outputEdges[j];
        // For KAN, return norm of control points
        let normWeight = Math.sqrt(edge.learnableFunction.controlPoints.reduce((sum, coeff) => sum + coeff * coeff, 0));
        weights.push(normWeight);
      }
    }
  }
  return weights;
}

function reset(onStartup = false) {
  lineChart.reset();
  state.serialize();
  if (!onStartup) {
    userHasInteracted();
  }
  player.pause();

  let suffix = state.numHiddenLayers !== 1 ? "s" : "";
  d3.select("#layers-label").text("Hidden layer" + suffix);
  d3.select("#num-layers").text(state.numHiddenLayers);

  // Make a KAN network.
  iter = 0;
  let numInputs = constructInput(0, 0).length;
  let shape = [numInputs].concat(state.networkShape).concat([1]);

  // Derive gridSize from numControlPoints
  const derivedGridSize = Math.max(1, Math.floor(state.numControlPoints) - 1);

  network = kan.buildKANNetwork(shape, constructInputIds(), derivedGridSize, state.degree, state.initNoise);

  // Populate histograms with initial forward passes using training data
  trainData.forEach((point) => {
    let input = constructInput(point.x, point.y);
    kan.kanForwardProp(network, input, true);
  });

  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  drawNetwork(network);
  updateUI(true);
};

function initTutorial() {
  if (state.tutorial == null || state.tutorial === '' || state.hideText) {
    return;
  }
  // Remove all other text.
  d3.selectAll("article div.l--body").remove();
  let tutorial = d3.select("article").append("div")
    .attr("class", "l--body");
  // Insert tutorial text.
  d3.html(`tutorials/${state.tutorial}.html`, (err, htmlFragment) => {
    if (err) {
      throw err;
    }
    tutorial.node().appendChild(htmlFragment);
    // If the tutorial has a <title> tag, set the page title to that.
    let title = tutorial.select("title");
    if (title.size()) {
      d3.select("header h1").style({
        "margin-top": "20px",
        "margin-bottom": "20px",
      })
        .text(title.text());
      document.title = title.text();
    }
  });
}

function drawDatasetThumbnails() {
  function renderThumbnail(canvas, dataGenerator) {
    let w = 100;
    let h = 100;
    canvas.setAttribute("width", w);
    canvas.setAttribute("height", h);
    let context = canvas.getContext("2d");
    let data = dataGenerator(200, 0);
    data.forEach(function (d) {
      context.fillStyle = colorScale(d.label);
      context.fillRect(w * (d.x + 1) / 2, h * (d.y + 1) / 2, 4, 4);
    });
    d3.select(canvas.parentNode).style("display", null);
  }
  d3.selectAll(".dataset").style("display", "none");

  if (state.problem === Problem.SYMBOLIC) {
    renderSymbolicThumbnail();
    return;
  }

  renderSymbolicThumbnail(null);

  if (state.problem === Problem.CLASSIFICATION) {
    for (let dataset in datasets) {
      let canvas: any =
        document.querySelector(`canvas[data-dataset=${dataset}]`);
      let dataGenerator = datasets[dataset];
      renderThumbnail(canvas, dataGenerator);
    }
  }
  if (state.problem === Problem.REGRESSION) {
    for (let regDataset in regDatasets) {
      let canvas: any =
        document.querySelector(`canvas[data-regDataset=${regDataset}]`);
      let dataGenerator = regDatasets[regDataset];
      renderThumbnail(canvas, dataGenerator);
    }
  }
}

function hideControls() {
  // Set display:none to all the UI elements that are hidden.
  let hiddenProps = state.getHiddenProps();
  hiddenProps.forEach(prop => {
    let controls = d3.selectAll(`.ui-${prop}`);
    if (controls.size() === 0) {
      console.warn(`0 html elements found with class .ui-${prop}`);
    }
    controls.style("display", "none");
  });

  // Also add checkbox for each hidable control in the "use it in classrom"
  // section.
  let hideControls = d3.select(".hide-controls");
  HIDABLE_CONTROLS.forEach(([text, id]) => {
    let label = hideControls.append("label")
      .attr("class", "mdl-checkbox mdl-js-checkbox mdl-js-ripple-effect");
    let input = label.append("input")
      .attr({
        type: "checkbox",
        class: "mdl-checkbox__input",
      });
    if (hiddenProps.indexOf(id) === -1) {
      input.attr("checked", "true");
    }
    input.on("change", function () {
      state.setHideProperty(id, !this.checked);
      state.serialize();
      userHasInteracted();
      d3.select(".hide-controls-link")
        .attr("href", window.location.href);
    });
    label.append("span")
      .attr("class", "mdl-checkbox__label label")
      .text(text);
  });
  d3.select(".hide-controls-link")
    .attr("href", window.location.href);
}

function generateData(firstTime = false, options: { allowSymbolicFallback?: boolean } = {}): boolean {
  let newSeed = state.seed;
  if (!firstTime) {
    newSeed = Math.random().toFixed(5);
  }
  Math.seedrandom(newSeed);

  let numSamples: number;
  let data: Example2D[] = [];

  if (state.problem === Problem.SYMBOLIC) {
    numSamples = NUM_SAMPLES_SYMBOLIC;
    const allowFallback = options.allowSymbolicFallback != null ?
      options.allowSymbolicFallback : firstTime;
    symbolicResult = generateSymbolicDataset(
      numSamples,
      state.noise / 100,
      state.symbolicExpression,
      allowFallback);
    updateSymbolicExpressionFeedback();
    if (!symbolicResult || !symbolicResult.isValid) {
      symbolicLibraryFit = null;
      refreshSymbolicLibraryUI();
      return false;
    }
    updateSymbolicLibraryFit();
    data = symbolicResult.data;
  } else {
    symbolicResult = null;
    symbolicLibraryFit = null;
    numSamples = (state.problem === Problem.REGRESSION) ?
      NUM_SAMPLES_REGRESS : NUM_SAMPLES_CLASSIFY;
    let generator = state.problem === Problem.CLASSIFICATION ?
      state.dataset : state.regDataset;
    data = generator(numSamples, state.noise / 100);
    clearSymbolicFeedback();
  }

  if (!data || data.length === 0) {
    return false;
  }

  // Shuffle the data in-place.
  shuffle(data);
  // Split into train and test data.
  let splitIndex = Math.floor(data.length * state.percTrainData / 100);
  trainData = data.slice(0, splitIndex);
  testData = data.slice(splitIndex);
  heatMap.updatePoints(trainData);
  heatMap.updateTestPoints(state.showTestData ? testData : []);

  if (!firstTime) {
    state.seed = newSeed;
    state.serialize();
    userHasInteracted();
  }

  return true;
}

function renderSymbolicThumbnail(result: SymbolicDatasetResult | null = symbolicResult): void {
  let canvas = document.getElementById("symbolic-thumbnail") as HTMLCanvasElement;
  if (!canvas) {
    return;
  }
  const width = canvas.width = 200;
  const height = canvas.height = 100;
  let ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fafafa";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#d0d0d0";
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();

  if (!result || !result.isValid) {
    return;
  }

  const samples: number[] = [];
  let maxAbs = 0;
  for (let i = 0; i < SYMBOLIC_PREVIEW_SAMPLES; i++) {
    const x = -1 + (2 * i) / Math.max(1, SYMBOLIC_PREVIEW_SAMPLES - 1);
    let value = 0;
    try {
      value = result.evaluator(x);
      if (!isFinite(value) || isNaN(value)) {
        value = 0;
      }
    } catch (e) {
      value = 0;
    }
    maxAbs = Math.max(maxAbs, Math.abs(value));
    samples.push(value);
  }
  if (!isFinite(maxAbs) || maxAbs < 1e-6) {
    maxAbs = 1;
  }

  ctx.strokeStyle = "#0877bd";
  ctx.lineWidth = 2;
  ctx.beginPath();
  samples.forEach((val, idx) => {
    const xPos = (idx / Math.max(1, SYMBOLIC_PREVIEW_SAMPLES - 1)) * width;
    const yPos = height / 2 - (val / maxAbs) * (height * 0.4);
    if (idx === 0) {
      ctx.moveTo(xPos, yPos);
    } else {
      ctx.lineTo(xPos, yPos);
    }
  });
  ctx.stroke();
}

function updateSymbolicErrorMessage(result: SymbolicDatasetResult | null = symbolicResult): void {
  let container = document.getElementById("symbolic-error");
  if (!container) {
    return;
  }
  if (!result) {
    container.textContent = "";
    (container as HTMLElement).style.display = "none";
    return;
  }
  let messages: string[] = [];
  if (result.compileError) {
    messages.push(result.compileError);
  }
  if (result.usedFallback) {
    messages.push("Using fallback sin(pi*x) function.");
  }
  if (messages.length === 0) {
    container.textContent = "";
    (container as HTMLElement).style.display = "none";
    return;
  }
  container.textContent = messages.join(" ");
  (container as HTMLElement).style.display = "block";
}

function updateSymbolicSanitizedExpression(result: SymbolicDatasetResult | null = symbolicResult): void {
  let wrapper = document.getElementById("symbolic-expression-display");
  let codeEl = document.getElementById("symbolic-sanitized-expression");
  if (!wrapper || !codeEl) {
    return;
  }
  if (!result || !result.sanitizedExpression || !result.isValid) {
    wrapper.style.display = "none";
    codeEl.textContent = "";
    return;
  }
  wrapper.style.display = "block";
  codeEl.textContent = result.sanitizedExpression;
}

function updateSymbolicExpressionFeedback(): void {
  renderSymbolicThumbnail();
  updateSymbolicErrorMessage();
  updateSymbolicSanitizedExpression();
}

function clearSymbolicFeedback(): void {
  renderSymbolicThumbnail(null);
  updateSymbolicErrorMessage(null);
  updateSymbolicSanitizedExpression(null);
}

function ensureSymbolicPlot(): SymbolicPlot | null {
  if (!symbolicPlot) {
    const container = d3.select("#symbolic-plot");
    if (container.empty()) {
      return null;
    }
    symbolicPlot = new SymbolicPlot(container);
  }
  return symbolicPlot;
}

function updateSymbolicPlot(): void {
  if (state.problem !== Problem.SYMBOLIC) {
    if (symbolicPlot) {
      symbolicPlot.clear();
      symbolicPlot.setVisible(false);
    }
    updateSymbolicComposition();
    return;
  }
  let plot = ensureSymbolicPlot();
  if (!plot) {
    return;
  }
  plot.setVisible(true);
  if (!symbolicResult || !symbolicResult.isValid) {
    plot.clear();
    return;
  }
  let scale = symbolicResult.normalizationScale || 1;
  let samples: SymbolicSample[] = [];
  for (let i = 0; i < SYMBOLIC_PREVIEW_SAMPLES; i++) {
    const x = -1 + (2 * i) / Math.max(1, SYMBOLIC_PREVIEW_SAMPLES - 1);
    let target = 0;
    try {
      target = symbolicResult.evaluator(x);
      if (!isFinite(target) || isNaN(target)) {
        target = 0;
      }
    } catch (e) {
      target = 0;
    }
    let targetNormalized = 0;
    try {
      targetNormalized = symbolicResult.normalizedEvaluator(x);
      if (!isFinite(targetNormalized) || isNaN(targetNormalized)) {
        targetNormalized = Math.max(-1, Math.min(1, target / scale));
      }
    } catch (e) {
      targetNormalized = Math.max(-1, Math.min(1, target / scale));
    }
    const input = constructInput(x, 0);
    const predictedNormalized = kan.kanForwardProp(network, input);
    const predicted = predictedNormalized * scale;
    samples.push({ x, target, targetNormalized, predicted });
  }
  const fit = updateSymbolicLibraryFit(samples);
  plot.update(samples, trainData, testData, scale, fit);
  updateSymbolicComposition();
}

function initSymbolicLibraryUI(): void {
  renderSymbolicLibraryCatalog();
  renderSymbolicLibrarySelection();
}

function getSymbolicLibraryEntry(id: string): SymbolicLibraryEntry | undefined {
  for (let i = 0; i < SYMBOLIC_LIBRARY_ENTRIES.length; i++) {
    if (SYMBOLIC_LIBRARY_ENTRIES[i].id === id) {
      return SYMBOLIC_LIBRARY_ENTRIES[i];
    }
  }
  return undefined;
}

function getSymbolicLibraryColor(id: string): string {
  let index = 0;
  for (let i = 0; i < SYMBOLIC_LIBRARY_ENTRIES.length; i++) {
    if (SYMBOLIC_LIBRARY_ENTRIES[i].id === id) {
      index = i;
      break;
    }
  }
  return SYMBOLIC_LIBRARY_COLORS[index % SYMBOLIC_LIBRARY_COLORS.length];
}

function isLibrarySelected(id: string): boolean {
  return Array.isArray(state.symbolicLibrarySelection) &&
    state.symbolicLibrarySelection.indexOf(id) !== -1;
}

function renderSymbolicLibraryCatalog(): void {
  let container = d3.select("#symbolic-library");
  if (container.empty()) {
    return;
  }
  let items = container.selectAll("button.symbolic-library-item")
    .data(SYMBOLIC_LIBRARY_ENTRIES, (d: any) => d.id);

  let enterSel = items.enter().append("button")
    .attr("type", "button")
    .attr("class", "symbolic-library-item")
    .on("click", function (d: SymbolicLibraryEntry) {
      toggleSymbolicLibraryEntry(d.id);
    });

  enterSel.append("span")
    .attr("class", "symbolic-library-chip");

  enterSel.append("span")
    .attr("class", "symbolic-library-item-label");

  items
    .classed("is-selected", (d: SymbolicLibraryEntry) => isLibrarySelected(d.id))
    .attr("title", (d: SymbolicLibraryEntry) => d.expression);

  items.select(".symbolic-library-chip")
    .style("background-color", (d: SymbolicLibraryEntry) => getSymbolicLibraryColor(d.id));

  items.select(".symbolic-library-item-label")
    .text((d: SymbolicLibraryEntry) => d.label);

  items.exit().remove();
}

function renderSymbolicLibrarySelection(): void {
  let container = d3.select("#symbolic-library-selected");
  if (container.empty()) {
    return;
  }
  container.selectAll("*").remove();

  let selectedIds = Array.isArray(state.symbolicLibrarySelection) ?
    state.symbolicLibrarySelection : [];

  if (!selectedIds.length) {
    container.append("div")
      .attr("class", "symbolic-library-empty")
      .text("Select functions to build a fit.");
    return;
  }

  selectedIds.forEach(id => {
    let entry = getSymbolicLibraryEntry(id);
    if (!entry) {
      return;
    }
    let row = container.append("div")
      .attr("class", "symbolic-library-row")
      .attr("data-entry-id", id);
    row.append("span")
      .attr("class", "symbolic-library-color")
      .style("background-color", getSymbolicLibraryColor(id));

    let labelCell = row.append("div")
      .attr("class", "symbolic-library-label");
    labelCell.text(entry.label);
    labelCell.append("span")
      .attr("class", "symbolic-library-expression")
      .text(entry.expression);

    row.append("span")
      .attr("class", "symbolic-library-coefficient")
      .text("—");

    row.append("button")
      .attr("type", "button")
      .attr("class", "symbolic-library-remove")
      .text("Remove")
      .on("click", () => removeSymbolicLibraryEntry(id));
  });
}

function updateSymbolicLibraryCoefficientsDisplay(): void {
  let container = d3.select("#symbolic-library-selected");
  if (container.empty()) {
    return;
  }

  if (!symbolicLibraryFit || !symbolicLibraryFit.basis || !symbolicLibraryFit.basis.length) {
    container.selectAll(".symbolic-library-coefficient").text("—");
    container.selectAll(".symbolic-library-rmse").remove();
    return;
  }

  const coefficientMap: { [key: string]: number } = {};
  let totalMagnitude = 0;
  symbolicLibraryFit.basis.forEach(entry => {
    coefficientMap[entry.entry.id] = entry.coefficient;
    totalMagnitude += Math.abs(entry.coefficient);
  });

  container.selectAll(".symbolic-library-row").each(function () {
    const row = d3.select(this);
    const id = row.attr("data-entry-id");
    if (!id) {
      return;
    }
    const coeff = coefficientMap[id];
    row.select(".symbolic-library-coefficient")
      .text((coeff != null && isFinite(coeff)) ? formatCoefficientWithPercentage(coeff, totalMagnitude) : "—");
  });

  let rmseSection = container.select(".symbolic-library-rmse");
  if (symbolicLibraryFit.combined && symbolicLibraryFit.combined.length) {
    if (rmseSection.empty()) {
      rmseSection = container.append("div")
        .attr("class", "symbolic-library-rmse");
    }
    rmseSection.text(`RMSE: ${formatLibraryNumber(symbolicLibraryFit.rmse)}`);
  } else if (!rmseSection.empty()) {
    rmseSection.remove();
  }
}

// Convert a raw coefficient into "value (percent%)" for the library UI.
function formatCoefficientWithPercentage(value: number, totalMagnitude: number): string {
  const formattedValue = formatLibraryNumber(value);
  if (!isFinite(totalMagnitude) || totalMagnitude <= 0) {
    return formattedValue;
  }
  const contribution = Math.abs(value) / totalMagnitude;
  if (!isFinite(contribution)) {
    return formattedValue;
  }
  const percent = (contribution * 100);
  const percentText = percent >= 10 ? percent.toFixed(0) : percent.toFixed(1);
  return `${formattedValue} (${percentText.replace(/\.0$/, "")}%)`;
}

function setSymbolicBestBasis(best: SymbolicBestBasisSummary | null): void {
  symbolicBestBasis = best;
  updateHoverCardSymbolicOverlay();
}

function computeBestSingleBasis(entries: SymbolicLibraryEntry[], xs: number[], normalizationScale: number, normalizedTarget: number[]): SymbolicBestBasisSummary | null {
  if (!entries.length || !xs.length || xs.length !== normalizedTarget.length) {
    return null;
  }

  let best: SymbolicBestBasisSummary | null = null;
  const numSamples = xs.length;
  const scale = Math.max(normalizationScale || 1, 1e-6);

  entries.forEach(entry => {
    const basisValues: number[] = new Array(numSamples);
    let numerator = 0;
    let denominator = 0;
    for (let i = 0; i < numSamples; i++) {
      const val = safeLibraryValue(entry.evaluator(xs[i]) / scale);
      basisValues[i] = val;
      numerator += val * normalizedTarget[i];
      denominator += val * val;
    }
    if (!isFinite(denominator) || Math.abs(denominator) < 1e-9) {
      return;
    }
    const coefficient = numerator / denominator;
    let mse = 0;
    const samples: SymbolicCurvePoint[] = new Array(numSamples);
    for (let i = 0; i < numSamples; i++) {
      const y = coefficient * basisValues[i];
      const diff = y - normalizedTarget[i];
      mse += diff * diff;
      samples[i] = { x: xs[i], y };
    }
    const rmse = Math.sqrt(mse / numSamples);
    if (!best || rmse < best.rmse) {
      best = {
        entry,
        coefficient,
        samples,
        color: getSymbolicLibraryColor(entry.id),
        rmse
      };
    }
  });

  return best;
}

function updateHoverCardSymbolicOverlay(): void {
  if (!hoverCardSplineChart) {
    return;
  }
  if (state.problem !== Problem.SYMBOLIC) {
    hoverCardSplineChart.setOverlayCurve(null);
    if (hoverCardLibraryInfo) {
      hoverCardLibraryInfo.style("display", "none").text("");
    }
    return;
  }

  if (!symbolicBestBasis || !symbolicBestBasis.samples || !symbolicBestBasis.samples.length) {
    hoverCardSplineChart.setOverlayCurve(null);
    if (hoverCardLibraryInfo) {
      hoverCardLibraryInfo.style("display", "block")
        .text("Select at least one library function to compare.");
    }
    return;
  }

  hoverCardSplineChart.setOverlayCurve(symbolicBestBasis.samples, {
    color: symbolicBestBasis.color
  });

  if (hoverCardLibraryInfo) {
    hoverCardLibraryInfo.style("display", "block");
    hoverCardLibraryInfo.selectAll("*").remove();

    hoverCardLibraryInfo.append("div")
      .attr("class", "symbolic-hover-library-title")
      .text("Best library match");

    const body = hoverCardLibraryInfo.append("div")
      .attr("class", "symbolic-hover-library-body");

    body.append("span")
      .attr("class", "symbolic-hover-library-chip")
      .style("background-color", symbolicBestBasis.color || SYMBOLIC_LIBRARY_COMBINED_COLOR);

    body.append("span")
      .attr("class", "symbolic-hover-library-label")
      .text(`${symbolicBestBasis.entry.label}`);

    hoverCardLibraryInfo.append("div")
      .attr("class", "symbolic-hover-library-score")
      .text(`Scale factor: ${formatLibraryNumber(symbolicBestBasis.coefficient)}`);

    hoverCardLibraryInfo.append("div")
      .attr("class", "symbolic-hover-library-rmse")
      .text(`RMSE vs target: ${formatLibraryNumber(symbolicBestBasis.rmse)}`);
  }
}

function refreshSymbolicLibraryUI(): void {
  renderSymbolicLibraryCatalog();
  renderSymbolicLibrarySelection();
  updateSymbolicLibraryCoefficientsDisplay();
}

function toggleSymbolicLibraryEntry(id: string): void {
  if (!Array.isArray(state.symbolicLibrarySelection)) {
    state.symbolicLibrarySelection = [];
  }
  let next = state.symbolicLibrarySelection.slice();
  let index = next.indexOf(id);
  if (index >= 0) {
    next.splice(index, 1);
  } else {
    next.push(id);
  }
  state.symbolicLibrarySelection = next;
  state.serialize();
  userHasInteracted();
  updateSymbolicLibraryFit();
  updateSymbolicPlot();
}

function removeSymbolicLibraryEntry(id: string): void {
  if (!Array.isArray(state.symbolicLibrarySelection)) {
    return;
  }
  let next = state.symbolicLibrarySelection.filter(item => item !== id);
  if (next.length === state.symbolicLibrarySelection.length) {
    return;
  }
  state.symbolicLibrarySelection = next;
  state.serialize();
  userHasInteracted();
  updateSymbolicLibraryFit();
  updateSymbolicPlot();
}

function updateSymbolicLibraryFit(samples?: SymbolicSample[]): SymbolicLibraryFit | null {
  if (state.problem !== Problem.SYMBOLIC || !symbolicResult || !symbolicResult.isValid) {
    symbolicLibraryFit = null;
    setSymbolicBestBasis(null);
    if (!samples) {
      refreshSymbolicLibraryUI();
    } else {
      updateSymbolicLibraryCoefficientsDisplay();
    }
    return null;
  }
  let selectedIds = Array.isArray(state.symbolicLibrarySelection) ?
    state.symbolicLibrarySelection.filter(id => !!getSymbolicLibraryEntry(id)) : [];
  if (selectedIds.length === 0) {
    symbolicLibraryFit = null;
    setSymbolicBestBasis(null);
    if (!samples) {
      refreshSymbolicLibraryUI();
    } else {
      updateSymbolicLibraryCoefficientsDisplay();
    }
    return null;
  }
  let entries = selectedIds.map(id => getSymbolicLibraryEntry(id)!)
    .filter((entry): entry is SymbolicLibraryEntry => !!entry);
  if (samples && samples.length) {
    symbolicLibraryFit = computeSymbolicLibraryFitFromSamples(entries, samples);
    updateSymbolicLibraryCoefficientsDisplay();
  } else {
    symbolicLibraryFit = computeSymbolicLibraryFitFromEvaluator(entries, symbolicResult);
    refreshSymbolicLibraryUI();
  }
  return symbolicLibraryFit;
}

function computeSymbolicLibraryFitFromEvaluator(entries: SymbolicLibraryEntry[], result: SymbolicDatasetResult): SymbolicLibraryFit | null {
  if (!entries.length || !result || !result.evaluator) {
    setSymbolicBestBasis(null);
    return null;
  }
  let xs: number[] = [];
  let target: number[] = [];
  let normalizedTarget: number[] = [];
  const normalizationScale = result.normalizationScale || 1;
  for (let i = 0; i < SYMBOLIC_PREVIEW_SAMPLES; i++) {
    const x = -1 + (2 * i) / Math.max(1, SYMBOLIC_PREVIEW_SAMPLES - 1);
    xs.push(x);
    target.push(safeLibraryValue(result.evaluator(x)));
    normalizedTarget.push(safeLibraryValue(result.normalizedEvaluator ? result.normalizedEvaluator(x) : result.evaluator(x) / normalizationScale));
  }
  setSymbolicBestBasis(computeBestSingleBasis(entries, xs, normalizationScale, normalizedTarget));
  return computeSymbolicLibraryFitInternal(entries, xs, target);
}

function computeSymbolicLibraryFitFromSamples(entries: SymbolicLibraryEntry[], samples: SymbolicSample[]): SymbolicLibraryFit | null {
  if (!entries.length || !samples || !samples.length) {
    setSymbolicBestBasis(null);
    return null;
  }
  const xs = samples.map(sample => sample.x);
  const targetRaw = samples.map(sample => safeLibraryValue(sample.target));
  const targetNormalized = samples.map(sample => safeLibraryValue(sample.targetNormalized));
  const normalizationScale = symbolicResult ? (symbolicResult.normalizationScale || 1) : 1;
  setSymbolicBestBasis(computeBestSingleBasis(entries, xs, normalizationScale, targetNormalized));
  return computeSymbolicLibraryFitInternal(entries, xs, targetRaw);
}

function computeSymbolicLibraryFitInternal(entries: SymbolicLibraryEntry[], xs: number[], target: number[]): SymbolicLibraryFit | null {
  if (!entries.length || !xs.length || xs.length !== target.length) {
    return null;
  }

  let design = entries.map(entry => xs.map(x => safeLibraryValue(entry.evaluator(x))));
  let n = entries.length;
  let ata: number[][] = [];
  let atb: number[] = [];
  for (let i = 0; i < n; i++) {
    ata[i] = new Array(n);
    for (let j = 0; j < n; j++) {
      ata[i][j] = 0;
    }
    atb[i] = 0;
  }

  for (let row = 0; row < xs.length; row++) {
    let t = target[row];
    for (let i = 0; i < n; i++) {
      let vi = design[i][row];
      atb[i] += vi * t;
      for (let j = 0; j < n; j++) {
        ata[i][j] += vi * design[j][row];
      }
    }
  }

  for (let i = 0; i < n; i++) {
    ata[i][i] += SYMBOLIC_LIBRARY_REGULARIZATION;
  }

  let coefficients = solveLinearSystem(ata, atb);
  if (!coefficients) {
    return null;
  }

  let combined: SymbolicCurvePoint[] = xs.map((x, idx) => {
    let y = 0;
    for (let j = 0; j < n; j++) {
      y += coefficients[j] * design[j][idx];
    }
    return { x, y };
  });

  let basis: SymbolicLibraryBasisResult[] = entries.map((entry, idx) => {
    return {
      entry,
      coefficient: coefficients[idx],
      samples: xs.map((x, sampleIdx) => ({
        x,
        y: coefficients[idx] * design[idx][sampleIdx]
      })),
      color: getSymbolicLibraryColor(entry.id)
    };
  });

  let mse = 0;
  for (let idx = 0; idx < combined.length; idx++) {
    let diff = combined[idx].y - target[idx];
    mse += diff * diff;
  }
  let rmse = combined.length ? Math.sqrt(mse / combined.length) : 0;

  return { combined, basis, rmse };
}

function solveLinearSystem(matrix: number[][], vector: number[]): number[] | null {
  let n = vector.length;
  if (n === 0) {
    return [];
  }
  let aug: number[][] = matrix.map((row, i) => {
    let copy = row.slice();
    copy.push(vector[i]);
    return copy;
  });

  for (let col = 0; col < n; col++) {
    let pivotRow = col;
    let pivotVal = Math.abs(aug[col][col]);
    for (let row = col + 1; row < n; row++) {
      let testVal = Math.abs(aug[row][col]);
      if (testVal > pivotVal) {
        pivotVal = testVal;
        pivotRow = row;
      }
    }
    if (pivotVal < 1e-12) {
      return null;
    }
    if (pivotRow !== col) {
      let tmp = aug[col];
      aug[col] = aug[pivotRow];
      aug[pivotRow] = tmp;
    }

    let pivot = aug[col][col];
    for (let k = col; k <= n; k++) {
      aug[col][k] /= pivot;
    }

    for (let row = 0; row < n; row++) {
      if (row === col) {
        continue;
      }
      let factor = aug[row][col];
      if (factor === 0) {
        continue;
      }
      for (let k = col; k <= n; k++) {
        aug[row][k] -= factor * aug[col][k];
      }
    }
  }

  let solution: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    solution[i] = aug[i][n];
  }
  return solution;
}

function safeLibraryValue(value: number): number {
  if (!isFinite(value) || isNaN(value)) {
    return 0;
  }
  return value;
}

function formatLibraryNumber(value: number): string {
  if (!isFinite(value)) {
    return "—";
  }
  let absVal = Math.abs(value);
  if (absVal >= 1000 || (absVal > 0 && absVal < 1e-3)) {
    return value.toExponential(2);
  }
  return value.toFixed(3);
}

function updateSymbolicComposition(): void {
  if (state.problem !== Problem.SYMBOLIC) {
    symbolicComposition = null;
    renderSymbolicCompositionPanel(null);
    return;
  }
  symbolicComposition = computeSymbolicCompositionSummary();
  renderSymbolicCompositionPanel(symbolicComposition);
}

function renderSymbolicCompositionPanel(summary: SymbolicCompositionSummary | null): void {
  const panel = d3.select("#symbolic-composition-panel");
  if (panel.empty()) {
    return;
  }
  const body = panel.select(".symbolic-composition-body");
  if (body.empty()) {
    return;
  }
  const expressionEl = body.select("#symbolic-composition-expression");
  const layersContainer = body.select("#symbolic-composition-layers");
  const emptyEl = body.select("#symbolic-composition-empty");

  if (!summary) {
    expressionEl.text("");
    layersContainer.selectAll("*").remove();
    let message = "";
    if (state.problem !== Problem.SYMBOLIC) {
      message = "Switch to symbolic regression to inspect compositions.";
    } else if (!symbolicResult || !symbolicResult.isValid) {
      message = "Enter a valid symbolic expression to inspect compositions.";
    } else if (!state.symbolicLibrarySelection || !state.symbolicLibrarySelection.length) {
      message = "Activate at least one library function to match splines.";
    } else {
      message = "No active spline matches yet.";
    }
    emptyEl.text(message).style("display", "block");
    panel.classed("has-data", false);
    return;
  }

  emptyEl.style("display", "none");
  panel.classed("has-data", true);
  expressionEl.text(`f(x) ≈ ${summary.expression}`);

  const layers = layersContainer.selectAll(".symbolic-composition-layer")
    .data(summary.layers, (d: any) => d.layerIndex);

  const layersEnter = layers.enter()
    .append("div")
    .attr("class", "symbolic-composition-layer");

  layersEnter.append("div")
    .attr("class", "symbolic-composition-layer-title");

  layersEnter.append("div")
    .attr("class", "symbolic-composition-layer-nodes");

  layers.exit().remove();

  layers.select(".symbolic-composition-layer-title")
    .text(d => d.title);

  layers.each(function (layerData) {
    const nodeContainer = d3.select(this).select(".symbolic-composition-layer-nodes");
    const nodeSel = nodeContainer.selectAll(".symbolic-composition-node")
      .data(layerData.nodes, (d: any) => d.nodeId);

    const nodeEnter = nodeSel.enter()
      .append("div")
      .attr("class", "symbolic-composition-node");

    nodeEnter.append("div")
      .attr("class", "symbolic-composition-node-header");

    nodeEnter.append("code")
      .attr("class", "symbolic-composition-node-expression");

    nodeEnter.append("ul")
      .attr("class", "symbolic-composition-contributions");

    nodeSel.exit().remove();

    nodeSel.classed("is-output", d => d.isOutput);

    nodeSel.select(".symbolic-composition-node-header")
      .text(d => d.label);

    nodeSel.select(".symbolic-composition-node-expression")
      .text(d => d.expression || "0");

    nodeSel.each(function (nodeData) {
      const list = d3.select(this).select(".symbolic-composition-contributions");
      list.selectAll("*").remove();
      if (!nodeData.contributions.length) {
        list.append("li")
          .attr("class", "symbolic-composition-contribution is-empty")
          .text("All incoming splines are currently pruned.");
        return;
      }
      nodeData.contributions.forEach(contrib => {
        const item = list.append("li")
          .attr("class", "symbolic-composition-contribution")
          .classed("is-pruned", contrib.pruned);
        item.append("span")
          .attr("class", "symbolic-composition-chip")
          .style("background-color", contrib.color)
          .text(contrib.entry.label);
        const body = item.append("div")
          .attr("class", "symbolic-composition-contribution-body");
        body.append("div")
          .attr("class", "symbolic-composition-contribution-label")
          .text(contrib.renderedExpression || contrib.entry.expression);
        body.append("div")
          .attr("class", "symbolic-composition-contribution-meta")
          .text(() => {
            const parts: string[] = [];
            parts.push(`coef ${formatLibraryNumber(contrib.coefficient)}`);
            parts.push(`rmse ${formatLibraryNumber(contrib.rmse)}`);
            parts.push(`src ${contrib.sourceLabel}`);
            if (contrib.pruned) {
              parts.push("pruned");
            }
            return parts.join(" | ");
          });
      });
    });
  });
}

function initSymbolicCompositionToggle(): void {
  const button = document.getElementById("symbolic-composition-toggle");
  if (!button) {
    return;
  }
  button.addEventListener("click", () => {
    const expanded = state.symbolicCompositionExpanded !== false;
    state.symbolicCompositionExpanded = !expanded;
    state.serialize();
    applySymbolicCompositionExpandedState();
  });
  applySymbolicCompositionExpandedState();
}

function applySymbolicCompositionExpandedState(): void {
  const expanded = state.symbolicCompositionExpanded !== false;
  const panel = document.getElementById("symbolic-composition-panel");
  if (panel) {
    panel.classList.toggle("is-collapsed", !expanded);
  }
  const button = document.getElementById("symbolic-composition-toggle");
  if (button) {
    button.setAttribute("aria-expanded", expanded ? "true" : "false");
    const label = button.querySelector(".symbolic-composition-toggle-label");
    if (label) {
      label.textContent = expanded ? "Hide" : "Show";
    }
  }
}

function computeSymbolicCompositionSummary(): SymbolicCompositionSummary | null {
  if (state.problem !== Problem.SYMBOLIC || !symbolicResult || !symbolicResult.isValid || !network || !network.length) {
    return null;
  }
  const selection = Array.isArray(state.symbolicLibrarySelection) ? state.symbolicLibrarySelection : [];
  if (!selection.length) {
    return null;
  }
  const entries = selection
    .map(id => getSymbolicLibraryEntry(id))
    .filter((entry): entry is SymbolicLibraryEntry => !!entry);
  if (!entries.length) {
    return null;
  }

  const trace = sampleSymbolicNetworkTrace();
  if (!trace) {
    return null;
  }

  const nodeById: { [id: string]: kan.KANNode } = {};
  const baseExpressions: { [id: string]: string } = {};
  const nodeLabels: { [id: string]: string } = {};

  network.forEach((layer, layerIdx) => {
    layer.forEach((node, idx) => {
      nodeById[node.id] = node;
      if (layerIdx === 0) {
        const feature = INPUTS[node.id];
        const label = state.problem === Problem.SYMBOLIC ? "x" : (feature && feature.label ? feature.label : node.id);
        baseExpressions[node.id] = label;
        nodeLabels[node.id] = label;
      } else if (layerIdx === network.length - 1) {
        nodeLabels[node.id] = "Output";
      } else {
        nodeLabels[node.id] = `Layer ${layerIdx} · Node ${idx + 1}`;
      }
    });
  });

  const nodeRms = computeRmsMap(trace.nodeOutputs);
  const matchMap = computeEdgeMatches(entries, trace, nodeRms, nodeLabels);
  const expressionCache: { [id: string]: string } = {};
  Object.keys(baseExpressions).forEach(key => {
    expressionCache[key] = baseExpressions[key];
  });
  const resolving: { [id: string]: boolean } = {};

  const resolveExpression = (nodeId: string): string => {
    if (expressionCache[nodeId] != null) {
      return expressionCache[nodeId];
    }
    if (resolving[nodeId]) {
      return nodeId;
    }
    resolving[nodeId] = true;
    const node = nodeById[nodeId];
    if (!node) {
      expressionCache[nodeId] = "0";
      resolving[nodeId] = false;
      return "0";
    }
    const usableMatches = node.inputEdges
      .map(edge => matchMap[edge.id])
      .filter((match): match is SymbolicEdgeMatch => !!match && !match.pruned);
    if (!usableMatches.length) {
      expressionCache[nodeId] = "0";
      resolving[nodeId] = false;
      return "0";
    }
    const terms = usableMatches.map(match => {
      const sourceExpr = resolveExpression(match.sourceId);
      const descriptor = describeEdgeTerm(match, sourceExpr);
      match.renderedExpression = descriptor.text;
      match.renderedFull = descriptor.full;
      match.termSign = descriptor.sign;
      match.sourceExpression = sourceExpr;
      return { sign: descriptor.sign, text: descriptor.text };
    });
    const combined = combineTerms(terms);
    expressionCache[nodeId] = combined;
    resolving[nodeId] = false;
    return combined;
  };

  const outputNode = network[network.length - 1][0];
  const layers: SymbolicCompositionLayer[] = [];

  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    const layerNodes = network[layerIdx];
    const title = layerIdx === network.length - 1 ? "Output layer" : `Layer ${layerIdx}`;
    const nodes: SymbolicNodeComposition[] = layerNodes.map((node, idx) => {
      const contributions = node.inputEdges
        .map(edge => matchMap[edge.id])
        .filter((match): match is SymbolicEdgeMatch => !!match)
        .sort((a, b) => {
          if (a.pruned !== b.pruned) {
            return a.pruned ? 1 : -1;
          }
          return (b.contributionRms || 0) - (a.contributionRms || 0);
        });
      const expression = resolveExpression(node.id);
      return {
        nodeId: node.id,
        label: nodeLabels[node.id] || `Node ${idx + 1}`,
        layerIndex: layerIdx,
        expression,
        contributions,
        isOutput: node === outputNode
      };
    });
    layers.push({ title, layerIndex: layerIdx, nodes });
  }

  Object.keys(matchMap).forEach(edgeId => {
    ensureEdgeRenderText(matchMap[edgeId], expressionCache, baseExpressions, nodeLabels);
  });

  const finalExpression = resolveExpression(outputNode.id) || "0";

  return {
    expression: finalExpression,
    layers
  };
}

function sampleSymbolicNetworkTrace(): SymbolicNetworkTrace | null {
  if (!network || !network.length) {
    return null;
  }
  const sampleXs: number[] = [];
  for (let i = 0; i < SYMBOLIC_PREVIEW_SAMPLES; i++) {
    sampleXs.push(-1 + (2 * i) / Math.max(1, SYMBOLIC_PREVIEW_SAMPLES - 1));
  }
  const nodeOutputs: { [id: string]: number[] } = {};
  const edgeInputs: { [id: string]: number[] } = {};
  const edgeOutputs: { [id: string]: number[] } = {};
  network.forEach(layer => {
    layer.forEach(node => {
      nodeOutputs[node.id] = [];
    });
  });

  sampleXs.forEach(x => {
    const inputVec = constructInput(x, 0);
    const inputLayer = network[0];
    inputLayer.forEach((node, idx) => {
      const value = inputVec[idx] != null ? inputVec[idx] : 0;
      node.output = value;
      nodeOutputs[node.id].push(value);
    });
    for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
      const currentLayer = network[layerIdx];
      currentLayer.forEach(node => {
        let sum = 0;
        node.inputEdges.forEach(edge => {
          const inputVal = edge.sourceNode.output;
          const edgeOutput = edge.learnableFunction.evaluate(inputVal);
          if (!edgeInputs[edge.id]) {
            edgeInputs[edge.id] = [];
          }
          if (!edgeOutputs[edge.id]) {
            edgeOutputs[edge.id] = [];
          }
          edgeInputs[edge.id].push(inputVal);
          edgeOutputs[edge.id].push(edgeOutput);
          sum += edgeOutput;
        });
        node.output = sum;
        if (!nodeOutputs[node.id]) {
          nodeOutputs[node.id] = [];
        }
        nodeOutputs[node.id].push(sum);
      });
    }
  });

  return { sampleXs, nodeOutputs, edgeInputs, edgeOutputs };
}

function computeEdgeMatches(
  entries: SymbolicLibraryEntry[],
  trace: SymbolicNetworkTrace,
  nodeRms: { [id: string]: number },
  nodeLabels: { [id: string]: string }
): { [edgeId: string]: SymbolicEdgeMatch } {
  const matches: { [edgeId: string]: SymbolicEdgeMatch } = {};
  if (!network) {
    return matches;
  }
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    const layer = network[layerIdx];
    layer.forEach(node => {
      node.inputEdges.forEach(edge => {
        const inputs = trace.edgeInputs[edge.id];
        const outputs = trace.edgeOutputs[edge.id];
        if (!inputs || !outputs || !inputs.length || inputs.length !== outputs.length) {
          return;
        }
        const best = findBestLibraryEntryForEdge(entries, inputs, outputs);
        if (!best) {
          return;
        }
        const contributionRms = computeRms(outputs);
        const destRms = nodeRms[edge.destNode.id] || 0;
        const pruneThreshold = Math.max(SYMBOLIC_COMPOSITION_MIN_RMS, destRms * SYMBOLIC_COMPOSITION_NODE_RATIO);
        const pruned = !edge.isActive || contributionRms < pruneThreshold;
        matches[edge.id] = {
          edgeId: edge.id,
          sourceId: edge.sourceNode.id,
          destId: edge.destNode.id,
          entry: best.entry,
          coefficient: best.coefficient,
          rmse: best.rmse,
          contributionRms,
          color: getSymbolicLibraryColor(best.entry.id),
          pruned,
          sourceLabel: nodeLabels[edge.sourceNode.id] || edge.sourceNode.id
        };
      });
    });
  }
  return matches;
}

function findBestLibraryEntryForEdge(
  entries: SymbolicLibraryEntry[],
  inputs: number[],
  outputs: number[]
): { entry: SymbolicLibraryEntry; coefficient: number; rmse: number } | null {
  if (!entries.length || !inputs.length || inputs.length !== outputs.length) {
    return null;
  }
  let best: { entry: SymbolicLibraryEntry; coefficient: number; rmse: number } | null = null;
  entries.forEach(entry => {
    const basisValues: number[] = new Array(inputs.length);
    let numerator = 0;
    let denominator = 0;
    for (let i = 0; i < inputs.length; i++) {
      const basis = safeLibraryValue(entry.evaluator(inputs[i]));
      basisValues[i] = basis;
      numerator += basis * outputs[i];
      denominator += basis * basis;
    }
    if (!isFinite(denominator) || Math.abs(denominator) < 1e-9) {
      return;
    }
    const coefficient = numerator / denominator;
    let mse = 0;
    for (let i = 0; i < inputs.length; i++) {
      const diff = coefficient * basisValues[i] - outputs[i];
      mse += diff * diff;
    }
    const rmse = Math.sqrt(mse / inputs.length);
    if (!best || rmse < best.rmse) {
      best = { entry, coefficient, rmse };
    }
  });
  return best;
}

function computeRms(values: number[]): number {
  if (!values || !values.length) {
    return 0;
  }
  let sumSq = 0;
  values.forEach(v => {
    sumSq += v * v;
  });
  return Math.sqrt(sumSq / values.length);
}

function computeRmsMap(map: { [id: string]: number[] }): { [id: string]: number } {
  const result: { [id: string]: number } = {};
  Object.keys(map).forEach(key => {
    result[key] = computeRms(map[key]);
  });
  return result;
}

function substituteEntryExpression(expression: string, inner: string): string {
  if (!expression || expression.indexOf("x") === -1) {
    return expression;
  }
  const replacement = `(${inner})`;
  let output = "";
  for (let i = 0; i < expression.length; i++) {
    const char = expression[i];
    if (char !== "x") {
      output += char;
      continue;
    }
    const prev = i > 0 ? expression[i - 1] : "";
    const next = i < expression.length - 1 ? expression[i + 1] : "";
    const prevIsWord = /[A-Za-z0-9_]/.test(prev);
    const nextIsWord = /[A-Za-z0-9_]/.test(next);
    if (prevIsWord || nextIsWord) {
      output += char;
    } else {
      output += replacement;
    }
  }
  return output;
}

function describeEdgeTerm(match: SymbolicEdgeMatch, sourceExpr: string): { sign: "+" | "-"; text: string; full: string } {
  const sign: "+" | "-" = match.coefficient >= 0 ? "+" : "-";
  const absCoeff = Math.abs(match.coefficient);
  const needsCoeff = Math.abs(absCoeff - 1) > SYMBOLIC_COMPOSITION_COEFF_EPS;
  const substituted = substituteEntryExpression(match.entry.expression, sourceExpr);
  const text = needsCoeff ? `${formatLibraryNumber(absCoeff)}*${substituted}` : substituted;
  const full = (sign === "-" ? "-" : "") + text;
  return { sign, text, full };
}

function combineTerms(terms: Array<{ sign: "+" | "-"; text: string }>): string {
  if (!terms.length) {
    return "0";
  }
  return terms.map((term, idx) => {
    if (idx === 0) {
      return (term.sign === "-" ? "-" : "") + term.text;
    }
    return ` ${term.sign} ${term.text}`;
  }).join("");
}

function ensureEdgeRenderText(
  match: SymbolicEdgeMatch,
  expressionCache: { [id: string]: string },
  baseExpressions: { [id: string]: string },
  nodeLabels: { [id: string]: string }
): void {
  if (!match || match.renderedExpression) {
    return;
  }
  const sourceExpr = expressionCache[match.sourceId] ||
    baseExpressions[match.sourceId] ||
    nodeLabels[match.sourceId] ||
    match.sourceLabel ||
    match.sourceId;
  const descriptor = describeEdgeTerm(match, sourceExpr);
  match.renderedExpression = descriptor.text;
  match.renderedFull = descriptor.full;
  match.termSign = descriptor.sign;
  match.sourceExpression = sourceExpr;
}

function updateProblemSpecificUI(): void {
  let isSymbolic = state.problem === Problem.SYMBOLIC;
  d3.selectAll(".ui-dataset").style("display", isSymbolic ? "none" : null);
  d3.selectAll(".dataset-list").style("display", isSymbolic ? "none" : null);
  d3.selectAll(".ui-symbolicExpression").style("display", isSymbolic ? "block" : "none");
  d3.select("#symbolic-plot").style("display", isSymbolic ? "block" : "none");
  d3.select("#symbolic-library-panel").style("display", isSymbolic ? "block" : "none");
  d3.select("#symbolic-composition-panel").style("display", isSymbolic ? "block" : "none");
  if (isSymbolic) {
    applySymbolicCompositionExpandedState();
  }
  if (!isSymbolic && symbolicPlot) {
    symbolicPlot.clear();
    symbolicPlot.setVisible(false);
  }
  if (isSymbolic) {
    let exprInput = document.getElementById("symbolicExpression") as HTMLInputElement;
    if (exprInput) {
      exprInput.value = state.symbolicExpression || "";
    }
    for (let inputName in INPUTS) {
      if (!SYMBOLIC_ALLOWED_INPUTS[inputName]) {
        state[inputName] = false;
      }
    }
    state.x = true;
  }
  refreshSymbolicLibraryUI();
  updateHoverCardSymbolicOverlay();
  updateSymbolicComposition();
}

let firstInteraction = true;
let parametersChanged = false;

function userHasInteracted() {
  if (!firstInteraction) {
    return;
  }
  firstInteraction = false;
  let page = 'index';
  if (state.tutorial != null && state.tutorial !== '') {
    page = `/v/tutorials/${state.tutorial}`;
  }
  ga('set', 'page', page);
  ga('send', 'pageview', { 'sessionControl': 'start' });
}

function simulationStarted() {
  ga('send', {
    hitType: 'event',
    eventCategory: 'Starting Simulation',
    eventAction: parametersChanged ? 'changed' : 'unchanged',
    eventLabel: state.tutorial == null ? '' : state.tutorial
  });
  parametersChanged = false;
}

function isCursorOverElement(element: HTMLElement, cursorX: number, cursorY: number): boolean {
  if (!element) return false;
  const rect = element.getBoundingClientRect();
  return cursorX >= rect.left && cursorX <= rect.right &&
    cursorY >= rect.top && cursorY <= rect.bottom;
}

function updateHoverCard(type: HoverType, nodeOrEdge?: kan.KANNode | kan.KANEdge, coordinates?: number[]) {
  let hovercard = d3.select("#hovercard");

  // Clear any pending hide timeout
  if (hoverCardHideTimeout !== null) {
    clearTimeout(hoverCardHideTimeout);
    hoverCardHideTimeout = null;
  }

  if (type == null) {
    // Hide the hover card immediately
    hovercard.style("display", "none");
    // Clean up hover card spline chart
    if (hoverCardSplineChart) {
      hoverCardSplineChart.clear();
      hoverCardSplineChart = null;
    }
    hoverCardLibraryInfo = null;
    currentHoverCardEdge = null;
    return;
  }

  // Position the hover card below the spline chart
  let finalX, finalY;

  // For weight hover cards, position below the spline chart
  // Center the hover card horizontally relative to the spline chart
  finalX = coordinates[0] - 150; // Center 300px hover card around spline chart
  finalY = coordinates[1] + (SPLINE_CHART_SIZE_Y / 2) + 10; // Position below spline chart with 10px gap

  // Ensure hover card stays within viewport bounds
  // Prevent going off left edge
  if (finalX < 10) {
    finalX = 10;
  }

  // Prevent going off right edge (300px hover card width + padding)
  if (finalX + 310 > window.innerWidth) {
    finalX = window.innerWidth - 310;
  }

  // Prevent going off bottom edge (200px hover card height + padding)
  if (finalY + 210 > window.innerHeight) {
    // If no room below, position above the spline chart
    finalY = coordinates[1] - (SPLINE_CHART_SIZE_Y / 2) - 210;
  }

  hovercard
    .style("left", finalX + "px")
    .style("top", finalY + "px")
    .style("display", "block");

  // Clear existing content
  hovercard.selectAll("*").remove();
  hoverCardLibraryInfo = null;

  if (type === HoverType.WEIGHT) {
    let edge = nodeOrEdge as kan.KANEdge;

    // Create extended SplineChart for learnable function
    let splineContainer = hovercard.append("div")
      .style("padding", "5px");



    // Get the last node id in the network (last layer, last node)
    const lastLayer = network[network.length - 1];
    const lastNode = lastLayer && lastLayer.length ? lastLayer[lastLayer.length - 1] : null;
    const lastId = lastNode ? lastNode.id : "N/A";
    const isDestOutput = edge.destNode.id === lastId;
    const isSourceInput = INPUTS.hasOwnProperty(edge.sourceNode.id);

    let leftText = isSourceInput ? 'Input ' + edge.sourceNode.id : 'Node ' + edge.sourceNode.id;
    let rightText = isDestOutput ? 'Output' : 'Node ' + edge.destNode.id;

    let edgeText = `${leftText} → ${rightText}`;

    hoverCardSplineChart = new SplineChart(splineContainer, {
      width: 300,
      height: 200,
      title: edgeText,
      showControlPoints: true,
      showOldControlPaths: false,
      showKnots: false,
      showGrid: true,
      showXAxisLabels: true,
      showYAxisLabels: true,
      showXAxisValues: true,
      showYAxisValues: true,
      showBorder: false,
      showActivationHistogram: true,
      histogramOpacity: 0.3,
      histogramColor: "#4A90E2",
      outputHistogramColor: "#4A90E2",
      interactive: true
    });

    // Set callback to update the main network visualization when control points change
    hoverCardSplineChart.setOnControlPointChange((index: number, newValue: number) => {
      // Clear the histogram since the spline function has changed
      edge.resetHistogram();

      // Run forward passes with training data to repopulate histograms
      trainData.forEach((point) => {
        let input = constructInput(point.x, point.y);
        kan.kanForwardProp(network, input, true);
      });

      // Only update the edge visualization immediately (lightweight)
      let edgeId = `${edge.sourceNode.id}-${edge.destNode.id}`;
      if (edgeSplineCharts[edgeId]) {
        edgeSplineCharts[edgeId].updateFunction(edge.learnableFunction);
      }

      // Update the hovercard's histogram display with fresh activation data
      const inputHistogramData = edge.getNormalizedHistogram();
      const outputHistogramData = edge.getNormalizedOutputHistogram();
      hoverCardSplineChart.updateFunction(edge.learnableFunction, inputHistogramData, outputHistogramData);

      // Update link colors/widths for this specific edge using absolute std values
      const inputStdBoundary = 0.6;  // Typical std for inputs in [-1, 1] range
      const outputStdBoundary = 0.6; // Typical std for outputs in [-1, 1] range

      const inputStd = edge.getInputActivationStd();
      const outputStd = edge.getOutputActivationStd();
      const normalizedInputStd = Math.min(1, inputStd / inputStdBoundary);
      const normalizedOutputStd = Math.min(1, outputStd / outputStdBoundary);

      let svgContainer = d3.select("g.core");
      svgContainer.select(`#link${edgeId}-part1`)
        .style({
          "stroke-width": linkWidthScale(normalizedInputStd),
          "stroke": linkColorScale(normalizedInputStd)
        });
      svgContainer.select(`#link${edgeId}-part2`)
        .style({
          "stroke-width": linkWidthScale(normalizedOutputStd),
          "stroke": linkColorScale(normalizedOutputStd)
        });
    });

    // Set callbacks for drag start and end to prevent hovercard from hiding during drag
    hoverCardSplineChart.setOnDragStart(() => {
      isDraggingControlPoint = true;
    });

    hoverCardSplineChart.setOnDragEnd(() => {
      isDraggingControlPoint = false;

      // NOW update expensive visualizations after drag is complete
      updateWeightsUI(network, d3.select("g.core"));
      updateDecisionBoundary(network, false);

      // Update the main heatmap
      let selectedId = selectedNodeId != null ?
        selectedNodeId : kan.getKANOutputNode(network).id;
      heatMap.updateBackground(boundary[selectedId], state.discretize);

      // Update all node-specific heatmaps
      d3.select("#network").selectAll("div.canvas")
        .each(function (data: { heatmap: HeatMap, id: string }) {
          data.heatmap.updateBackground(reduceMatrix(boundary[data.id], 10),
            state.discretize);
        });

      // Don't hide the hovercard after drag - let the normal mouse leave handlers manage it
    });

    // Update with the learnable function and histogram data
    const inputHistogramData = edge.getNormalizedHistogram();
    const outputHistogramData = edge.getNormalizedOutputHistogram();
    hoverCardSplineChart.updateFunction(edge.learnableFunction, inputHistogramData, outputHistogramData);
    updateHoverCardSymbolicOverlay();

    if (state.problem === Problem.SYMBOLIC) {
      hoverCardLibraryInfo = hovercard.append("div")
        .attr("class", "symbolic-hover-library-summary");
    } else {
      hoverCardLibraryInfo = null;
    }

    updateHoverCardSymbolicOverlay();

    currentHoverCardEdge = edge;
  }
}

function addPlusMinusControl(x: number, layerIdx: number) {
  let div = d3.select("#network").append("div")
    .classed("plus-minus-neurons", true)
    .style("position", "absolute")
    .style("left", `${x - 15}px`)
    .style("top", "-60px");

  let i = layerIdx - 1;
  let firstRow = div.append("div").attr("class", `ui-numHiddenLayers`);
  firstRow.append("button")
    .attr("class", "mdl-button mdl-js-button mdl-button--icon")
    .on("click", () => {
      let numNeurons = state.networkShape[i];
      if (numNeurons >= 8) {
        return;
      }
      state.networkShape[i]++;
      parametersChanged = true;
      reset();
    })
    .append("i")
    .attr("class", "material-icons")
    .text("add");

  firstRow.append("button")
    .attr("class", "mdl-button mdl-js-button mdl-button--icon")
    .on("click", () => {
      let numNeurons = state.networkShape[i];
      if (numNeurons <= 1) {
        return;
      }
      state.networkShape[i]--;
      parametersChanged = true;
      reset();
    })
    .append("i")
    .attr("class", "material-icons")
    .text("remove");

  let suffix = state.networkShape[i] > 1 ? "s" : "";
  firstRow.append("span").text(
    state.networkShape[i] + " Node" + suffix
  );
}

function getRelativeHeight(selection: d3.Selection<any>) {
  let node = selection.node() as HTMLElement;
  if (!node) {
    return 0;
  }
  return node.offsetHeight + node.offsetTop;
}

updateProblemSpecificUI();
drawDatasetThumbnails();
initTutorial();
makeGUI();

// Set up hovercard event handlers to keep it visible when mouse is over it
// and hide it when mouse leaves it
// Use native DOM addEventListener with capture:true to catch events from child elements
const hovercardElement = document.getElementById("hovercard");

hovercardElement.addEventListener("mouseenter", function () {
  // Cancel any pending hide timeout when mouse enters hovercard
  if (hoverCardHideTimeout !== null) {
    clearTimeout(hoverCardHideTimeout);
    hoverCardHideTimeout = null;
  }
});

hovercardElement.addEventListener("mousemove", function (event: MouseEvent) {
  // Track cursor position
  lastMousePosition = { x: event.pageX, y: event.pageY };
}, true); // Use capture phase to catch events from child elements

hovercardElement.addEventListener("mouseleave", function () {
  // Don't hide the hovercard if user is dragging a control point
  if (isDraggingControlPoint) {
    return;
  }
  // Hide the hovercard when mouse leaves it
  updateHoverCard(null);
});

generateData(true);
reset(true);
hideControls();
