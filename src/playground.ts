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
import {HeatMap, reduceMatrix} from "./heatmap";
import {SplineChart} from "./splinechart";
import {
  State,
  datasets,
  regDatasets,
  problems,
  getKeyFromValue,
  Problem
} from "./state";
import {Example2D, shuffle} from "./dataset";
import {AppendingLineChart} from "./linechart";
import {NetworkLayoutManager} from "./layout";
import * as d3 from 'd3';

let mainWidth;

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

const RECT_SIZE = 30;
const NUM_SAMPLES_CLASSIFY = 500;
const NUM_SAMPLES_REGRESS = 1200;
const DENSITY = 100;
const SPLINE_CHART_SIZE_X = 30;
const SPLINE_CHART_SIZE_Y = 30;
const NODE_SPACING = 25;

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

let INPUTS: {[name: string]: InputFeature} = {
  "x": {f: (x, y) => x, label: "X_1"},
  "y": {f: (x, y) => y, label: "X_2"},
  "xSquared": {f: (x, y) => x * x, label: "X_1^2"},
  "ySquared": {f: (x, y) => y * y,  label: "X_2^2"},
  "xTimesY": {f: (x, y) => x * y, label: "X_1X_2"},
  "sinX": {f: (x, y) => Math.sin(x), label: "sin(X_1)"},
  "sinY": {f: (x, y) => Math.sin(y), label: "sin(X_2)"},
};

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
];

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

let boundary: {[id: string]: number[][]} = {};
let selectedNodeId: string = null;
// Plot the heatmap.
let xDomain: [number, number] = [-1, 1];
let heatMap =
    new HeatMap(300, DENSITY, xDomain, xDomain, d3.select("#heatmap"),
        {showAxes: true});
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
let edgeSplineCharts: {[edgeId: string]: SplineChart} = {};
// Hover card spline chart
let hoverCardSplineChart: SplineChart = null;
// Current edge being displayed in hover card
let currentHoverCardEdge: kan.KANEdge = null;
// Layout manager for coordinating positions
let layoutManager: NetworkLayoutManager = null;
// Timeout for hiding hover card with delay
let hoverCardHideTimeout: number = null;
// Track if user is currently dragging a control point
let isDraggingControlPoint: boolean = false;
// Track cursor position during interactions
let lastMousePosition: {x: number, y: number} = {x: 0, y: 0};

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
    generateData();
    parametersChanged = true;
  });

  let dataThumbnails = d3.selectAll("canvas[data-dataset]");
  dataThumbnails.on("click", function() {
    let newDataset = datasets[this.dataset.dataset];
    if (newDataset === state.dataset) {
      return; // No-op.
    }
    state.dataset =  newDataset;
    dataThumbnails.classed("selected", false);
    d3.select(this).classed("selected", true);
    generateData();
    parametersChanged = true;
    reset();
  });

  let datasetKey = getKeyFromValue(datasets, state.dataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-dataset=${datasetKey}]`)
    .classed("selected", true);

  let regDataThumbnails = d3.selectAll("canvas[data-regDataset]");
  regDataThumbnails.on("click", function() {
    let newDataset = regDatasets[this.dataset.regdataset];
    if (newDataset === state.regDataset) {
      return; // No-op.
    }
    state.regDataset =  newDataset;
    regDataThumbnails.classed("selected", false);
    d3.select(this).classed("selected", true);
    generateData();
    parametersChanged = true;
    reset();
  });

  let regDatasetKey = getKeyFromValue(regDatasets, state.regDataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-regDataset=${regDatasetKey}]`)
    .classed("selected", true);

  d3.select("#add-layers").on("click", () => {
    if (state.numHiddenLayers >= 3) {
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

  let showTestData = d3.select("#show-test-data").on("change", function() {
    state.showTestData = this.checked;
    state.serialize();
    userHasInteracted();
    heatMap.updateTestPoints(state.showTestData ? testData : []);
  });
  // Check/uncheck the checkbox according to the current state.
  showTestData.property("checked", state.showTestData);

  let discretize = d3.select("#discretize").on("change", function() {
    state.discretize = this.checked;
    state.serialize();
    userHasInteracted();
    updateUI();
  });
  // Check/uncheck the checbox according to the current state.
  discretize.property("checked", state.discretize);

  let percTrain = d3.select("#percTrainData").on("input", function() {
    state.percTrainData = this.value;
    d3.select("label[for='percTrainData'] .value").text(this.value);
    generateData();
    parametersChanged = true;
    reset();
  });
  percTrain.property("value", state.percTrainData);
  d3.select("label[for='percTrainData'] .value").text(state.percTrainData);

  let noise = d3.select("#noise").on("input", function() {
    state.noise = this.value;
    d3.select("label[for='noise'] .value").text(this.value);
    generateData();
    parametersChanged = true;
    reset();
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

  let batchSize = d3.select("#batchSize").on("input", function() {
    state.batchSize = this.value;
    d3.select("label[for='batchSize'] .value").text(this.value);
    parametersChanged = true;
    reset();
  });
  batchSize.property("value", state.batchSize);
  d3.select("label[for='batchSize'] .value").text(state.batchSize);

  let problemDropdown = d3.select("#problem").on("change", function() {
    state.problem = problems[this.value];
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
    generateData();
    drawDatasetThumbnails();
    reset();
  });
  problemDropdown.property("value",
      getKeyFromValue(problems, state.problem));

  let learningRate = d3.select("#learningRate").on("change", function() {
    state.learningRate = +this.value;
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
    // reset(); not needed
  });
  learningRate.property("value", state.learningRate);

  // numControlPoints UI
  let numControlPointsSel = d3.select("#numControlPoints").on("change", function() {
    state.numControlPoints = +this.value;
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
    reset();
  });
  numControlPointsSel.property("value", state.numControlPoints);

  let degree = d3.select("#degree").on("change", function() {
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
  d3.select("#degree").on("change.gridOptions", function() {
    const degVal = +(this as HTMLSelectElement).value;
    const currentCP = +d3.select("#numControlPoints").property("value") || state.numControlPoints;
    updateNumControlPointsOptionsForDegree(degVal, currentCP);
  });

  // Add initNoise control
  const initNoiseSel = d3.select("#initNoise");

  let initNoise = initNoiseSel.on("change", function() {
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
}

function updateWeightsUI(network: kan.KANNode[][], container) {
  // Use fixed boundaries for activation std (reasonable values based on typical activation ranges)
  const inputStdBoundary = 0.6;  // 0.6 is Typical std for inputs in [-1, 1] range
  const outputStdBoundary = 0.6; // 0.6 is Typical std for outputs in [-1, 1] range
  
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    
    // Update all the visual elements with absolute activation stds
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
        
        // Update the first link (source to spline chart) - use input activation std
        container.select(`#link${edgeId}-part1`)
            .style({
              "stroke-dashoffset": -iter / 3,
              "stroke-width": linkWidthScale(normalizedInputStd),
              "stroke": linkColorScale(normalizedInputStd)
            });
            
        // Update the second link (spline chart to destination) - use output activation std
        container.select(`#link${edgeId}-part2`)
            .style({
              "stroke-dashoffset": -iter / 3,
              "stroke-width": linkWidthScale(normalizedOutputStd),
              "stroke": linkColorScale(normalizedOutputStd)
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
  let activeOrNotClass = state[nodeId] ? "active" : "inactive";
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
    .on("mouseenter", function() {
      selectedNodeId = nodeId;
      div.classed("hovered", true);
      nodeGroup.classed("hovered", true);
      updateDecisionBoundary(network, false);
      heatMap.updateBackground(boundary[nodeId], state.discretize);
    })
    .on("mouseleave", function() {
      selectedNodeId = null;
      div.classed("hovered", false);
      nodeGroup.classed("hovered", false);
      updateDecisionBoundary(network, false);
      heatMap.updateBackground(boundary[kan.getKANOutputNode(network).id],
          state.discretize);
    });
  if (isInput) {
    div.on("click", function() {
      state[nodeId] = !state[nodeId];
      parametersChanged = true;
      reset();
    });
    div.style("cursor", "pointer");
  }
  if (isInput) {
    div.classed(activeOrNotClass, true);
  }
  let nodeHeatMap = new HeatMap(RECT_SIZE, DENSITY / 10, xDomain,
      xDomain, div, {noSvg: true});
  div.datum({heatmap: nodeHeatMap, id: nodeId});

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

  // Get node IDs for input layer
  let nodeIds = Object.keys(INPUTS);
  
  // Initialize layout manager with initial dimensions
  layoutManager = new NetworkLayoutManager(
    RECT_SIZE,
    NODE_SPACING,
    padding,
    width,
    600, // Initial height, will be updated
    SPLINE_CHART_SIZE_X,
    SPLINE_CHART_SIZE_Y
  );
  
  // Calculate required height based on network structure
  let maxY = layoutManager.calculateRequiredHeight(network, nodeIds.length, nodeIds);
  svg.attr("height", maxY);
  
  // Update layout manager with final height
  layoutManager.updateDimensions(width, maxY);

  // Get complete layout from layout manager
  const layout = layoutManager.getCompleteNetworkLayout(network, nodeIds);
  const { layers, node2coord, edgesByLayer } = layout;

  let container = svg.append("g")
    .classed("core", true)
    .attr("transform", `translate(${padding},${padding})`);

  let numLayers = network.length;
  let calloutThumb = d3.select(".callout.thumbnail").style("display", "none");
  let calloutWeights = d3.select(".callout.weights").style("display", "none");
  let idWithCallout = null;
  let targetIdWithCallout = null;

  // Draw all layers using layout manager data
  layers.forEach((layer, layerIdx) => {
    const isInputLayer = layerIdx === 0;
    const isOutputLayer = layerIdx === numLayers - 1;
    const isHiddenLayer = !isInputLayer && !isOutputLayer;

    // Add plus/minus controls for hidden layers
    if (isHiddenLayer) {
      const layerX = layoutManager.calculateLayerX(layerIdx, numLayers);
      addPlusMinusControl(layerX - RECT_SIZE / 2, layerIdx);
    }

    // Draw nodes for this layer
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

      // Show callout to thumbnails for hidden layers
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

  // Draw all edges with spline charts using layout manager data
  for (let layerIdx = 1; layerIdx < numLayers; layerIdx++) {
    const layerEdges = edgesByLayer[layerIdx];
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

      // Show callout to weights for appropriate edges
      if (targetIdWithCallout == null && globalEdgeIdx === layerEdges.length - 1) {
        const prevLayer = network[layerIdx - 1];
        const lastNodePrevLayer = prevLayer[prevLayer.length - 1];
        if (edge.sourceNode.id === lastNodePrevLayer.id &&
            (edge.sourceNode.id !== idWithCallout || numLayers <= 5) &&
            edge.destNode.id !== idWithCallout &&
            prevLayer.length >= currentLayer.length) {
          calloutWeights.style({
            display: null,
            top: `${splinePosition.y + SPLINE_CHART_SIZE_Y/2 + 5}px`,
            left: `${splinePosition.x + 3}px`
          });
          targetIdWithCallout = edge.destNode.id;
        }
      }
    });
  }


  // Adjust the height of the features column.
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
    edge: kan.KANEdge, node2coord: {[id: string]: {cx: number, cy: number}},
    network: kan.KANNode[][], container,
    isFirst: boolean, index: number, length: number, 
    splinePosition: {x: number, y: number},
    linkPath?: {sourceToSpline: any, splineToTarget: any}) {
  
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
  
  // Create spline chart div
  let splineDiv = d3.select("#network").append("div")
    .attr("class", "spline-chart")
    .attr("id", `spline-${edgeId}`)
    .style({
      position: "absolute",
      left: `${splineX - SPLINE_CHART_SIZE_X / 2 + 1}px`,
      top: `${splineY - SPLINE_CHART_SIZE_Y / 2 + 3}px`,
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

  // Add hover functionality to spline chart div
  splineDiv.on("mouseenter", function() {
    updateHoverCard(HoverType.WEIGHT, edge, [splineX, splineY]);
  }).on("mouseleave", function() {
    // Don't hide if user is dragging a control point
    if (isDraggingControlPoint) {
      return;
    }
    // Delay hiding to allow mouse to move to hovercard
    hoverCardHideTimeout = setTimeout(() => {
      updateHoverCard(null);
    }, 100);
  }).on("mousemove", function() {
    // Track cursor position
    const event = d3.event as MouseEvent;
    lastMousePosition = {x: event.pageX, y: event.pageY};
  });

  // Draw first link: source node to spline chart
  let line1 = container.insert("path", ":first-child");
  let datum1;
  if (linkPath) {
    // Use pre-calculated path from layout manager
    datum1 = {
      source: { y: linkPath.sourceToSpline.source.y, x: linkPath.sourceToSpline.source.x },
      target: { y: linkPath.sourceToSpline.target.y, x: linkPath.sourceToSpline.target.x }
    };
  } else {
    // Fallback to manual calculation
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
    // Use pre-calculated path from layout manager
    datum2 = {
      source: { y: linkPath.splineToTarget.source.y, x: linkPath.splineToTarget.source.x },
      target: { y: linkPath.splineToTarget.target.y, x: linkPath.splineToTarget.target.x }
    };
  } else {
    // Fallback to manual calculation
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
      kan.kanForwardProp(network, input);
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
    let output = kan.kanForwardProp(network, input);
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
      .each(function(data: {heatmap: HeatMap, id: string}) {
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
}

function constructInputIds(): string[] {
  let result: string[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      result.push(inputName);
    }
  }
  return result;
}

function constructInput(x: number, y: number): number[] {
  let input: number[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
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

function reset(onStartup=false) {
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
  let numInputs = constructInput(0 , 0).length;
  let shape = [numInputs].concat(state.networkShape).concat([1]);

  // Derive gridSize from numControlPoints
  const derivedGridSize = Math.max(1, Math.floor(state.numControlPoints) - 1);

  network = kan.buildKANNetwork(shape, constructInputIds(), derivedGridSize, state.degree, state.initNoise);
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
    data.forEach(function(d) {
      context.fillStyle = colorScale(d.label);
      context.fillRect(w * (d.x + 1) / 2, h * (d.y + 1) / 2, 4, 4);
    });
    d3.select(canvas.parentNode).style("display", null);
  }
  d3.selectAll(".dataset").style("display", "none");

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
    input.on("change", function() {
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

function generateData(firstTime = false) {
  if (!firstTime) {
    // Change the seed.
    state.seed = Math.random().toFixed(5);
    state.serialize();
    userHasInteracted();
  }
  Math.seedrandom(state.seed);
  let numSamples = (state.problem === Problem.REGRESSION) ?
      NUM_SAMPLES_REGRESS : NUM_SAMPLES_CLASSIFY;
  let generator = state.problem === Problem.CLASSIFICATION ?
      state.dataset : state.regDataset;
  let data = generator(numSamples, state.noise / 100);
  // Shuffle the data in-place.
  shuffle(data);
  // Split into train and test data.
  let splitIndex = Math.floor(data.length * state.percTrainData / 100);
  trainData = data.slice(0, splitIndex);
  testData = data.slice(splitIndex);
  heatMap.updatePoints(trainData);
  heatMap.updateTestPoints(state.showTestData ? testData : []);
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
  ga('send', 'pageview', {'sessionControl': 'start'});
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
  
  hovercard.style({
    "left": finalX + "px",
    "top": finalY + "px",
    "display": "block"
  });
  
  // Clear existing content
  hovercard.selectAll("*").remove();
  
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
      height: 300,
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
      // Only update the edge visualization immediately (lightweight)
      let edgeId = `${edge.sourceNode.id}-${edge.destNode.id}`;
      if (edgeSplineCharts[edgeId]) {
        edgeSplineCharts[edgeId].updateFunction(edge.learnableFunction);
      }
      
      // Update the hovercard's histogram display to reflect current activations
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
          .each(function(data: {heatmap: HeatMap, id: string}) {
        data.heatmap.updateBackground(reduceMatrix(boundary[data.id], 10),
            state.discretize);
      });
      
      // Don't hide the hovercard after drag - let the normal mouse leave handlers manage it
    });
    
    // Update with the learnable function and histogram data
    const inputHistogramData = edge.getNormalizedHistogram();
    const outputHistogramData = edge.getNormalizedOutputHistogram();
    hoverCardSplineChart.updateFunction(edge.learnableFunction, inputHistogramData, outputHistogramData);
    currentHoverCardEdge = edge;
  }
}

function addPlusMinusControl(x: number, layerIdx: number) {
  let div = d3.select("#network").append("div")
    .classed("plus-minus-neurons", true)
    .style({
      position: "absolute",
      left: `${x - 15}px`, // Horizontally center the control
      top: "-60px"  // Move higher to avoid overlap with heatmaps
    });

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

drawDatasetThumbnails();
initTutorial();
makeGUI();

// Set up hovercard event handlers to keep it visible when mouse is over it
// and hide it when mouse leaves it
// Use native DOM addEventListener with capture:true to catch events from child elements
const hovercardElement = document.getElementById("hovercard");

hovercardElement.addEventListener("mouseenter", function() {
  // Cancel any pending hide timeout when mouse enters hovercard
  if (hoverCardHideTimeout !== null) {
    clearTimeout(hoverCardHideTimeout);
    hoverCardHideTimeout = null;
  }
});

hovercardElement.addEventListener("mousemove", function(event: MouseEvent) {
  // Track cursor position
  lastMousePosition = {x: event.pageX, y: event.pageY};
}, true); // Use capture phase to catch events from child elements

hovercardElement.addEventListener("mouseleave", function() {
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
