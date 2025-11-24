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
import * as d3 from "d3";

/**
 * Interface for node position information
 */
export interface NodePosition {
  cx: number;  // center x coordinate
  cy: number;  // center y coordinate
}

/**
 * Interface for spline chart position information
 */
export interface SplinePosition {
  x: number;  // center x coordinate
  y: number;  // center y coordinate
}

/**
 * Interface for link path information
 */
export interface LinkPath {
  sourceToSpline: {
    source: { x: number, y: number },
    target: { x: number, y: number }
  };
  splineToTarget: {
    source: { x: number, y: number },
    target: { x: number, y: number }
  };
}

/**
 * Interface for layer layout information
 */
export interface LayerLayout {
  cx: number;           // center x coordinate for the layer
  nodePositions: NodePosition[];
  layerIndex: number;
}

/**
 * Interface for edge information used in layout calculations
 */
interface EdgeInfo {
  edge: kan.KANEdge;
  sourceY: number;
  destY: number;
  destNodeIdx: number;
  sourceNodeIdx: number;
  edgeKey: string;
}

/**
 * NetworkLayoutManager handles the positioning of both heatmap nodes and spline charts
 * in a unified grid system, ensuring proper alignment of visual elements.
 * 
 * This class centralizes all layout calculations for:
 * - Input layer nodes (heatmaps)
 * - Hidden layer nodes (heatmaps)
 * - Output layer node (heatmap)
 * - Spline charts (positioned between layers)
 * - Link paths connecting nodes through spline charts
 */
export class NetworkLayoutManager {
  private rectSize: number;
  private spacing: number;
  private padding: number;
  private svgWidth: number;
  private svgHeight: number;
  private splineChartWidth: number;
  private splineChartHeight: number;
  private featureWidth: number = 118;  // Width of the feature column

  // Getter methods for external access
  getRectSize(): number { return this.rectSize; }
  getSpacing(): number { return this.spacing; }
  getPadding(): number { return this.padding; }

  constructor(
    rectSize: number,
    spacing: number,
    padding: number,
    svgWidth: number,
    svgHeight: number,
    splineChartWidth: number,
    splineChartHeight: number
  ) {
    this.rectSize = rectSize;
    this.spacing = spacing;
    this.padding = padding;
    this.svgWidth = svgWidth;
    this.svgHeight = svgHeight;
    this.splineChartWidth = splineChartWidth;
    this.splineChartHeight = splineChartHeight;
  }

  /**
   * Update SVG dimensions
   */
  updateDimensions(svgWidth: number, svgHeight: number): void {
    this.svgWidth = svgWidth;
    this.svgHeight = svgHeight;
  }

  /**
   * Calculate X coordinate for a layer in the network
   */
  calculateLayerX(layerIdx: number, numLayers: number): number {
    if (layerIdx === 0) {
      // Input layer
      return this.rectSize / 2 + 50;
    } else if (layerIdx === numLayers - 1) {
      // Output layer
      return this.svgWidth + this.rectSize / 2;
    } else {
      // Hidden layers - distribute evenly between input and output
      const layerScale = d3.scale.ordinal<number, number>()
        .domain(d3.range(1, numLayers - 1))
        .rangePoints([this.featureWidth, this.svgWidth - this.rectSize], 0.7);
      return layerScale(layerIdx) + this.rectSize / 2;
    }
  }

  /**
   * Get layout information for all layers in the network
   */
  getNetworkLayout(network: kan.KANNode[][], inputIds: string[]): {
    layers: LayerLayout[],
    node2coord: {[id: string]: NodePosition}
  } {
    const numLayers = network.length;
    const layers: LayerLayout[] = [];
    const node2coord: {[id: string]: NodePosition} = {};

    // Input layer (layer 0)
    const inputCx = this.calculateLayerX(0, numLayers);
    const inputPositions: NodePosition[] = [];
    inputIds.forEach((nodeId, i) => {
      const pos = this.getNodePosition(i, inputCx);
      node2coord[nodeId] = pos;
      inputPositions.push(pos);
    });
    layers.push({
      cx: inputCx,
      nodePositions: inputPositions,
      layerIndex: 0
    });

    // Hidden layers and output layer
    for (let layerIdx = 1; layerIdx < numLayers; layerIdx++) {
      const layerNodes = network[layerIdx];
      const cx = this.calculateLayerX(layerIdx, numLayers);
      const nodePositions: NodePosition[] = [];

      layerNodes.forEach((node, i) => {
        const pos = this.getNodePosition(i, cx);
        node2coord[node.id] = pos;
        nodePositions.push(pos);
      });

      layers.push({
        cx: cx,
        nodePositions: nodePositions,
        layerIndex: layerIdx
      });
    }

    return { layers, node2coord };
  }

  /**
   * Calculate the y-coordinate for a node at a given index in the grid
   */
  private nodeIndexScale(nodeIndex: number): number {
    return nodeIndex * (this.rectSize + this.spacing);
  }

  /**
   * Calculate center position for a node
   */
  getNodePosition(nodeIndex: number, cx: number): NodePosition {
    return {
      cx: cx,
      cy: this.nodeIndexScale(nodeIndex) + this.rectSize / 2
    };
  }

  /**
   * Sort edges by destination node index first, then by source node index
   * to ensure proper alignment with the grid system
   */
  private sortEdges(
    network: kan.KANNode[][],
    layerIdx: number,
    node2coord: {[id: string]: NodePosition},
    inputIds: string[]
  ): EdgeInfo[] {
    let allEdges: EdgeInfo[] = [];
    let currentLayer = network[layerIdx];

    for (let nodeIdx = 0; nodeIdx < currentLayer.length; nodeIdx++) {
      let node = currentLayer[nodeIdx];
      for (let edgeIdx = 0; edgeIdx < node.inputEdges.length; edgeIdx++) {
        let edge = node.inputEdges[edgeIdx];
        let edgeKey = `${edge.sourceNode.id}-${edge.destNode.id}`;
        let sourceCoord = node2coord[edge.sourceNode.id];
        let destCoord = node2coord[edge.destNode.id];

        // Find source node index for proper alignment
        let sourceNodeIdx = -1;
        if (layerIdx === 1) {
          // For first layer, find index in input IDs
          sourceNodeIdx = inputIds.indexOf(edge.sourceNode.id);
        } else {
          // For other layers, find index in previous layer
          let prevLayer = network[layerIdx - 1];
          for (let i = 0; i < prevLayer.length; i++) {
            if (prevLayer[i].id === edge.sourceNode.id) {
              sourceNodeIdx = i;
              break;
            }
          }
        }

        allEdges.push({
          edge: edge,
          sourceY: sourceCoord.cy,
          destY: destCoord.cy,
          destNodeIdx: nodeIdx,
          sourceNodeIdx: sourceNodeIdx,
          edgeKey: edgeKey
        });
      }
    }

    // Sort edges by destination node index first, then by source node index
    allEdges.sort((a, b) => {
      if (a.destNodeIdx !== b.destNodeIdx) {
        return a.destNodeIdx - b.destNodeIdx;
      }
      return a.sourceNodeIdx - b.sourceNodeIdx;
    });

    return allEdges;
  }

  /**
   * Calculate positions for all spline charts in a layer, aligned with the grid system
   */
  calculateSplinePositions(
    network: kan.KANNode[][],
    layerIdx: number,
    node2coord: {[id: string]: NodePosition},
    inputIds: string[]
  ): {[edgeKey: string]: SplinePosition} {
    let sortedEdges = this.sortEdges(network, layerIdx, node2coord, inputIds);
    let positions: {[edgeKey: string]: SplinePosition} = {};

    sortedEdges.forEach((edgeInfo, index) => {
      let sourceCoord = node2coord[edgeInfo.edge.sourceNode.id];
      let destCoord = node2coord[edgeInfo.edge.destNode.id];

      // X position: midpoint between source and destination
      let splineX = (sourceCoord.cx + destCoord.cx) / 2;

      // Y position: aligned with grid system
      let splineY = this.nodeIndexScale(index) + this.rectSize / 2;

      // Constrain X within SVG bounds
      let minX = this.padding + this.splineChartWidth / 2;
      let maxX = this.svgWidth - this.padding - this.splineChartWidth / 2;
      splineX = Math.max(minX, Math.min(maxX, splineX));

      positions[edgeInfo.edgeKey] = {
        x: splineX,
        y: splineY
      };
    });

    return positions;
  }

  /**
   * Calculate the path data for a link between a node and a spline chart,
   * or between a spline chart and a node
   */
  calculateLinkPath(
    edge: kan.KANEdge,
    node2coord: {[id: string]: NodePosition},
    splinePosition: SplinePosition,
    edgeIndex: number,
    totalEdges: number
  ): LinkPath {
    const source = node2coord[edge.sourceNode.id];
    const dest = node2coord[edge.destNode.id];

    // First path: source node to spline chart
    const sourceToSpline = {
      source: {
        x: source.cy,
        y: source.cx + this.rectSize / 2 + 2
      },
      target: {
        x: splinePosition.y,
        y: splinePosition.x - this.splineChartWidth / 2
      }
    };

    // Second path: spline chart to destination node
    // Add slight vertical offset for multiple edges to the same node
    const verticalOffset = ((edgeIndex - (totalEdges - 1) / 2) / totalEdges) * 12;
    const splineToTarget = {
      source: {
        x: splinePosition.y,
        y: splinePosition.x + this.splineChartWidth / 2
      },
      target: {
        x: dest.cy + verticalOffset,
        y: dest.cx - this.rectSize / 2
      }
    };

    return { sourceToSpline, splineToTarget };
  }

  /**
   * Get the bounding box for a spline chart at a given position
   */
  getSplineChartBounds(splinePosition: SplinePosition): {
    left: number,
    top: number,
    right: number,
    bottom: number,
    width: number,
    height: number
  } {
    return {
      left: splinePosition.x - this.splineChartWidth / 2,
      top: splinePosition.y - this.splineChartHeight / 2,
      right: splinePosition.x + this.splineChartWidth / 2,
      bottom: splinePosition.y + this.splineChartHeight / 2,
      width: this.splineChartWidth,
      height: this.splineChartHeight
    };
  }

  /**
   * Get the bounding box for a node at a given position
   */
  getNodeBounds(nodePosition: NodePosition): {
    left: number,
    top: number,
    right: number,
    bottom: number,
    width: number,
    height: number
  } {
    return {
      left: nodePosition.cx - this.rectSize / 2,
      top: nodePosition.cy - this.rectSize / 2,
      right: nodePosition.cx + this.rectSize / 2,
      bottom: nodePosition.cy + this.rectSize / 2,
      width: this.rectSize,
      height: this.rectSize
    };
  }

  /**
   * Calculate positions for plus/minus controls for adding/removing neurons
   */
  getPlusMinusControlPosition(layerCx: number): { x: number, y: number } {
    return {
      x: layerCx,
      y: this.padding
    };
  }

  /**
   * Constrain a position to stay within SVG bounds
   */
  constrainToSVGBounds(x: number, y: number, elementWidth: number, elementHeight: number): {x: number, y: number} {
    const minX = this.padding + elementWidth / 2;
    const maxX = this.svgWidth - this.padding - elementWidth / 2;
    const minY = this.padding + elementHeight / 2;
    const maxY = this.svgHeight - this.padding - elementHeight / 2;

    return {
      x: Math.max(minX, Math.min(maxX, x)),
      y: Math.max(minY, Math.min(maxY, y))
    };
  }

  /**
   * Calculate the maximum height needed for the SVG based on network structure
   */
  calculateRequiredHeight(
    network: kan.KANNode[][],
    numInputs: number
  ): number {
    // Calculate the maximum number of edges in any single layer
    let maxEdgesInLayer = 0;
    for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
      let currentLayer = network[layerIdx];
      let edgesInLayer = currentLayer.reduce((sum, node) => sum + node.inputEdges.length, 0);
      maxEdgesInLayer = Math.max(maxEdgesInLayer, edgesInLayer);
    }
    let maxYForEdges = this.nodeIndexScale(maxEdgesInLayer);

    // Calculate height needed for input nodes
    let maxYForInputs = this.nodeIndexScale(numInputs);

    // Calculate height needed for intermediate layers
    let maxYForLayers = 0;
    for (let layerIdx = 1; layerIdx < network.length - 1; layerIdx++) {
      let numNodes = network[layerIdx].length;
      maxYForLayers = Math.max(maxYForLayers, this.nodeIndexScale(numNodes));
    }

    // Return the maximum with a minimum height
    return Math.max(maxYForEdges, maxYForInputs, maxYForLayers, 400);
  }

  /**
   * Get all edge information for a layer, including their positions and link paths
   */
  getLayerEdgesLayout(
    network: kan.KANNode[][],
    layerIdx: number,
    node2coord: {[id: string]: NodePosition},
    inputIds: string[]
  ): {
    edgeKey: string,
    edge: kan.KANEdge,
    splinePosition: SplinePosition,
    linkPath: LinkPath,
    edgeIndex: number,
    totalEdgesForNode: number
  }[] {
    const splinePositions = this.calculateSplinePositions(network, layerIdx, node2coord, inputIds);
    const edgesLayout: {
      edgeKey: string,
      edge: kan.KANEdge,
      splinePosition: SplinePosition,
      linkPath: LinkPath,
      edgeIndex: number,
      totalEdgesForNode: number
    }[] = [];

    const currentLayer = network[layerIdx];

    for (let nodeIdx = 0; nodeIdx < currentLayer.length; nodeIdx++) {
      const node = currentLayer[nodeIdx];
      const totalEdgesForNode = node.inputEdges.length;

      for (let edgeIdx = 0; edgeIdx < node.inputEdges.length; edgeIdx++) {
        const edge = node.inputEdges[edgeIdx];
        const edgeKey = `${edge.sourceNode.id}-${edge.destNode.id}`;
        const splinePosition = splinePositions[edgeKey];
        const linkPath = this.calculateLinkPath(edge, node2coord, splinePosition, edgeIdx, totalEdgesForNode);

        edgesLayout.push({
          edgeKey,
          edge,
          splinePosition,
          linkPath,
          edgeIndex: edgeIdx,
          totalEdgesForNode
        });
      }
    }

    return edgesLayout;
  }

  /**
   * Get complete layout information for the entire network
   */
  getCompleteNetworkLayout(
    network: kan.KANNode[][],
    inputIds: string[]
  ): {
    layers: LayerLayout[],
    node2coord: {[id: string]: NodePosition},
    edgesByLayer: {[layerIdx: number]: {
      edgeKey: string,
      edge: kan.KANEdge,
      splinePosition: SplinePosition,
      linkPath: LinkPath,
      edgeIndex: number,
      totalEdgesForNode: number
    }[]}
  } {
    const { layers, node2coord } = this.getNetworkLayout(network, inputIds);
    const edgesByLayer: {[layerIdx: number]: any[]} = {};

    // Calculate edge layouts for each layer
    for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
      edgesByLayer[layerIdx] = this.getLayerEdgesLayout(network, layerIdx, node2coord, inputIds);
    }

    return {
      layers,
      node2coord,
      edgesByLayer
    };
  }

  /**
   * Get rendering properties for a node (for use with D3 or direct DOM manipulation)
   */
  getNodeRenderProps(nodePosition: NodePosition): {
    svgTransform: string,
    divLeft: string,
    divTop: string,
    x: number,
    y: number
  } {
    const x = nodePosition.cx - this.rectSize / 2;
    const y = nodePosition.cy - this.rectSize / 2;

    return {
      svgTransform: `translate(${x},${y})`,
      divLeft: `${x + this.padding}px`,
      divTop: `${y + this.padding}px`,
      x,
      y
    };
  }

  /**
   * Get rendering properties for a spline chart
   */
  getSplineChartRenderProps(splinePosition: SplinePosition): {
    divLeft: string,
    divTop: string,
    width: string,
    height: string
  } {
    const bounds = this.getSplineChartBounds(splinePosition);
    
    return {
      divLeft: `${bounds.left + 1}px`,
      divTop: `${bounds.top}px`,
      width: `${bounds.width}px`,
      height: `${bounds.height}px`
    };
  }
}
