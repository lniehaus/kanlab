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
 */
export class NetworkLayoutManager {
  private rectSize: number;
  private spacing: number;
  private padding: number;
  private svgWidth: number;
  private svgHeight: number;
  private splineChartWidth: number;
  private splineChartHeight: number;

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
      let gridY = this.nodeIndexScale(index) + this.rectSize / 2;

      // Constrain within SVG bounds
      let minY = this.padding + this.splineChartHeight / 2;
      let maxY = this.svgHeight - this.padding - this.splineChartHeight / 2;
      let splineY = Math.max(minY, Math.min(maxY, gridY));

      // Constrain X as well
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
   * Calculate the maximum height needed for the SVG based on network structure
   */
  calculateRequiredHeight(
    network: kan.KANNode[][],
    numInputs: number
  ): number {
    // Calculate height needed for all edges (spline charts)
    let totalEdges = 0;
    for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
      let currentLayer = network[layerIdx];
      totalEdges += currentLayer.reduce((sum, node) => sum + node.inputEdges.length, 0);
    }
    let maxYForEdges = this.nodeIndexScale(totalEdges);

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
}
