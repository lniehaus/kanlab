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
   * Calculate the total number of columns in the network visualization
   * Columns alternate: heatmap -> spline -> heatmap -> spline -> ... -> heatmap
   * For N layers: 1 input + (N-1) hidden/output = N heatmap columns
   *               N-1 spline columns between them
   *               Total = N + (N-1) = 2N - 1 columns
   */
  private calculateTotalColumns(numLayers: number): number {
    return 2 * numLayers - 1;
  }

  /**
   * Calculate X coordinate for a specific column index with equal spacing
   * Column indices: 0 = input heatmaps, 1 = first splines, 2 = first hidden heatmaps, etc.
   */
  private calculateColumnX(columnIdx: number, totalColumns: number): number {
    const startX = this.rectSize / 2 + 50; // Starting position for input layer
    const endX = this.svgWidth + this.rectSize / 2; // Ending position for output layer
    const availableWidth = endX - startX;
    const columnSpacing = availableWidth / (totalColumns - 1);
    
    return startX + (columnIdx * columnSpacing);
  }

  /**
   * Map layer index to column index
   * Layer 0 -> Column 0 (input heatmaps)
   * Layer 1 -> Column 2 (first hidden layer heatmaps, after first spline column)
   * Layer 2 -> Column 4 (second hidden layer heatmaps, after second spline column)
   * etc.
   */
  private layerToColumnIndex(layerIdx: number): number {
    return layerIdx * 2;
  }

  /**
   * Calculate column index for spline charts between two layers
   * Splines between layer 0 and 1 -> Column 1
   * Splines between layer 1 and 2 -> Column 3
   * etc.
   */
  private splineColumnIndex(destLayerIdx: number): number {
    return destLayerIdx * 2 - 1;
  }

  /**
   * Calculate X coordinate for a layer in the network
   * Uses column-based positioning for equal spacing
   */
  calculateLayerX(layerIdx: number, numLayers: number): number {
    const totalColumns = this.calculateTotalColumns(numLayers);
    const columnIdx = this.layerToColumnIndex(layerIdx);
    return this.calculateColumnX(columnIdx, totalColumns);
  }

  /**
   * Calculate the number of elements in each column
   * Columns alternate between node layers (heatmaps) and edge layers (spline charts)
   * For input features, returns the span from first to last active feature (not just count)
   */
  private getColumnElementCounts(network: kan.KANNode[][], numInputs: number, inputIds: string[]): number[] {
    const counts: number[] = [];
    const numLayers = network.length;
    
    // Column 0: input layer heatmaps - calculate span from first to last active feature
    // The active features are those in the first layer of the network
    const activeInputIds = network[0].map(node => node.id);
    
    // Find first and last index of active features in the full inputIds array
    let firstActiveIdx = -1;
    let lastActiveIdx = -1;
    
    for (let i = 0; i < inputIds.length; i++) {
      if (activeInputIds.indexOf(inputIds[i]) !== -1) {
        if (firstActiveIdx === -1) {
          firstActiveIdx = i;
        }
        lastActiveIdx = i;
      }
    }
    
    // The span is from first to last active feature (inclusive)
    // This means if features 0 and 6 are active, span is 7 positions
    if (firstActiveIdx !== -1 && lastActiveIdx !== -1) {
      counts.push(lastActiveIdx - firstActiveIdx + 1);
    } else {
      // Fallback to number of inputs if no active features found
      counts.push(numInputs);
    }
    
    // For each layer after input
    for (let layerIdx = 1; layerIdx < numLayers; layerIdx++) {
      // Column for spline charts (edges coming into this layer)
      const numEdges = network[layerIdx].reduce((sum, node) => sum + node.inputEdges.length, 0);
      counts.push(numEdges);
      
      // Column for heatmaps (nodes in this layer)
      counts.push(network[layerIdx].length);
    }
    
    return counts;
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
    
    // Calculate column element counts and find max
    const columnCounts = this.getColumnElementCounts(network, inputIds.length, inputIds);
    const maxColumnCount = Math.max(...columnCounts);
    
    // Calculate vertical offset for input layer (column 0)
    const inputColumnCount = columnCounts[0];
    const inputOffset = ((maxColumnCount - inputColumnCount) * (this.rectSize + this.spacing)) / 2;

    // Input layer (layer 0)
    const inputCx = this.calculateLayerX(0, numLayers);
    const inputPositions: NodePosition[] = [];
    inputIds.forEach((nodeId, i) => {
      const pos = this.getNodePosition(i, inputCx, inputOffset);
      node2coord[nodeId] = pos;
      inputPositions.push(pos);
    });
    layers.push({
      cx: inputCx,
      nodePositions: inputPositions,
      layerIndex: 0
    });

    // Hidden layers and output layer - initially positioned using grid
    for (let layerIdx = 1; layerIdx < numLayers; layerIdx++) {
      const layerNodes = network[layerIdx];
      const cx = this.calculateLayerX(layerIdx, numLayers);
      const nodePositions: NodePosition[] = [];
      
      // Column index for this layer's heatmaps
      const columnIdx = layerIdx * 2;
      const layerColumnCount = columnCounts[columnIdx];
      const layerOffset = ((maxColumnCount - layerColumnCount) * (this.rectSize + this.spacing)) / 2;

      layerNodes.forEach((node, i) => {
        const pos = this.getNodePosition(i, cx, layerOffset);
        node2coord[node.id] = pos;
        nodePositions.push(pos);
      });

      layers.push({
        cx: cx,
        nodePositions: nodePositions,
        layerIndex: layerIdx
      });
    }

    // Adjust Y positions of hidden layer nodes based on their input spline charts
    for (let layerIdx = 1; layerIdx < numLayers - 1; layerIdx++) {
      // Calculate spline positions for this layer
      const splinePositions = this.calculateSplinePositions(network, layerIdx, node2coord, inputIds, columnCounts, maxColumnCount);
      
      // For each node in the hidden layer, calculate mean Y of input splines
      const layerNodes = network[layerIdx];
      layerNodes.forEach((node, nodeIdx) => {
        const inputSplineYs: number[] = [];
        
        // Collect Y positions of all input spline charts
        node.inputEdges.forEach(edge => {
          const edgeKey = `${edge.sourceNode.id}-${edge.destNode.id}`;
          const splinePos = splinePositions[edgeKey];
          if (splinePos) {
            inputSplineYs.push(splinePos.y);
          }
        });
        
        // Calculate mean Y position
        if (inputSplineYs.length > 0) {
          const meanY = inputSplineYs.reduce((sum, y) => sum + y, 0) / inputSplineYs.length;
          
          // Update node position
          node2coord[node.id].cy = meanY;
          layers[layerIdx].nodePositions[nodeIdx].cy = meanY;
        }
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
  getNodePosition(nodeIndex: number, cx: number, yOffset: number = 0): NodePosition {
    return {
      cx: cx,
      cy: this.nodeIndexScale(nodeIndex) + this.rectSize / 2 + yOffset
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
    inputIds: string[],
    columnCounts?: number[],
    maxColumnCount?: number
  ): {[edgeKey: string]: SplinePosition} {
    let sortedEdges = this.sortEdges(network, layerIdx, node2coord, inputIds);
    let positions: {[edgeKey: string]: SplinePosition} = {};
    
    // Calculate vertical offset for spline chart column
    let splineOffset = 0;
    if (columnCounts && maxColumnCount) {
      const splineColumnIdx = layerIdx * 2 - 1;
      const splineColumnCount = columnCounts[splineColumnIdx];
      splineOffset = ((maxColumnCount - splineColumnCount) * (this.rectSize + this.spacing)) / 2;
    }

    // Calculate the number of layers to determine column positioning
    const numLayers = network.length;
    const totalColumns = this.calculateTotalColumns(numLayers);
    const splineColIdx = this.splineColumnIndex(layerIdx);
    const splineX = this.calculateColumnX(splineColIdx, totalColumns);

    sortedEdges.forEach((edgeInfo, index) => {
      // X position: use dedicated column position for uniform spacing
      // (no longer calculating midpoint between nodes)

      // Y position: aligned with grid system, with offset for centering
      let splineY = this.nodeIndexScale(index) + this.rectSize / 2 + splineOffset;

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
        y: splinePosition.x - (this.splineChartWidth / 2) - 1
      }
    };

    // Second path: spline chart to destination node
    // Add slight vertical offset for multiple edges to the same node
    const verticalOffset = ((edgeIndex - (totalEdges - 1) / 2) / totalEdges) * 12;
    const splineToTarget = {
      source: {
        x: splinePosition.y,
        y: splinePosition.x + (this.splineChartWidth / 2) + 1
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
   * Also accounts for centered input column potentially extending beyond bottom
   */
  calculateRequiredHeight(
    network: kan.KANNode[][],
    numInputs: number,
    inputIds?: string[],
    padding: number = 50
  ): number {
    // Calculate the maximum number of edges in any single layer
    let maxEdgesInLayer = 0;
    for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
      let currentLayer = network[layerIdx];
      let edgesInLayer = currentLayer.reduce((sum, node) => sum + node.inputEdges.length, 0);
      maxEdgesInLayer = Math.max(maxEdgesInLayer, edgesInLayer);
    }
    let maxYForEdges = this.nodeIndexScale(maxEdgesInLayer);

    // Calculate height needed for input nodes (uncentered)
    let maxYForInputs = this.nodeIndexScale(numInputs);

    // Calculate height needed for intermediate layers
    let maxYForLayers = 0;
    for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
      let numNodes = network[layerIdx].length;
      maxYForLayers = Math.max(maxYForLayers, this.nodeIndexScale(numNodes));
    }

    // Calculate the maximum column count to determine centering
    let maxColumnCount = Math.max(maxEdgesInLayer, numInputs);
    for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
      maxColumnCount = Math.max(maxColumnCount, network[layerIdx].length);
    }
    
    // Calculate the input column active span to determine its offset when centered
    let inputColumnActiveSpan = numInputs;
    if (inputIds && inputIds.length > 0) {
      const activeInputIds = network[0].map(node => node.id);
      let firstActiveIdx = -1;
      let lastActiveIdx = -1;
      
      for (let i = 0; i < inputIds.length; i++) {
        if (activeInputIds.indexOf(inputIds[i]) !== -1) {
          if (firstActiveIdx === -1) {
            firstActiveIdx = i;
          }
          lastActiveIdx = i;
        }
      }
      
      if (firstActiveIdx !== -1 && lastActiveIdx !== -1) {
        inputColumnActiveSpan = lastActiveIdx - firstActiveIdx + 1;
      }
    }
    
    // When input column is centered based on active span, the offset is:
    const inputOffset = ((maxColumnCount - inputColumnActiveSpan) * (this.rectSize + this.spacing)) / 2;
    
    // The bottom of the last input element (all inputs are drawn) will be at:
    const inputColumnBottom = inputOffset + this.nodeIndexScale(numInputs);
    
    // Ensure SVG height accommodates the potentially extended input column
    maxYForEdges = maxYForEdges + padding;
    maxYForInputs = maxYForInputs + padding;
    maxYForLayers = maxYForLayers + padding;
    const height = Math.max(maxYForEdges, maxYForInputs, maxYForLayers, inputColumnBottom, 400);
    return height;
  }

  /**
   * Get all edge information for a layer, including their positions and link paths
   */
  getLayerEdgesLayout(
    network: kan.KANNode[][],
    layerIdx: number,
    node2coord: {[id: string]: NodePosition},
    inputIds: string[],
    columnCounts?: number[],
    maxColumnCount?: number
  ): {
    edgeKey: string,
    edge: kan.KANEdge,
    splinePosition: SplinePosition,
    linkPath: LinkPath,
    edgeIndex: number,
    totalEdgesForNode: number
  }[] {
    const splinePositions = this.calculateSplinePositions(network, layerIdx, node2coord, inputIds, columnCounts, maxColumnCount);
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
    
    // Calculate column counts for centering
    const columnCounts = this.getColumnElementCounts(network, inputIds.length, inputIds);
    const maxColumnCount = Math.max(...columnCounts);

    // Calculate edge layouts for each layer
    for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
      edgesByLayer[layerIdx] = this.getLayerEdgesLayout(network, layerIdx, node2coord, inputIds, columnCounts, maxColumnCount);
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
