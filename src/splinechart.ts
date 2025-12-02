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

import * as d3 from 'd3';
import { LearnableFunction } from './kan';

export interface SplineChartSettings {
  [key: string]: any;
  showControlPoints?: boolean;
  showOldControlPaths?: boolean;
  showKnots?: boolean;
  showGrid?: boolean;
  showXAxisLabels?: boolean;
  showYAxisLabels?: boolean;
  showXAxisValues?: boolean;
  showYAxisValues?: boolean;
  showBorder?: boolean;
  showActivationHistogram?: boolean;
  histogramOpacity?: number;
  histogramColor?: string;
  outputHistogramColor?: string;
  histogramSize?: number;
  histogramGap?: number;
  title?: string;
  width?: number;
  height?: number;
  interactive?: boolean;
}

/**
 * A chart for visualizing B-spline learnable functions from KAN networks.
 * Shows the spline curve, control points, knot positions, and grid lines.
 */
export class SplineChart {
  protected settings: SplineChartSettings = {
    showControlPoints: false,
    showOldControlPaths: false,
    showKnots: false,
    showGrid: false,
    showXAxisLabels: false,
    showYAxisLabels: false,
    showXAxisValues: false,
    showYAxisValues: false,
    showBorder: false,
    showActivationHistogram: false,
    histogramOpacity: 0.3,
    histogramColor: "#4A90E2",
    outputHistogramColor: "#4A90E2",
    histogramSize: 50,
    histogramGap: 10,
    title: "Learnable Function",
    width: 300,
    height: 200,
    interactive: false
  };
  
  protected svg: any;
  protected xScale: any;
  protected yScale: any;
  private container: any;
  private chartContainer: any;
  protected width: number;
  protected height: number;
  // private baseMargin = { top: 30, right: 20, bottom: 40, left: 40 };
  // private margin = { top: 30, right: 20, bottom: 40, left: 40 };
  private baseMargin = { top: 2, right: 2, bottom: 2, left: 2 };
  private margin = { top: 2, right: 2, bottom: 2, left: 2 };
  private currentFunction: LearnableFunction | null = null;
  private dragBehavior: any = null;
  private onControlPointChange: ((index: number, newValue: number) => void) | null = null;
  private onDragStart: (() => void) | null = null;
  private onDragEnd: (() => void) | null = null;
  private isDragging: boolean = false;
  private targetYDomain: [number, number] | null = null;
  private smoothingTimer: any = null;

  constructor(container: any, userSettings?: SplineChartSettings) {
    if (userSettings != null) {
      // Overwrite the defaults with the user-specified settings
      for (let prop in userSettings) {
        this.settings[prop] = userSettings[prop];
      }
    }

    this.updateMargins();
    this.width = this.settings.width! - this.margin.left - this.margin.right;
    this.height = this.settings.height! - this.margin.top - this.margin.bottom;
    this.container = container;

    this.initializeChart();
  }

  private updateMargins(): void {
    // Start with base margins
    this.margin = { ...this.baseMargin };

    // Expand left margin if y-axis elements are shown
    if (this.settings.showYAxisLabels && this.settings.showYAxisValues) {
      this.margin.left = 50;
    } else if (this.settings.showYAxisLabels || this.settings.showYAxisValues) {
      this.margin.left = 30; 
    }

    // Reduce right margin when no additional right-side content is needed
    // This is especially useful when the chart is purely functional without legends or extra labels
    if (this.settings.showYAxisLabels || this.settings.showYAxisValues) {
      this.margin.right = 20; // Minimal margin to match left side
    }

    // Extend bottom margin if x-axis elements are shown
    if (this.settings.showXAxisLabels && this.settings.showXAxisValues) {
      this.margin.bottom = 40; 
    } else if (this.settings.showXAxisLabels || this.settings.showXAxisValues) {
      this.margin.bottom = 30; 
    }

    // Extend top margin for title and histogram
    if (this.settings.title && this.settings.title.trim() !== "") {
      this.margin.top = 30; // Minimal margin when title exists
    }
    
    // Add additional space at top if histogram is shown (Seaborn joint plot style)
    if (this.settings.showActivationHistogram) {
      const histogramSize = this.settings.histogramSize || 50;
      const histogramGap = this.settings.histogramGap || 10;
      this.margin.top += histogramSize + histogramGap;
      this.margin.right += histogramSize + histogramGap;
    }
  }

  private initializeChart(): void {
    // Create a wrapper div for the chart with optional border
    this.chartContainer = this.container
      .append("div")
      .attr("class", "spline-chart-wrapper")
      .style({
        "display": "inline-block",
        "position": "relative"
      });

    // Apply border if enabled
    this.updateBorder();

    // Create SVG
    this.svg = this.chartContainer
      .append("svg")
      .attr("width", this.settings.width)
      .attr("height", this.settings.height)
      .append("g")
      .attr("transform", `translate(${this.margin.left},${this.margin.top})`);

    // Set up scales
    this.xScale = d3.scale.linear()
      .domain([-1, 1])
      .range([0, this.width]);

    this.yScale = d3.scale.linear()
      .domain([-2, 2])
      .range([this.height, 0]);

    // Add axes
    this.addAxes();

    // Add title
    if (this.settings.title) {
      // Position title above histogram if shown, otherwise just above plot
      const titleY = this.settings.showActivationHistogram ? -70 : -10;
      this.svg.append("text")
        .attr("class", "spline-title")
        .attr("x", this.width / 2)
        .attr("y", titleY)
        .attr("text-anchor", "middle")
        .style("font-size", "14px")
        .style("font-weight", "bold")
        .text(this.settings.title);
    }

    // Add grid if enabled
    if (this.settings.showGrid) {
      this.addGrid();
    }
  }

  private recreateChart(): void {
    // Remove existing SVG
    this.chartContainer.select("svg").remove();

    // Update margins and dimensions
    this.updateMargins();
    this.width = this.settings.width! - this.margin.left - this.margin.right;
    this.height = this.settings.height! - this.margin.top - this.margin.bottom;

    // Recreate SVG with new transform
    this.svg = this.chartContainer
      .append("svg")
      .attr("width", this.settings.width)
      .attr("height", this.settings.height)
      .append("g")
      .attr("transform", `translate(${this.margin.left},${this.margin.top})`);

    // Update scales with new dimensions
    this.xScale = d3.scale.linear()
      .domain([-1, 1])
      .range([0, this.width]);

    this.yScale = d3.scale.linear()
      .domain([-2, 2])
      .range([this.height, 0]);

    // Add axes
    this.addAxes();

    // Add title
    if (this.settings.title) {
      // Position title above histogram if shown, otherwise just above plot
      const titleY = this.settings.showActivationHistogram ? -70 : -10;
      this.svg.append("text")
        .attr("class", "spline-title")
        .attr("x", this.width / 2)
        .attr("y", titleY)
        .attr("text-anchor", "middle")
        .style("font-size", "14px")
        .style("font-weight", "bold")
        .text(this.settings.title);
    }

    // Add grid if enabled
    if (this.settings.showGrid) {
      this.addGrid();
    }
  }

  private updateBorder(): void {
    if (this.settings.showBorder) {
      this.chartContainer.style({
        "border": "2px solid black",
        "border-radius": "3px",
        "box-shadow": "0 2px 5px rgba(0,0,0,0.2)"
      });
    } else {
      this.chartContainer.style({
        "border": "none",
        "border-radius": "0",
        "box-shadow": "none"
      });
    }
  }

  private addAxes(): void {
    // X axis
    let xAxis = d3.svg.axis()
      .scale(this.xScale)
      .orient("bottom");

    if (this.settings.showXAxisValues) {
      xAxis.ticks(5);
    } else {
      xAxis.ticks(0);
    }

    this.svg.append("g")
      .attr("class", "x axis")
      .attr("transform", `translate(0,${this.height})`)
      .call(xAxis);

    // Y axis
    let yAxis = d3.svg.axis()
      .scale(this.yScale)
      .orient("left");

    if (this.settings.showYAxisValues) {
      yAxis.ticks(5);
    } else {
      yAxis.ticks(0);
    }

    this.svg.append("g")
      .attr("class", "y axis")
      .call(yAxis);

    // Add axis labels conditionally
    if (this.settings.showYAxisLabels) {
      this.svg.append("text")
        .attr("class", "axis-label y-label")
        .attr("transform", "rotate(-90)")
        .attr("y", 0 - this.margin.left)
        .attr("x", 0 - (this.height / 2))
        .attr("dy", "1em")
        .style("text-anchor", "middle")
        .style("font-size", "12px")
        .text("f(x)");
    }

    if (this.settings.showXAxisLabels) {
      this.svg.append("text")
        .attr("class", "axis-label x-label")
        .attr("transform", `translate(${this.width / 2}, ${this.height + this.margin.bottom - 5})`)
        .style("text-anchor", "middle")
        .style("font-size", "12px")
        .text("x");
    }
  }

  private addGrid(): void {
    // Vertical grid lines
    this.svg.append("g")
      .attr("class", "grid")
      .selectAll("line.vertical")
      .data(this.xScale.ticks(5))
      .enter()
      .append("line")
      .attr("class", "vertical")
      .attr("x1", (d: number) => this.xScale(d))
      .attr("x2", (d: number) => this.xScale(d))
      .attr("y1", 0)
      .attr("y2", this.height)
      .style("stroke", "#e0e0e0")
      .style("stroke-width", 1);

    // Horizontal grid lines
    this.svg.append("g")
      .attr("class", "grid")
      .selectAll("line.horizontal")
      .data(this.yScale.ticks(5))
      .enter()
      .append("line")
      .attr("class", "horizontal")
      .attr("x1", 0)
      .attr("x2", this.width)
      .attr("y1", (d: number) => this.yScale(d))
      .attr("y2", (d: number) => this.yScale(d))
      .style("stroke", "#e0e0e0")
      .style("stroke-width", 1);
  }

  /**
   * Update the chart with a new learnable function
   * Uses D3's enter/update/exit pattern to preserve DOM elements and event handlers
   */
  updateFunction(learnableFunction: LearnableFunction, inputHistogramData?: number[], outputHistogramData?: number[]): void {
    this.currentFunction = learnableFunction;
    
    // Update Y scale based on function range
    this.updateYScale();
    
    // Use update pattern instead of remove/recreate for better performance
    // and to preserve event handlers 
    
    // Update histograms FIRST (so they're behind the curve)
    if (this.settings.showActivationHistogram) {
      if (inputHistogramData && inputHistogramData.length > 0) {
        this.updateActivationHistogram(inputHistogramData);
      } else {
        // Remove histograms if no data
        this.svg.selectAll(".activation-histogram").remove();
      }
      
      if (outputHistogramData && outputHistogramData.length > 0) {
        this.updateOutputHistogram(outputHistogramData);
      } else {
        // Remove output histograms if no data
        this.svg.selectAll(".output-histogram").remove();
      }
    } else {
      // Remove histograms if not enabled
      this.svg.selectAll(".activation-histogram").remove();
      this.svg.selectAll(".output-histogram").remove();
    }

    // Update the spline curve (using transition instead of remove/create)
    this.updateSplineCurve();

    // Update control points if enabled (using D3 data binding)
    if (this.settings.showControlPoints) {
      this.updateControlPoints();
    } else {
      // Remove control points if not enabled
      this.svg.selectAll(".control-point").remove();
      this.svg.selectAll(".control-polygon").remove();
    }

    // Update knots if enabled
    if (this.settings.showKnots) {
      this.updateKnots();
    } else {
      // Remove knots if not enabled
      this.svg.selectAll(".knot-line").remove();
    }
  }

  private updateYScale(): void {
    if (!this.currentFunction) return;

    // Sample the function to determine Y range
    const numSamples = 100;
    let minY = Infinity;
    let maxY = -Infinity;

    for (let i = 0; i <= numSamples; i++) {
      const x = -1 + (2 * i) / numSamples;
      const y = this.currentFunction.evaluate(x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }

    // If showing control points, include them in the Y range calculation
    if (this.settings.showControlPoints) {
      const controlPoints = this.currentFunction.controlPoints;
      for (let cp of controlPoints) {
        minY = Math.min(minY, cp);
        maxY = Math.max(maxY, cp);
      }
    }

    // Add some padding
    const padding = Math.max(0.5, (maxY - minY) * 0.1);
    minY -= padding;
    maxY += padding;

    // For interactive charts, ensure minimum range of -1.1 to +1.1
    if (this.settings.interactive) {
      minY = Math.min(minY, -1.1);
      maxY = Math.max(maxY, 1.1);
    }

    // If dragging, store target domain and apply smoothing
    if (this.isDragging) {
      this.targetYDomain = [minY, maxY];
      this.smoothYScale();
      return;
    }

    // Update scale
    this.yScale.domain([minY, maxY]);

    // Update Y axis
    let yAxis = d3.svg.axis()
      .scale(this.yScale)
      .orient("left");

    if (this.settings.showYAxisValues) {
      yAxis.ticks(5);
    } else {
      yAxis.ticks(0);
    }

    this.svg.select(".y.axis")
      .transition()
      .duration(200)
      .call(yAxis);

    // Update horizontal grid lines
    if (this.settings.showGrid) {
      // Remove old grid lines and redraw them
      this.svg.selectAll("line.horizontal").remove();
      
      this.svg.select(".grid")
        .selectAll("line.horizontal")
        .data(this.yScale.ticks(5))
        .enter()
        .append("line")
        .attr("class", "horizontal")
        .attr("x1", 0)
        .attr("x2", this.width)
        .attr("y1", (d: number) => this.yScale(d))
        .attr("y2", (d: number) => this.yScale(d))
        .style("stroke", "#e0e0e0")
        .style("stroke-width", 1);
    }
  }

  private drawSplineCurve(): void {
    if (!this.currentFunction) return;

    // Generate points along the spline curve
    const numPoints = 200;
    const lineData: Array<[number, number]> = [];

    for (let i = 0; i <= numPoints; i++) {
      const x = -1 + (2 * i) / numPoints;
      const y = this.currentFunction.evaluate(x);
      lineData.push([this.xScale(x), this.yScale(y)]);
    }

    // Create line generator
    const line = d3.svg.line()
      .x((d: [number, number]) => d[0])
      .y((d: [number, number]) => d[1])
      .interpolate("linear");

    // Draw the curve
    this.svg.append("path")
      .datum(lineData)
      .attr("class", "spline-curve")
      .attr("d", line)
      .style("fill", "none")
      .style("stroke", "#1B998B")
      .style("stroke-width", 3)
      .style("stroke-linecap", "round");
  }

  /**
   * Update spline curve using D3 transitions instead of remove/recreate
   * This preserves the DOM element and is more performant
   */
  private updateSplineCurve(): void {
    if (!this.currentFunction) return;

    // Generate points along the spline curve
    const numPoints = 200;
    const lineData: Array<[number, number]> = [];

    for (let i = 0; i <= numPoints; i++) {
      const x = -1 + (2 * i) / numPoints;
      const y = this.currentFunction.evaluate(x);
      lineData.push([this.xScale(x), this.yScale(y)]);
    }

    // Create line generator
    const line = d3.svg.line()
      .x((d: [number, number]) => d[0])
      .y((d: [number, number]) => d[1])
      .interpolate("linear");

    // Use D3 data binding pattern (enter/update/exit)
    const curve = this.svg.selectAll(".spline-curve")
      .data([lineData]); // Single element array for single curve

    // ENTER: Create new curve if it doesn't exist
    curve.enter()
      .append("path")
      .attr("class", "spline-curve")
      .style("fill", "none")
      .style("stroke", "#1B998B")
      .style("stroke-width", 3)
      .style("stroke-linecap", "round");

    // UPDATE: Transition existing curve to new path
    // Use CSS transform for smooth animation 
    curve.transition()
      .duration(50) // Fast transition for training
      .attr("d", line);

    // EXIT: Remove any extra curves (shouldn't happen, but good practice)
    curve.exit().remove();
  }

  private drawControlPoints(): void {
    if (!this.currentFunction) return;

    const controlPoints = this.currentFunction.controlPoints;
    const numPoints = controlPoints.length;

    // Map control point indices to x positions
    // For clamped B-splines, control points are roughly uniformly distributed
    const controlPointData = controlPoints.map((y, i) => {
      // Approximate x position based on control point index
      const x = -1 + (2 * i) / (numPoints - 1);
      return { x, y, index: i };
    });

    // Create or update drag behavior if interactive
    if (this.settings.interactive && !this.dragBehavior) {
      this.dragBehavior = d3.behavior.drag()
        .origin((d: any) => {
          return { x: this.xScale(d.x), y: this.yScale(d.y) };
        })
        .on("dragstart", () => {
          // Set dragging flag for smooth Y-axis transitions
          this.isDragging = true;
          // Notify that dragging has started
          if (this.onDragStart) {
            this.onDragStart();
          }
        })
        .on("drag", (d: any) => {
          // Update y position based on drag
          const event: any = d3.event;
          const newY = this.yScale.invert(event.y);
          d.y = newY;
          
          // Update the control point in the function
          if (this.currentFunction) {
            this.currentFunction.controlPoints[d.index] = newY;
          }
          
          // Update visual position
          d3.select(event.sourceEvent.target)
            .attr("cy", event.y);
          
          // Redraw the spline curve and control polygon
          this.redrawCurve();
          
          // Notify callback if set
          if (this.onControlPointChange) {
            this.onControlPointChange(d.index, newY);
          }
        })
        .on("dragend", () => {
          // Clear dragging flag and ensure final update
          this.isDragging = false;
          if (this.smoothingTimer) {
            clearTimeout(this.smoothingTimer);
            this.smoothingTimer = null;
          }
          // Apply final target domain if it exists
          if (this.targetYDomain) {
            this.yScale.domain(this.targetYDomain);
            this.targetYDomain = null;
            // Update axis with the final domain
            let yAxis = d3.svg.axis()
              .scale(this.yScale)
              .orient("left");
            if (this.settings.showYAxisValues) {
              yAxis.ticks(5);
            } else {
              yAxis.ticks(0);
            }
            this.svg.select(".y.axis").call(yAxis);
          }
          // Notify that dragging has ended
          if (this.onDragEnd) {
            this.onDragEnd();
          }
        });
    }

    // Draw control points
    const points = this.svg.selectAll(".control-point")
      .data(controlPointData)
      .enter()
      .append("circle")
      .attr("class", "control-point")
      .attr("cx", (d: any) => this.xScale(d.x))
      .attr("cy", (d: any) => this.yScale(d.y))
      .attr("r", this.settings.interactive ? 6 : 4)
      .style("fill", this.settings.interactive ? "#E67E22" : "#E67E22")
      .style("stroke", "#D35400")
      .style("stroke-width", 2)
      .style("cursor", this.settings.interactive ? "ns-resize" : "default");

    // Add drag behavior if interactive
    if (this.settings.interactive && this.dragBehavior) {
      points.call(this.dragBehavior);
    }

    // Add tooltips
    points.append("title")
      .text((d: any) => `Control Point ${d.index}\nPosition: (${d.x.toFixed(2)}, ${d.y.toFixed(3)})${this.settings.interactive ? '\nDrag to adjust' : ''}`);

    // Draw lines connecting control points
    if (controlPointData.length > 1) {
      const controlLine = d3.svg.line()
        .x((d: any) => this.xScale(d.x))
        .y((d: any) => this.yScale(d.y))
        .interpolate("linear");

      // Create a unique class for the current control polygon
      const timestamp = Date.now();
      const polygonClass = this.settings.showOldControlPaths ? 
        `control-polygon control-polygon-${timestamp}` : 
        "control-polygon";

      this.svg.append("path")
        .datum(controlPointData)
        .attr("class", polygonClass)
        .attr("d", controlLine)
        .style("fill", "none")
        .style("stroke", "#E67E22")
        .style("stroke-width", 1)
        .style("stroke-dasharray", "5,5")
        .style("opacity", this.settings.showOldControlPaths ? 0.3 : 0.5);

      // If showing old paths, fade them over time
      if (this.settings.showOldControlPaths) {
        // Fade older paths
        this.svg.selectAll(".control-polygon")
          .filter(function() {
            return !d3.select(this).classed(`control-polygon-${timestamp}`);
          })
          .transition()
          .duration(500)
          .style("opacity", 0.1);
      }
    }
  }

  /**
   * Update control points using D3 data binding pattern 
   * This preserves DOM elements and event handlers
   */
  private updateControlPoints(): void {
    if (!this.currentFunction) return;

    const controlPoints = this.currentFunction.controlPoints;
    const numPoints = controlPoints.length;

    // Map control point indices to x positions
    const controlPointData = controlPoints.map((y, i) => {
      const x = -1 + (2 * i) / (numPoints - 1);
      return { x, y, index: i };
    });

    // Create or update drag behavior if interactive
    if (this.settings.interactive && !this.dragBehavior) {
      this.dragBehavior = d3.behavior.drag()
        .origin((d: any) => {
          return { x: this.xScale(d.x), y: this.yScale(d.y) };
        })
        .on("dragstart", () => {
          this.isDragging = true;
          if (this.onDragStart) {
            this.onDragStart();
          }
        })
        .on("drag", (d: any) => {
          const event: any = d3.event;
          const newY = this.yScale.invert(event.y);
          d.y = newY;
          
          if (this.currentFunction) {
            this.currentFunction.controlPoints[d.index] = newY;
          }
          
          d3.select(event.sourceEvent.target)
            .attr("cy", event.y);
          
          this.redrawCurve();
          
          if (this.onControlPointChange) {
            this.onControlPointChange(d.index, newY);
          }
        })
        .on("dragend", () => {
          this.isDragging = false;
          if (this.smoothingTimer) {
            clearTimeout(this.smoothingTimer);
            this.smoothingTimer = null;
          }
          if (this.targetYDomain) {
            this.yScale.domain(this.targetYDomain);
            this.targetYDomain = null;
            let yAxis = d3.svg.axis()
              .scale(this.yScale)
              .orient("left");
            if (this.settings.showYAxisValues) {
              yAxis.ticks(5);
            } else {
              yAxis.ticks(0);
            }
            this.svg.select(".y.axis").call(yAxis);
          }
          if (this.onDragEnd) {
            this.onDragEnd();
          }
        });
    }

    // Use D3 data binding pattern (enter/update/exit)
    const points = this.svg.selectAll(".control-point")
      .data(controlPointData, (d: any) => d.index); // Key function for proper data binding

    // ENTER: Create new control points if needed
    const pointsEnter = points.enter()
      .append("circle")
      .attr("class", "control-point")
      .attr("r", this.settings.interactive ? 6 : 4)
      .style("fill", "#E67E22")
      .style("stroke", "#D35400")
      .style("stroke-width", 2)
      .style("cursor", this.settings.interactive ? "ns-resize" : "default");

    // Add drag behavior and tooltips to new points
    if (this.settings.interactive && this.dragBehavior) {
      pointsEnter.call(this.dragBehavior);
    }
    
    pointsEnter.append("title");

    // UPDATE: Transition existing points to new positions
    points.transition()
      .duration(50) // Fast transition for training
      .attr("cx", (d: any) => this.xScale(d.x))
      .attr("cy", (d: any) => this.yScale(d.y));

    // Update tooltips
    points.select("title")
      .text((d: any) => `Control Point ${d.index}\nPosition: (${d.x.toFixed(2)}, ${d.y.toFixed(3)})${this.settings.interactive ? '\nDrag to adjust' : ''}`);

    // EXIT: Remove extra control points
    points.exit().remove();

    // Update control polygon
    this.updateControlPolygon(controlPointData);
  }

  /**
   * Update control polygon connecting control points
   */
  private updateControlPolygon(controlPointData: any[]): void {
    if (controlPointData.length <= 1) {
      this.svg.selectAll(".control-polygon").remove();
      return;
    }

    const controlLine = d3.svg.line()
      .x((d: any) => this.xScale(d.x))
      .y((d: any) => this.yScale(d.y))
      .interpolate("linear");

    // Use D3 data binding for control polygon
    const polygon = this.svg.selectAll(".control-polygon")
      .data([controlPointData]); // Single polygon

    // ENTER: Create new polygon if needed
    polygon.enter()
      .append("path")
      .attr("class", "control-polygon")
      .style("fill", "none")
      .style("stroke", "#E67E22")
      .style("stroke-width", 1)
      .style("stroke-dasharray", "5,5")
      .style("opacity", 0.5);

    // UPDATE: Transition to new path
    polygon.transition()
      .duration(50)
      .attr("d", controlLine);

    // EXIT: Remove extra polygons
    polygon.exit().remove();
  }

  private drawKnots(): void {
    if (!this.currentFunction) return;

    const knots = this.currentFunction.knotVector;
    const degree = this.currentFunction.degree;

    // Filter out repeated knots at boundaries for cleaner visualization
    const uniqueKnots = knots.filter((knot, i) => {
      // Skip the first degree+1 and last degree+1 knots (boundary repetitions)
      return i > degree && i < knots.length - degree - 1;
    });

    // Draw knot lines
    this.svg.selectAll(".knot-line")
      .data(uniqueKnots)
      .enter()
      .append("line")
      .attr("class", "knot-line")
      .attr("x1", (d: number) => this.xScale(d))
      .attr("x2", (d: number) => this.xScale(d))
      .attr("y1", 0)
      .attr("y2", this.height)
      .style("stroke", "#9C27B0")
      .style("stroke-width", 2)
      .style("stroke-dasharray", "3,3")
      .style("opacity", 0.7)
      .append("title")
      .text((d: number) => `Knot at x = ${d.toFixed(3)}`);
  }

  /**
   * Update knots using D3 data binding
   */
  private updateKnots(): void {
    if (!this.currentFunction) return;

    const knots = this.currentFunction.knotVector;
    const degree = this.currentFunction.degree;

    // Filter out repeated knots at boundaries
    const uniqueKnots = knots.filter((knot, i) => {
      return i > degree && i < knots.length - degree - 1;
    });

    // Use D3 data binding for knot lines
    const knotLines = this.svg.selectAll(".knot-line")
      .data(uniqueKnots);

    // ENTER: Create new knot lines
    const knotLinesEnter = knotLines.enter()
      .append("line")
      .attr("class", "knot-line")
      .attr("y1", 0)
      .attr("y2", this.height)
      .style("stroke", "#9C27B0")
      .style("stroke-width", 2)
      .style("stroke-dasharray", "3,3")
      .style("opacity", 0.7);

    knotLinesEnter.append("title");

    // UPDATE: Transition knot lines to new positions
    knotLines.transition()
      .duration(50)
      .attr("x1", (d: number) => this.xScale(d))
      .attr("x2", (d: number) => this.xScale(d));

    // Update tooltips
    knotLines.select("title")
      .text((d: number) => `Knot at x = ${d.toFixed(3)}`);

    // EXIT: Remove extra knot lines
    knotLines.exit().remove();
  }

  /**
   * Draw activation histogram as semi-transparent bars above the main plot (Seaborn joint plot style)
   */
  private drawActivationHistogram(histogramData: number[]): void {
    if (!histogramData || histogramData.length === 0) return;
    
    const numBins = histogramData.length;
    const histogramHeight = this.settings.histogramSize || 50;
    const histogramGap = this.settings.histogramGap || 10;
    
    // The histogram bins represent the range [-1, 1] (matching xScale domain)
    const xMin = -1;
    const xMax = 1;
    const binWidth = (xMax - xMin) / numBins;
    
    // Create histogram group positioned above the main plot
    const histogramGroup = this.svg.append("g")
      .attr("class", "activation-histogram")
      .attr("transform", `translate(0, ${-histogramHeight - histogramGap})`);
    
    // Draw bars (growing upward from bottom) using xScale for positioning
    histogramGroup.selectAll("rect.histogram-bar")
      .data(histogramData)
      .enter()
      .append("rect")
      .attr("class", "histogram-bar")
      .attr("x", (d: number, i: number) => {
        const binStart = xMin + i * binWidth;
        return this.xScale(binStart);
      })
      .attr("y", (d: number) => histogramHeight - (d * histogramHeight)) // Start from bottom of histogram area
      .attr("width", (d: number, i: number) => {
        const binStart = xMin + i * binWidth;
        const binEnd = xMin + (i + 1) * binWidth;
        return Math.max(1, this.xScale(binEnd) - this.xScale(binStart) - 1);
      })
      .attr("height", (d: number) => d * histogramHeight)
      .style("fill", this.settings.histogramColor || "#4A90E2")
      .style("opacity", this.settings.histogramOpacity || 0.3)
      .style("pointer-events", "none");
  }

  /**
   * Update activation histogram using D3 data binding
   */
  private updateActivationHistogram(histogramData: number[]): void {
    if (!histogramData || histogramData.length === 0) {
      this.svg.selectAll(".activation-histogram").remove();
      return;
    }
    
    const numBins = histogramData.length;
    const histogramHeight = this.settings.histogramSize || 50;
    const histogramGap = this.settings.histogramGap || 10;
    
    const xMin = -1;
    const xMax = 1;
    const binWidth = (xMax - xMin) / numBins;
    
    // Use D3 data binding for histogram group
    let histogramGroup = this.svg.selectAll(".activation-histogram")
      .data([histogramData]); // Single histogram group
    
    // ENTER: Create histogram group if it doesn't exist
    histogramGroup.enter()
      .append("g")
      .attr("class", "activation-histogram")
      .attr("transform", `translate(0, ${-histogramHeight - histogramGap})`);
    
    // Update bars using data binding
    const bars = histogramGroup.selectAll("rect.histogram-bar")
      .data((d: number[]) => d);
    
    // ENTER: Create new bars
    bars.enter()
      .append("rect")
      .attr("class", "histogram-bar")
      .style("fill", this.settings.histogramColor || "#4A90E2")
      .style("opacity", this.settings.histogramOpacity || 0.3)
      .style("pointer-events", "none");
    
    // UPDATE: Transition bars to new values
    bars.transition()
      .duration(50)
      .attr("x", (d: number, i: number) => {
        const binStart = xMin + i * binWidth;
        return this.xScale(binStart);
      })
      .attr("y", (d: number) => histogramHeight - (d * histogramHeight))
      .attr("width", (d: number, i: number) => {
        const binStart = xMin + i * binWidth;
        const binEnd = xMin + (i + 1) * binWidth;
        return Math.max(1, this.xScale(binEnd) - this.xScale(binStart) - 1);
      })
      .attr("height", (d: number) => d * histogramHeight);
    
    // EXIT: Remove extra bars
    bars.exit().remove();
  }

  /**
   * Draw output histogram as semi-transparent bars on the right side (Seaborn joint plot style)
   */
  private drawOutputHistogram(histogramData: number[]): void {
    if (!histogramData || histogramData.length === 0) return;
    
    const numBins = histogramData.length;
    const histogramWidth = this.settings.histogramSize || 50;
    const histogramGap = this.settings.histogramGap || 10;
    
    // The histogram bins represent the range [-2, 2] (matching yScale domain)
    const yMin = -2;
    const yMax = 2;
    const binHeight = (yMax - yMin) / numBins;
    
    // Create histogram group positioned to the right of the main plot
    const histogramGroup = this.svg.append("g")
      .attr("class", "output-histogram")
      .attr("transform", `translate(${this.width + histogramGap}, 0)`);
    
    // Draw bars (growing rightward from left edge) using yScale for positioning
    histogramGroup.selectAll("rect.output-histogram-bar")
      .data(histogramData)
      .enter()
      .append("rect")
      .attr("class", "output-histogram-bar")
      .attr("x", 0) // Start from left edge
      .attr("y", (d: number, i: number) => {
        const binStart = yMin + i * binHeight;
        const binEnd = yMin + (i + 1) * binHeight;
        return this.yScale(binEnd); // yScale is inverted (higher y values = lower on screen)
      })
      .attr("width", (d: number) => d * histogramWidth)
      .attr("height", (d: number, i: number) => {
        const binStart = yMin + i * binHeight;
        const binEnd = yMin + (i + 1) * binHeight;
        return Math.max(1, Math.abs(this.yScale(binStart) - this.yScale(binEnd)) - 1);
      })
      .style("fill", this.settings.outputHistogramColor || "#E24A90")
      .style("opacity", this.settings.histogramOpacity || 0.3)
      .style("pointer-events", "none");
  }

  /**
   * Update output histogram using D3 data binding
   */
  private updateOutputHistogram(histogramData: number[]): void {
    if (!histogramData || histogramData.length === 0) {
      this.svg.selectAll(".output-histogram").remove();
      return;
    }
    
    const numBins = histogramData.length;
    const histogramWidth = this.settings.histogramSize || 50;
    const histogramGap = this.settings.histogramGap || 10;
    
    const yMin = -2;
    const yMax = 2;
    const binHeight = (yMax - yMin) / numBins;
    
    // Use D3 data binding for histogram group
    let histogramGroup = this.svg.selectAll(".output-histogram")
      .data([histogramData]); // Single histogram group
    
    // ENTER: Create histogram group if it doesn't exist
    histogramGroup.enter()
      .append("g")
      .attr("class", "output-histogram")
      .attr("transform", `translate(${this.width + histogramGap}, 0)`);
    
    // Update bars using data binding
    const bars = histogramGroup.selectAll("rect.output-histogram-bar")
      .data((d: number[]) => d);
    
    // ENTER: Create new bars
    bars.enter()
      .append("rect")
      .attr("class", "output-histogram-bar")
      .attr("x", 0)
      .style("fill", this.settings.outputHistogramColor || "#E24A90")
      .style("opacity", this.settings.histogramOpacity || 0.3)
      .style("pointer-events", "none");
    
    // UPDATE: Transition bars to new values
    bars.transition()
      .duration(50)
      .attr("y", (d: number, i: number) => {
        const binStart = yMin + i * binHeight;
        const binEnd = yMin + (i + 1) * binHeight;
        return this.yScale(binEnd);
      })
      .attr("width", (d: number) => d * histogramWidth)
      .attr("height", (d: number, i: number) => {
        const binStart = yMin + i * binHeight;
        const binEnd = yMin + (i + 1) * binHeight;
        return Math.max(1, Math.abs(this.yScale(binStart) - this.yScale(binEnd)) - 1);
      });
    
    // EXIT: Remove extra bars
    bars.exit().remove();
  }

  /**
   * Clear the chart
   */
  clear(): void {
    this.svg.selectAll(".spline-curve").remove();
    this.svg.selectAll(".control-point").remove();
    this.svg.selectAll(".control-polygon").remove();
    this.svg.selectAll(".knot-line").remove();
    this.currentFunction = null;
  }

  /**
   * Smoothly interpolate Y-axis scale towards target domain
   */
  private smoothYScale(): void {
    if (!this.targetYDomain) return;

    const currentDomain = this.yScale.domain();
    const [targetMin, targetMax] = this.targetYDomain;
    const [currentMin, currentMax] = currentDomain;

    // Interpolation factor (0.15 = 15% of the way to target per frame)
    const alpha = 0.15;

    // Calculate new domain
    const newMin = currentMin + (targetMin - currentMin) * alpha;
    const newMax = currentMax + (targetMax - currentMax) * alpha;

    // Check if we're close enough to target (within 0.01)
    const isClose = Math.abs(newMin - targetMin) < 0.01 && Math.abs(newMax - targetMax) < 0.01;

    if (isClose) {
      // Snap to target
      this.yScale.domain([targetMin, targetMax]);
    } else {
      // Apply interpolated domain
      this.yScale.domain([newMin, newMax]);

      // Schedule next smoothing step
      if (this.smoothingTimer) {
        clearTimeout(this.smoothingTimer);
      }
      this.smoothingTimer = setTimeout(() => this.smoothYScale(), 16); // ~60fps
    }

    // Update Y axis without transition (we're handling the smoothing manually)
    let yAxis = d3.svg.axis()
      .scale(this.yScale)
      .orient("left");

    if (this.settings.showYAxisValues) {
      yAxis.ticks(5);
    } else {
      yAxis.ticks(0);
    }

    this.svg.select(".y.axis")
      .call(yAxis);

    // Update horizontal grid lines if shown
    if (this.settings.showGrid) {
      this.svg.selectAll("line.horizontal").remove();
      
      this.svg.select(".grid")
        .selectAll("line.horizontal")
        .data(this.yScale.ticks(5))
        .enter()
        .append("line")
        .attr("class", "horizontal")
        .attr("x1", 0)
        .attr("x2", this.width)
        .attr("y1", (d: number) => this.yScale(d))
        .attr("y2", (d: number) => this.yScale(d))
        .style("stroke", "#e0e0e0")
        .style("stroke-width", 1);
    }
  }

  /**
   * Redraw only the spline curve and control polygon (used during drag)
   */
  /**
   * Redraw curve during drag (optimized with update pattern)
   */
  private redrawCurve(): void {
    if (!this.currentFunction) return;

    // Use update methods instead of remove/recreate
    this.updateSplineCurve();

    // Update control polygon
    const controlPoints = this.currentFunction.controlPoints;
    const numPoints = controlPoints.length;
    const controlPointData = controlPoints.map((y, i) => {
      const x = -1 + (2 * i) / (numPoints - 1);
      return { x, y, index: i };
    });

    this.updateControlPolygon(controlPointData);
  }

  /**
   * Set callback for control point changes
   */
  setOnControlPointChange(callback: (index: number, newValue: number) => void): void {
    this.onControlPointChange = callback;
  }

  /**
   * Set callback for drag start
   */
  setOnDragStart(callback: () => void): void {
    this.onDragStart = callback;
  }

  /**
   * Set callback for drag end
   */
  setOnDragEnd(callback: () => void): void {
    this.onDragEnd = callback;
  }

  /**
   * Update chart settings
   */
  updateSettings(newSettings: SplineChartSettings): void {
    const oldSettings = { ...this.settings };
    
    for (let prop in newSettings) {
      this.settings[prop] = newSettings[prop];
    }

    // Check if margin-affecting settings changed
    const marginAffectingSettings = ['showYAxisLabels', 'showYAxisValues', 'showXAxisLabels', 'showXAxisValues', 'title'];
    const needsRecreation = marginAffectingSettings.some(setting => 
      setting in newSettings && oldSettings[setting] !== newSettings[setting]
    );

    if (needsRecreation) {
      // Store current function to re-render after recreation
      const currentFunc = this.currentFunction;
      this.recreateChart();
      if (currentFunc) {
        this.updateFunction(currentFunc);
      }
    } else {
      // Update border if border setting changed
      if ('showBorder' in newSettings) {
        this.updateBorder();
      }

      // Update axes if only axis settings changed (but not margin-affecting ones)
      if ('showGrid' in newSettings) {
        // Re-add or remove grid
        this.svg.selectAll(".grid").remove();
        if (this.settings.showGrid) {
          this.addGrid();
        }
      }

      // Re-render with new settings
      if (this.currentFunction) {
        this.updateFunction(this.currentFunction);
      }
    }
  }

  private updateAxes(): void {
    // This method is now mainly used for updating existing axes without recreation
    // Remove existing axis labels
    this.svg.selectAll(".axis-label").remove();

    // Update X axis
    let xAxis = d3.svg.axis()
      .scale(this.xScale)
      .orient("bottom");

    if (this.settings.showXAxisValues) {
      xAxis.ticks(5);
    } else {
      xAxis.ticks(0);
    }

    this.svg.select(".x.axis")
      .call(xAxis);

    // Update Y axis
    let yAxis = d3.svg.axis()
      .scale(this.yScale)
      .orient("left");

    if (this.settings.showYAxisValues) {
      yAxis.ticks(5);
    } else {
      yAxis.ticks(0);
    }

    this.svg.select(".y.axis")
      .call(yAxis);

    // Re-add axis labels if enabled
    if (this.settings.showYAxisLabels) {
      this.svg.append("text")
        .attr("class", "axis-label y-label")
        .attr("transform", "rotate(-90)")
        .attr("y", 0 - this.margin.left)
        .attr("x", 0 - (this.height / 2))
        .attr("dy", "1em")
        .style("text-anchor", "middle")
        .style("font-size", "12px")
        .text("f(x)");
    }

    if (this.settings.showXAxisLabels) {
      this.svg.append("text")
        .attr("class", "axis-label x-label")
        .attr("transform", `translate(${this.width / 2}, ${this.height + this.margin.bottom - 5})`)
        .style("text-anchor", "middle")
        .style("font-size", "12px")
        .text("x");
    }
  }

  /**
   * Export the current function as data points
   */
  exportData(numPoints: number = 100): Array<{x: number, y: number}> {
    if (!this.currentFunction) {
      return [];
    }

    const data: Array<{x: number, y: number}> = [];
    for (let i = 0; i <= numPoints; i++) {
      const x = -1 + (2 * i) / numPoints;
      const y = this.currentFunction.evaluate(x);
      data.push({ x, y });
    }

    return data;
  }

  /**
   * Get statistics about the current function
   */
  getFunctionStats(): {
    min: number,
    max: number,
    mean: number,
    controlPointCount: number,
    degree: number
  } | null {
    if (!this.currentFunction) {
      return null;
    }

    const data = this.exportData(200);
    const values = data.map(d => d.y);
    
    return {
      min: Math.min(...values),
      max: Math.max(...values),
      mean: values.reduce((a, b) => a + b, 0) / values.length,
      controlPointCount: this.currentFunction.controlPoints.length,
      degree: this.currentFunction.degree
    };
  }
}

/**
 * Multi-function spline chart for comparing multiple learnable functions
 */
export class MultiSplineChart extends SplineChart {
  private functions: {[id: string]: LearnableFunction} = {};
  private colors: string[] = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"];
  /**
   * Add or update a function
   */
  addFunction(id: string, learnableFunction: LearnableFunction): void {
    this.functions[id] = learnableFunction;
    this.updateMultipleFunctions();
  }

  /**
   * Remove a function
   */
  removeFunction(id: string): void {
    delete this.functions[id];
    this.updateMultipleFunctions();
  }

  /**
   * Clear all functions
   */
  clearAll(): void {
    this.functions = {};
    this.clear();
  }

  private updateMultipleFunctions(): void {
    // Clear previous visualizations
    this.clear();

    const functionIds = Object.keys(this.functions);
    if (functionIds.length === 0) return;

    // Update Y scale to fit all functions
    this.updateYScaleForMultiple();

    // Draw each function with different colors
    let colorIndex = 0;
    for (let id of functionIds) {
      const func = this.functions[id];
      const color = this.colors[colorIndex % this.colors.length];
      this.drawFunctionCurve(func, color, id);
      colorIndex++;
    }
  }

  private updateYScaleForMultiple(): void {
    let minY = Infinity;
    let maxY = -Infinity;

    // Sample all functions to determine Y range
    const functionIds = Object.keys(this.functions);
    for (let id of functionIds) {
      const func = this.functions[id];
      const numSamples = 100;
      for (let i = 0; i <= numSamples; i++) {
        const x = -1 + (2 * i) / numSamples;
        const y = func.evaluate(x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }
    }

    // Add padding
    const padding = Math.max(0.5, (maxY - minY) * 0.1);
    minY -= padding;
    maxY += padding;

    // Update scale and axis
    this.yScale.domain([minY, maxY]);
    
    let yAxis = d3.svg.axis()
      .scale(this.yScale)
      .orient("left");

    if (this.settings.showYAxisValues) {
      yAxis.ticks(5);
    } else {
      yAxis.ticks(0);
    }

    this.svg.select(".y.axis")
      .transition()
      .duration(300)
      .call(yAxis);
  }

  private drawFunctionCurve(func: LearnableFunction, color: string, id: string): void {
    // Generate curve points
    const numPoints = 200;
    const lineData: Array<[number, number]> = [];

    for (let i = 0; i <= numPoints; i++) {
      const x = -1 + (2 * i) / numPoints;
      const y = func.evaluate(x);
      lineData.push([this.xScale(x), this.yScale(y)]);
    }

    // Create line generator
    const line = d3.svg.line()
      .x((d: [number, number]) => d[0])
      .y((d: [number, number]) => d[1])
      .interpolate("linear");

    // Draw the curve
    this.svg.append("path")
      .datum(lineData)
      .attr("class", `spline-curve function-${id}`)
      .attr("d", line)
      .style("fill", "none")
      .style("stroke", color)
      .style("stroke-width", 2)
      .style("stroke-linecap", "round")
      .append("title")
      .text(`Function: ${id}`);
  }
}