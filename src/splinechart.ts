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
  title?: string;
  width?: number;
  height?: number;
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
    title: "Learnable Function",
    width: 300,
    height: 200
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

    // Reduce top margin if title exists
    if (this.settings.title && this.settings.title.trim() !== "") {
      this.margin.top = 30; // Minimal margin to match bottom when no axes
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
      this.svg.append("text")
        .attr("class", "spline-title")
        .attr("x", this.width / 2)
        .attr("y", -10)
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
      this.svg.append("text")
        .attr("class", "spline-title")
        .attr("x", this.width / 2)
        .attr("y", -10)
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
        "box-shadow": "0 2px 5px rgba(0,0,0,0.2)",
        "transition": "all 0.2s ease"
      });
      
      // Add hover effects
      this.chartContainer
        .on("mouseenter", function() {
          d3.select(this).style({
            "opacity": "1.0",
            "border": "2px solid #666"
          });
        })
        .on("mouseleave", function() {
          d3.select(this).style({
            "opacity": null,
            "border": "2px solid black"
          });
        });
    } else {
      this.chartContainer.style({
        "border": "none",
        "border-radius": "0",
        "box-shadow": "none",
        "transition": null
      });
      
      // Remove hover effects
      this.chartContainer
        .on("mouseenter", null)
        .on("mouseleave", null);
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
   */
  updateFunction(learnableFunction: LearnableFunction): void {
    this.currentFunction = learnableFunction;
    
    // Update Y scale based on function range
    this.updateYScale();
    
    // Clear previous function visualization
    this.svg.selectAll(".spline-curve").remove();
    this.svg.selectAll(".control-point").remove();
    this.svg.selectAll(".knot-line").remove();
    
    // Clear old control paths only if showOldControlPaths is false
    if (!this.settings.showOldControlPaths) {
      this.svg.selectAll(".control-polygon").remove();
    }

    // Draw the spline curve
    this.drawSplineCurve();

    // Draw control points if enabled
    if (this.settings.showControlPoints) {
      this.drawControlPoints();
    }

    // Draw knots if enabled
    if (this.settings.showKnots) {
      this.drawKnots();
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
      .duration(0) // 300
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

    // Draw control points
    this.svg.selectAll(".control-point")
      .data(controlPointData)
      .enter()
      .append("circle")
      .attr("class", "control-point")
      .attr("cx", (d: any) => this.xScale(d.x))
      .attr("cy", (d: any) => this.yScale(d.y))
      .attr("r", 4)
      .style("fill", "#E67E22")
      .style("stroke", "#D35400")
      .style("stroke-width", 2)
      .append("title")
      .text((d: any) => `Control Point ${d.index}\nPosition: (${d.x.toFixed(2)}, ${d.y.toFixed(3)})`);

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