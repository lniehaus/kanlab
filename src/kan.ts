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

/**
 * An error function and its derivative.
 */
export interface ErrorFunction {
  error: (output: number, target: number) => number;
  der: (output: number, target: number) => number;
}

/** Built-in error functions */
export class Errors {
  public static SQUARE: ErrorFunction = {
    error: (output: number, target: number) =>
               0.5 * Math.pow(output - target, 2),
    der: (output: number, target: number) => output - target
  };
}

/**
 * A learnable univariate function represented as a B-spline.
 * This is the core component of a KAN (Kolmogorov-Arnold Network).
 * 
 * Initialization:
 * - When using "xavier" initialization, this class implements a basis-agnostic
 *   Glorot-like initialization scheme that accounts for B-spline basis function properties.
 * - Each control point is initialized from N(0, σ_m²) where σ_m depends on:
 *   1. The expected squared value of the basis function: E[B_m(x)²]
 *   2. The expected squared value of its derivative: E[B'_m(x)²]
 *   3. The layer dimensions (fanIn, fanOut)
 * - This ensures proper variance preservation in both forward and backward passes.
 * - See KAN_INITIALIZATION.md for details.
 */
export class LearnableFunction {
  id: string;
  controlPoints: number[] = [];
  knotVector: number[] = [];
  gridSize: number;
  degree: number;
  initNoise: number | "xavier" | "linear";
  inputRange: [number, number] = [-6, 6];
  private fanIn: number;
  private fanOut: number;
  
  constructor(
    id: string,
    gridSize: number = 5,
    range: [number, number] = [-6, 6],
    degree: number = 3,
    initNoise: number | "xavier" | "linear" = 0.3,
    fanIn: number = 1,
    fanOut: number = 1
  ) {
    this.id = id;
    this.gridSize = gridSize;
    this.degree = Math.min(degree, gridSize - 1);
    this.initNoise = initNoise;
    this.inputRange = range;
    this.fanIn = Math.max(1, fanIn | 0);
    this.fanOut = Math.max(1, fanOut | 0);
    this.initializeKnotVector();
    this.initializeControlPoints();
  }

  private initializeKnotVector(): void {
    const [min, max] = this.inputRange;
    const numControlPoints = this.gridSize + 1;
    const numKnots = numControlPoints + this.degree + 1;
    
    this.knotVector = [];
    
    // Create clamped knot vector
    // First degree+1 knots are min
    for (let i = 0; i <= this.degree; i++) {
      this.knotVector.push(min);
    }
    
    // Internal knots are uniformly distributed
    const numInternalKnots = numKnots - 2 * (this.degree + 1);
    for (let i = 1; i <= numInternalKnots; i++) {
      this.knotVector.push(min + (max - min) * i / (numInternalKnots + 1));
    }
    
    // Last degree+1 knots are max
    for (let i = 0; i <= this.degree; i++) {
      this.knotVector.push(max);
    }
  }

  private initializeControlPoints(): void {
    const numControlPoints = this.gridSize + 1;
    this.controlPoints = [];

    const randUniform = (a: number, b: number) => a + (b - a) * Math.random();

    if (this.initNoise === "linear") {
      // Linear initialization: creates identity function or negative identity function
      // Uses He-style scaling to determine the scale
      const safeFanIn = Math.max(1, this.fanIn | 0);
      const safeFanOut = Math.max(1, this.fanOut | 0);
      const limit = Math.sqrt(2 / safeFanIn); // He et al. ReLU

      // 50% chance for positive or negative identity
      const isPositive = Math.random() < 0.5;
      for (let i = 0; i < numControlPoints; i++) {
        const t = i / (numControlPoints - 1); // 0 to 1
        if (isPositive) {
          this.controlPoints.push(-limit + 2 * limit * t); // -limit to +limit (identity)
        } else {
          this.controlPoints.push(limit - 2 * limit * t); // +limit to -limit (negative identity)
        }
      }
      return;
    }

    if (this.initNoise === "xavier") {
      // Basis-agnostic Glorot-like initialization for KANs
      // Based on the paper's proposed scheme: σ_m = gain * sqrt(1/D * 2/(d_I * μ_m^(0) + d_O * μ_m^(1)))
      // where μ_m^(0) = E[B_m(x)^2] and μ_m^(1) = E[B'_m(x)^2]
      
      const safeFanIn = Math.max(1, this.fanIn | 0);
      const safeFanOut = Math.max(1, this.fanOut | 0);
      
      // Xavier/Glorot uses gain = 1.0
      const gain = 1.0;
      
      // Compute μ_m^(0) and μ_m^(1) for each basis function
      const mu0 = this.computeBasisFunctionExpectations();
      const mu1 = this.computeBasisDerivativeExpectations();
      
      // Number of basis functions (D in the paper)
      const D = numControlPoints;
      
      // Initialize each control point with basis-specific variance
      for (let m = 0; m < numControlPoints; m++) {
        // Formula from the paper: σ_m = gain * sqrt(1/D * 2/(d_I * μ_m^(0) + d_O * μ_m^(1)))
        const denominator = safeFanIn * mu0[m] + safeFanOut * mu1[m];
        const sigma_m = denominator > 0 ? 
          gain * Math.sqrt((1.0 / D) * (2.0 / denominator)) : 
          gain * Math.sqrt(2.0 / (safeFanIn + safeFanOut)); // Fallback to standard Glorot
        
        // Sample from N(0, σ_m^2) using Box-Muller transform
        const u1 = Math.random();
        const u2 = Math.random();
        const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        this.controlPoints.push(z * sigma_m);
      }
      return;
    }

    // Default: small symmetric noise around 0.
    const noise = typeof this.initNoise === "number" ? this.initNoise : 0.3;
    for (let i = 0; i < numControlPoints; i++) {
      this.controlPoints.push((Math.random() - 0.5) * noise);
    }
  }

  /**
   * Compute E[B_m(x)^2] for each basis function m
   * Assumes input x ~ N(0, 1) as per the paper's assumptions
   */
  private computeBasisFunctionExpectations(): number[] {
    const numControlPoints = this.gridSize + 1;
    const expectations: number[] = [];
    
    // Monte Carlo estimation: sample from N(0, 1)
    const numSamples = 10000;
    const samples: number[] = [];
    
    // Generate samples from N(0, 1) using Box-Muller transform
    for (let i = 0; i < numSamples; i++) {
      const u1 = Math.random();
      const u2 = Math.random();
      const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      // Clamp to input range to avoid out-of-bounds issues
      samples.push(Math.max(this.inputRange[0], Math.min(this.inputRange[1], z)));
    }
    
    // For each basis function (control point)
    for (let m = 0; m < numControlPoints; m++) {
      let sum = 0;
      for (const x of samples) {
        const span = this.findKnotSpan(x);
        const basisValues = this.computeBasisFunctions(span, x);
        const p = this.degree;
        
        // Find which basis function corresponds to control point m
        let basisValue = 0;
        for (let j = 0; j <= p; j++) {
          const controlPointIndex = span - p + j;
          if (controlPointIndex === m) {
            basisValue = basisValues[j];
            break;
          }
        }
        
        sum += basisValue * basisValue;
      }
      expectations.push(sum / numSamples);
    }
    
    return expectations;
  }

  /**
   * Compute E[B'_m(x)^2] for each basis function derivative
   * Assumes input x ~ N(0, 1) as per the paper's assumptions
   */
  private computeBasisDerivativeExpectations(): number[] {
    const numControlPoints = this.gridSize + 1;
    const expectations: number[] = [];
    
    // Monte Carlo estimation: sample from N(0, 1)
    const numSamples = 10000;
    const samples: number[] = [];
    
    // Generate samples from N(0, 1) using Box-Muller transform
    for (let i = 0; i < numSamples; i++) {
      const u1 = Math.random();
      const u2 = Math.random();
      const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      // Clamp to input range to avoid out-of-bounds issues
      samples.push(Math.max(this.inputRange[0], Math.min(this.inputRange[1], z)));
    }
    
    // For each basis function (control point)
    for (let m = 0; m < numControlPoints; m++) {
      let sum = 0;
      for (const x of samples) {
        // Compute derivative of basis function using finite differences
        const h = 1e-6;
        const x1 = Math.max(this.inputRange[0], x - h);
        const x2 = Math.min(this.inputRange[1], x + h);
        
        // Get basis function values at x1 and x2
        const span1 = this.findKnotSpan(x1);
        const basis1 = this.computeBasisFunctions(span1, x1);
        const span2 = this.findKnotSpan(x2);
        const basis2 = this.computeBasisFunctions(span2, x2);
        
        const p = this.degree;
        
        // Find basis function value for control point m at x1 and x2
        let basisValue1 = 0;
        for (let j = 0; j <= p; j++) {
          const controlPointIndex = span1 - p + j;
          if (controlPointIndex === m) {
            basisValue1 = basis1[j];
            break;
          }
        }
        
        let basisValue2 = 0;
        for (let j = 0; j <= p; j++) {
          const controlPointIndex = span2 - p + j;
          if (controlPointIndex === m) {
            basisValue2 = basis2[j];
            break;
          }
        }
        
        // Compute derivative
        const derivative = (x2 - x1) > 1e-10 ? (basisValue2 - basisValue1) / (x2 - x1) : 0;
        sum += derivative * derivative;
      }
      expectations.push(sum / numSamples);
    }
    
    return expectations;
  }

  /** Evaluate the B-spline at input x using de Boor's algorithm */
  evaluate(x: number): number {
    // Clamp input to range
    x = Math.max(this.inputRange[0], Math.min(this.inputRange[1], x));
    
    // Find the knot span
    const span = this.findKnotSpan(x);
    
    // Evaluate using de Boor's algorithm
    return this.deBoor(span, x);
  }

  /** Find the knot span index for parameter x */
  private findKnotSpan(x: number): number {
    const n = this.controlPoints.length - 1; // Number of control points - 1
    const p = this.degree;
    
    if (x >= this.knotVector[n + 1]) {
      return n;
    }
    if (x <= this.knotVector[p]) {
      return p;
    }
    
    // Binary search
    let low = p;
    let high = n + 1;
    let mid = Math.floor((low + high) / 2);
    
    while (x < this.knotVector[mid] || x >= this.knotVector[mid + 1]) {
      if (x < this.knotVector[mid]) {
        high = mid;
      } else {
        low = mid;
      }
      mid = Math.floor((low + high) / 2);
    }
    
    return mid;
  }

  /** de Boor's algorithm for B-spline evaluation */
  private deBoor(span: number, x: number): number {
    const p = this.degree;
    
    // Initialize with control points
    let d: number[] = [];
    for (let j = 0; j <= p; j++) {
      d[j] = this.controlPoints[span - p + j];
    }
    
    // Apply de Boor's algorithm
    for (let r = 1; r <= p; r++) {
      for (let j = p; j >= r; j--) {
        const knotLeft = this.knotVector[span - p + j];
        const knotRight = this.knotVector[span + j - r + 1];
        const alpha = (x - knotLeft) / (knotRight - knotLeft);
        d[j] = (1 - alpha) * d[j - 1] + alpha * d[j];
      }
    }
    
    return d[p];
  }

  /** Compute derivative using finite differences */
  derivative(x: number): number {
    const h = 1e-6;
    const x1 = Math.max(this.inputRange[0], x - h);
    const x2 = Math.min(this.inputRange[1], x + h);
    
    if (x2 - x1 < 1e-10) {
      return 0;
    }
    
    return (this.evaluate(x2) - this.evaluate(x1)) / (x2 - x1);
  }

  /** Update control points based on gradients */
  updateParameters(gradients: number[], learningRate: number): void {
    for (let i = 0; i < this.controlPoints.length && i < gradients.length; i++) {
      this.controlPoints[i] -= learningRate * gradients[i];
    }
  }

  /** Get gradients with respect to control points for given input */
  getControlPointGradients(x: number): number[] {
    x = Math.max(this.inputRange[0], Math.min(this.inputRange[1], x));
    
    const span = this.findKnotSpan(x);
    const p = this.degree;
    const gradients: number[] = [];
    
    // Initialize gradients array - Fix: Use loop instead of fill()
    for (let i = 0; i < this.controlPoints.length; i++) {
      gradients.push(0);
    }
    
    // Compute basis functions using de Boor's algorithm
    const basisFunctions = this.computeBasisFunctions(span, x);
    
    // Set gradients for active control points
    for (let j = 0; j <= p; j++) {
      const controlPointIndex = span - p + j;
      if (controlPointIndex >= 0 && controlPointIndex < gradients.length) {
        gradients[controlPointIndex] = basisFunctions[j];
      }
    }
    
    return gradients;
  }

  /** Compute basis functions for given span and parameter */
  private computeBasisFunctions(span: number, x: number): number[] {
    const p = this.degree;
    const basisFunctions = new Array(p + 1);
    
    // Initialize
    let left = new Array(p + 1);
    let right = new Array(p + 1);
    basisFunctions[0] = 1.0;
    
    for (let j = 1; j <= p; j++) {
      left[j] = x - this.knotVector[span + 1 - j];
      right[j] = this.knotVector[span + j] - x;
      let saved = 0.0;
      
      for (let r = 0; r < j; r++) {
        const temp = basisFunctions[r] / (right[r + 1] + left[j - r]);
        basisFunctions[r] = saved + right[r + 1] * temp;
        saved = left[j - r] * temp;
      }
      basisFunctions[j] = saved;
    }
    
    return basisFunctions;
  }
}

/**
 * A KAN edge that connects two nodes with a learnable function
 */
export class KANEdge {
  id: string;
  sourceNode: KANNode;
  destNode: KANNode;
  learnableFunction: LearnableFunction;
  lastInput: number = 0;
  accGradients: number[] = [];
  numAccumulatedGrads: number = 0;
  isActive: boolean = true;
  
  // Histogram tracking for activation visualization
  activationHistogram: number[] = [];
  outputHistogram: number[] = [];
  histogramBins: number = 20;
  histogramRange: [number, number] = [-6, 6];
  outputHistogramRange: [number, number] = [-1, 1];
  histogramDecayFactor: number = 0.995; // Decay old counts: 0.99 = faster decay, 0.999 = slower decay
  outputHistogramDecayFactor: number = 0.95; // Faster decay for outputs to respond quickly to control point changes
  
  // Track observed ranges for adaptive histograms
  private observedInputMin: number = Infinity;
  private observedInputMax: number = -Infinity;
  private observedOutputMin: number = Infinity;
  private observedOutputMax: number = -Infinity;
  private useAdaptiveRanges: boolean = false;

  constructor(
    source: KANNode,
    dest: KANNode,
    gridSize: number = 5,
    degree: number = 3,
    initNoise: number | "xavier" | "linear" = 0.3,
    fanIn: number = 1,
    fanOut: number = 1
  ) {
    this.id = source.id + "-" + dest.id;
    this.sourceNode = source;
    this.destNode = dest;
    this.learnableFunction = new LearnableFunction(
      this.id, gridSize, [-6, 6], degree, initNoise, fanIn, fanOut
    );
    
    const numControlPoints = gridSize + 1;
    this.accGradients = [];
    for (let i = 0; i < numControlPoints; i++) {
      this.accGradients.push(0);
    }
    
    // Initialize histogram
    this.resetHistogram();
  }

  /** Forward pass through the edge */
  forward(input: number, recordHistogram: boolean = true): number {
    this.lastInput = input;
    
    // If edge is deactivated, return 0
    if (!this.isActive) {
      return 0;
    }
    
    if (recordHistogram) {
      this.recordActivation(input);
    }
    const output = this.learnableFunction.evaluate(input);
    if (recordHistogram) {
      this.recordOutput(output);
    }
    return output;
  }
  
  /** Record activation for histogram visualization */
  recordActivation(input: number): void {
    // Apply decay to all bins to fade old data
    for (let i = 0; i < this.histogramBins; i++) {
      this.activationHistogram[i] *= this.histogramDecayFactor;
    }
    
    // Track observed range for adaptive histograms
    this.observedInputMin = Math.min(this.observedInputMin, input);
    this.observedInputMax = Math.max(this.observedInputMax, input);
    
    // Find bin index (no clipping - values outside range go to edge bins)
    const [min, max] = this.histogramRange;
    const binWidth = (max - min) / this.histogramBins;
    const binIndex = Math.floor((input - min) / binWidth);
    const clampedIndex = Math.max(0, Math.min(this.histogramBins - 1, binIndex));
    
    // Increment count
    this.activationHistogram[clampedIndex]++;
  }
  
  /** Record output for histogram visualization */
  recordOutput(output: number): void {
    // Apply faster decay to output bins to quickly respond to control point changes
    for (let i = 0; i < this.histogramBins; i++) {
      this.outputHistogram[i] *= this.outputHistogramDecayFactor;
    }
    
    // Track observed range for adaptive histograms
    this.observedOutputMin = Math.min(this.observedOutputMin, output);
    this.observedOutputMax = Math.max(this.observedOutputMax, output);
    
    // Find bin index (no clipping - values outside range go to edge bins)
    const [min, max] = this.outputHistogramRange;
    const binWidth = (max - min) / this.histogramBins;
    const binIndex = Math.floor((output - min) / binWidth);
    const clampedIndex = Math.max(0, Math.min(this.histogramBins - 1, binIndex));
    
    // Increment count
    this.outputHistogram[clampedIndex]++;
  }
  
  /** Get normalized histogram for visualization */
  getNormalizedHistogram(): number[] {
    const maxCount = Math.max(...this.activationHistogram, 1);
    return this.activationHistogram.map(count => count / maxCount);
  }
  
  /** Get normalized output histogram for visualization */
  getNormalizedOutputHistogram(): number[] {
    const maxCount = Math.max(...this.outputHistogram, 1);
    return this.outputHistogram.map(count => count / maxCount);
  }
  
  /** Helper method to calculate standard deviation from histogram data */
  private calculateStdFromHistogram(histogram: number[], range: [number, number]): number {
    const totalCount = histogram.reduce((sum, count) => sum + count, 0);
    if (totalCount === 0) return 0;
    
    const [min, max] = range;
    const binWidth = (max - min) / this.histogramBins;
    
    // Calculate mean
    let mean = 0;
    for (let i = 0; i < this.histogramBins; i++) {
      const binCenter = min + (i + 0.5) * binWidth;
      mean += binCenter * histogram[i];
    }
    mean /= totalCount;
    
    // Calculate variance
    let variance = 0;
    for (let i = 0; i < this.histogramBins; i++) {
      const binCenter = min + (i + 0.5) * binWidth;
      const diff = binCenter - mean;
      variance += diff * diff * histogram[i];
    }
    variance /= totalCount;
    
    return Math.sqrt(variance);
  }
  
  /** Calculate standard deviation of input activations from histogram */
  getInputActivationStd(): number {
    return this.calculateStdFromHistogram(this.activationHistogram, this.histogramRange);
  }
  
  /** Calculate standard deviation of output activations from histogram */
  getOutputActivationStd(): number {
    return this.calculateStdFromHistogram(this.outputHistogram, this.outputHistogramRange);
  }
  
  /** Get observed input activation range */
  getObservedInputRange(): [number, number] {
    return [this.observedInputMin, this.observedInputMax];
  }
  
  /** Get observed output activation range */
  getObservedOutputRange(): [number, number] {
    return [this.observedOutputMin, this.observedOutputMax];
  }
  
  /** Enable adaptive histogram ranges based on observed values */
  enableAdaptiveRanges(): void {
    this.useAdaptiveRanges = true;
    if (this.observedInputMin < Infinity && this.observedInputMax > -Infinity) {
      const inputPadding = (this.observedInputMax - this.observedInputMin) * 0.1;
      this.histogramRange = [
        this.observedInputMin - inputPadding,
        this.observedInputMax + inputPadding
      ];
    }
    if (this.observedOutputMin < Infinity && this.observedOutputMax > -Infinity) {
      const outputPadding = (this.observedOutputMax - this.observedOutputMin) * 0.1;
      this.outputHistogramRange = [
        this.observedOutputMin - outputPadding,
        this.observedOutputMax + outputPadding
      ];
    }
  }
  
  /** Update output histogram range based on current spline output range */
  updateOutputHistogramRange(): void {
    if (!this.learnableFunction) return;
    
    // Sample the spline to find its actual output range
    let minOutput = Infinity;
    let maxOutput = -Infinity;
    
    for (let i = 0; i <= 100; i++) {
      const x = -6 + (12 * i) / 100;
      const y = this.learnableFunction.evaluate(x);
      minOutput = Math.min(minOutput, y);
      maxOutput = Math.max(maxOutput, y);
    }
    
    // Add padding
    const padding = Math.max(0.5, (maxOutput - minOutput) * 0.2);
    this.outputHistogramRange = [minOutput - padding, maxOutput + padding];
  }
  
  /** Reset histogram */
  resetHistogram(): void {
    this.activationHistogram = [];
    this.outputHistogram = [];
    for (let i = 0; i < this.histogramBins; i++) {
      this.activationHistogram.push(0);
      this.outputHistogram.push(0);
    }
    
    // Reset observed ranges
    this.observedInputMin = Infinity;
    this.observedInputMax = -Infinity;
    this.observedOutputMin = Infinity;
    this.observedOutputMax = -Infinity;
  }

  /** Accumulate gradients for parameter updates */
  accumulateGradients(outputGradient: number): void {
    // Don't accumulate gradients if edge is inactive
    if (!this.isActive) {
      return;
    }
    
    // Get gradients with respect to control points
    const controlPointGradients = this.learnableFunction.getControlPointGradients(this.lastInput);
    
    // Accumulate gradients
    for (let i = 0; i < this.accGradients.length && i < controlPointGradients.length; i++) {
      this.accGradients[i] += outputGradient * controlPointGradients[i];
    }
    
    this.numAccumulatedGrads++;
  }

  /** Update parameters using accumulated gradients */
  updateParameters(learningRate: number): void {
    // Don't update parameters if edge is inactive
    if (!this.isActive) {
      return;
    }
    
    if (this.numAccumulatedGrads > 0) {
      const avgGradients = this.accGradients.map(g => g / this.numAccumulatedGrads);
      this.learnableFunction.updateParameters(avgGradients, learningRate);
      
      // Reset accumulators
      for (let i = 0; i < this.accGradients.length; i++) {
        this.accGradients[i] = 0;
      }
      this.numAccumulatedGrads = 0;
    }
  }
}

/**
 * A node in a Kolmogorov-Arnold Network
 */
export class KANNode {
  id: string;
  /** Input edges */
  inputEdges: KANEdge[] = [];
  /** Output edges */
  outputEdges: KANEdge[] = [];
  /** Cached output value */
  output: number = 0;
  /** Error derivative with respect to this node's output */
  outputDer: number = 0;
  /** Whether this node is active */
  isActive: boolean = true;

  constructor(id: string) {
    this.id = id;
  }

  /** Forward pass: sum all edge outputs */
  forward(recordHistogram: boolean = true): number {
    // If node is deactivated, output is 0
    if (!this.isActive) {
      this.output = 0;
      return 0;
    }
    
    this.output = 0;
    for (const edge of this.inputEdges) {
      this.output += edge.forward(edge.sourceNode.output, recordHistogram);
    }
    return this.output;
  }

  /** Backward pass: distribute gradients to input edges */
  backward(): void {
    // Don't backpropagate if node is inactive
    if (!this.isActive) {
      return;
    }
    
    for (const edge of this.inputEdges) {
      const inputGrad = this.outputDer * edge.learnableFunction.derivative(edge.lastInput);
      edge.accumulateGradients(this.outputDer);
      edge.sourceNode.outputDer += inputGrad;
    }
  }
}

/**
 * Build a Kolmogorov-Arnold Network
 */
export function buildKANNetwork(
  networkShape: number[],
  inputIds: string[],
  gridSize: number = 5,
  degree: number = 3,
  initNoise: number | "xavier" | "linear" = 0.3
): KANNode[][] {
  const numLayers = networkShape.length;
  let nodeId = 1;
  const network: KANNode[][] = [];

  // Create nodes for each layer
  for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    const isInputLayer = layerIdx === 0;
    const currentLayer: KANNode[] = [];
    network.push(currentLayer);
    const numNodes = networkShape[layerIdx];
    for (let i = 0; i < numNodes; i++) {
      const id = isInputLayer ? inputIds[i] : nodeId.toString();
      if (!isInputLayer) nodeId++;
      const node = new KANNode(id);
      currentLayer.push(node);
    }
  }

  // Create edges between layers
  for (let layerIdx = 1; layerIdx < numLayers; layerIdx++) {
    const prevLayer = network[layerIdx - 1];
    const currentLayer = network[layerIdx];
    const fanIn = prevLayer.length;
    const fanOut = (layerIdx < numLayers - 1) ? network[layerIdx + 1].length : 1;

    for (const destNode of currentLayer) {
      for (const sourceNode of prevLayer) {
        const edge = new KANEdge(
          sourceNode, destNode, gridSize, degree, initNoise, fanIn, fanOut
        );
        sourceNode.outputEdges.push(edge);
        destNode.inputEdges.push(edge);
      }
    }
  }

  return network;
}

/**
 * Forward propagation through KAN network
 */
export function kanForwardProp(network: KANNode[][], inputs: number[], recordHistogram: boolean = true): number {
  const inputLayer = network[0];
  if (inputs.length !== inputLayer.length) {
    throw new Error("Number of inputs must match input layer size");
  }

  // Set input layer outputs
  for (let i = 0; i < inputLayer.length; i++) {
    inputLayer[i].output = inputs[i];
  }

  // Forward propagate through remaining layers
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    const currentLayer = network[layerIdx];
    for (const node of currentLayer) {
      node.forward(recordHistogram);
    }
  }

  return network[network.length - 1][0].output;
}

/**
 * Backward propagation through KAN network
 */
export function kanBackProp(
  network: KANNode[][],
  target: number,
  errorFunc: ErrorFunction
): void {
  // Initialize output gradient
  const outputNode = network[network.length - 1][0];
  outputNode.outputDer = errorFunc.der(outputNode.output, target);

  // Backward propagate through layers
  for (let layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
    const currentLayer = network[layerIdx];
    
    // Reset input node gradients for this layer
    if (layerIdx > 1) {
      const prevLayer = network[layerIdx - 1];
      for (const node of prevLayer) {
        node.outputDer = 0;
      }
    }
    
    // Backward pass for current layer
    for (const node of currentLayer) {
      node.backward();
    }
  }
}

/**
 * Update KAN network parameters
 */
export function updateKANWeights(network: KANNode[][], learningRate: number): void {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    const currentLayer = network[layerIdx];
    for (const node of currentLayer) {
      // Update edge parameters
      for (const edge of node.inputEdges) {
        edge.updateParameters(learningRate);
      }
    }
  }
}

/** Get output node from KAN network */
export function getKANOutputNode(network: KANNode[][]): KANNode {
  return network[network.length - 1][0];
}

/** Reset all activation histograms in the network */
export function resetKANHistograms(network: KANNode[][]): void {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    const currentLayer = network[layerIdx];
    for (const node of currentLayer) {
      for (const edge of node.inputEdges) {
        edge.resetHistogram();
      }
    }
  }
}

/** Iterate over all nodes in KAN network */
export function forEachKANNode(
  network: KANNode[][],
  ignoreInputs: boolean,
  accessor: (node: KANNode) => any
): void {
  for (let layerIdx = ignoreInputs ? 1 : 0; layerIdx < network.length; layerIdx++) {
    const currentLayer = network[layerIdx];
    for (const node of currentLayer) {
      accessor(node);
    }
  }
}