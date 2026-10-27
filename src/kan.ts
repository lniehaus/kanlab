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
 * - When using "glorot" initialization, this class implements a basis-agnostic initialization scheme that accounts for B-spline basis function properties.
 * - Each control point is initialized from N(0, σ_m²) where σ_m depends on:
 *   1. The expected squared value of the basis function: E[B_m(x)²]
 *   2. The expected squared value of its derivative: E[B'_m(x)²]
 *   3. The layer dimensions (fanIn, fanOut)
 * - This ensures proper variance preservation in both forward and backward passes.
 */
export class LearnableFunction {
  id: string;
  controlPoints: number[] = [];
  knotVector: number[] = [];
  gridSize: number;
  degree: number;
  initNoise: number | "lecun" | "glorot" | "identity";
  inputRange: [number, number] = [-6, 6];
  private fanIn: number;
  private fanOut: number;

  constructor(
    id: string,
    gridSize: number = 5,
    range: [number, number] = [-6, 6],
    degree: number = 3,
    initNoise: number | "lecun" | "glorot" | "identity" = 0.3,
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

    if (this.initNoise === "identity") {
      // Identity initialization: creates identity function or negative identity function
      // Control points range from -6 to +6
      const limit = 6;

      // 50% chance for positive or negative identity
      const isPositive = Math.random() < 0.5;
      for (let i = 0; i < numControlPoints; i++) {
        const t = i / (numControlPoints - 1); // 0 to 1
        if (isPositive) {
          this.controlPoints.push(-limit + 2 * limit * t); // -6 to +6 (identity)
        } else {
          this.controlPoints.push(limit - 2 * limit * t); // +6 to -6 (negative identity)
        }
      }
      return;
    }

    if (this.initNoise === "lecun") {
      // LeCun-inspired initialization for KANs (Rigas et al., 2025)
      // Preserves forward-pass variance: σ = sqrt(Var(x) / (n_in · D · μ_B^(0)))
      // where μ_B^(0) is the averaged E[B_m(x)^2] over all basis functions,
      // D = number of control points, and Var(x) = (b-a)^2/12 for U(a,b).
      const safeFanIn = Math.max(1, this.fanIn | 0);
      const D = numControlPoints;
      const [a, b] = this.inputRange;
      const varX = (b - a) * (b - a) / 12; // Var of U(a, b)

      const { mu0Avg } = this.computeUniformBasisExpectationsAveraged();

      const denom = safeFanIn * D * mu0Avg;
      const sigma = denom > 0 ?
        Math.sqrt(varX / denom) :
        Math.sqrt(1.0 / (safeFanIn * D)); // Fallback

      for (let i = 0; i < numControlPoints; i++) {
        const u1 = Math.random();
        const u2 = Math.random();
        const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        this.controlPoints.push(z * sigma);
      }
      return;
    }

    if (this.initNoise === "glorot") {
      // Glorot-inspired initialization for KANs (Rigas et al., 2025)
      // Balances forward- and backward-pass variance:
      // σ = sqrt(1/D · 2 / (n_in · μ_B^(0) + n_out · μ_B^(1)))
      // where μ_B^(0) and μ_B^(1) are the averaged expectations over all
      // basis functions of B_m(x)^2 and B'_m(x)^2, respectively.
      const safeFanIn = Math.max(1, this.fanIn | 0);
      const safeFanOut = Math.max(1, this.fanOut | 0);
      const D = numControlPoints;

      const { mu0Avg, mu1Avg } = this.computeUniformBasisExpectationsAveraged();

      const denom = safeFanIn * mu0Avg + safeFanOut * mu1Avg;
      const sigma = denom > 0 ?
        Math.sqrt((1.0 / D) * (2.0 / denom)) :
        Math.sqrt(2.0 / (safeFanIn + safeFanOut)); // Fallback to standard Glorot

      for (let i = 0; i < numControlPoints; i++) {
        const u1 = Math.random();
        const u2 = Math.random();
        const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        this.controlPoints.push(z * sigma);
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
   * Compute averaged E[B_m(x)^2] and E[B'_m(x)^2] over all basis functions,
   * using Monte Carlo sampling with x ~ U(lo, hi).
   * From Rigas et al. (2025), "Initialization Schemes for KANs".
   *
   * Goal: estimate two scalars needed by the LeCun / Glorot formulas:
   *   μ_B^(0) = (1/D) Σ_m  E_x[ B_m(x)^2 ]   (forward-pass moment)
   *   μ_B^(1) = (1/D) Σ_m  E_x[ B'_m(x)^2 ]   (backward-pass moment)
   *
   * Returns { mu0Avg, mu1Avg } — the two scalar averages.
   */
  private computeUniformBasisExpectationsAveraged(): { mu0Avg: number; mu1Avg: number } {
    // D = number of B-spline basis functions (= gridSize + 1 control points)
    const numControlPoints = this.gridSize + 1;
    // Number of Monte Carlo samples drawn from U(lo, hi)
    const numSamples = 10000;
    // [lo, hi] = input domain of the spline (maps to [-1,1] in the paper)
    const [lo, hi] = this.inputRange;
    // p = polynomial degree of the B-spline
    const p = this.degree;

    // mu0Sums[m] accumulates Σ_s B_m(x_s)^2  (unnormalized E[B_m^2])
    // mu1Sums[m] accumulates Σ_s B'_m(x_s)^2  (unnormalized E[B'_m^2])
    const mu0Sums: number[] = [];
    const mu1Sums: number[] = [];
    for (let i = 0; i < numControlPoints; i++) {
      mu0Sums.push(0);
      mu1Sums.push(0);
    }

    for (let s = 0; s < numSamples; s++) {
      // Draw x uniformly from the spline domain
      const x = lo + (hi - lo) * Math.random();

      // ── Forward moment: accumulate B_m(x)^2 ──
      // Only (p+1) basis functions are nonzero at any x; they sit at
      // control-point indices [span-p .. span].
      const span = this.findKnotSpan(x);
      const basisValues = this.computeBasisFunctions(span, x);
      for (let j = 0; j <= p; j++) {
        const cpIdx = span - p + j;  // control-point index for this basis fn
        if (cpIdx >= 0 && cpIdx < numControlPoints) {
          mu0Sums[cpIdx] += basisValues[j] * basisValues[j];
        }
      }

      // ── Backward moment: accumulate B'_m(x)^2 via central differences ──
      const h = 1e-6;  // finite-difference step
      const x1 = Math.max(lo, x - h);
      const x2 = Math.min(hi, x + h);
      const span1 = this.findKnotSpan(x1);
      const basis1 = this.computeBasisFunctions(span1, x1);
      const span2 = this.findKnotSpan(x2);
      const basis2 = this.computeBasisFunctions(span2, x2);
      const dx = x2 - x1;  // actual step width (clipped at boundaries)

      // Expand the (p+1)-length local basis vectors into full D-length
      // arrays so that we can pair B_m(x1) with B_m(x2) for each m,
      // even when x1 and x2 straddle different knot spans.
      const bv1: number[] = [];
      const bv2: number[] = [];
      for (let i = 0; i < numControlPoints; i++) {
        bv1.push(0);
        bv2.push(0);
      }
      for (let j = 0; j <= p; j++) {
        const idx1 = span1 - p + j;
        if (idx1 >= 0 && idx1 < numControlPoints) bv1[idx1] = basis1[j];
        const idx2 = span2 - p + j;
        if (idx2 >= 0 && idx2 < numControlPoints) bv2[idx2] = basis2[j];
      }
      // Approximate B'_m(x) ≈ (B_m(x+h) - B_m(x-h)) / (2h)
      if (dx > 1e-10) {
        for (let m = 0; m < numControlPoints; m++) {
          const deriv = (bv2[m] - bv1[m]) / dx;
          mu1Sums[m] += deriv * deriv;
        }
      }
    }

    // ── Aggregate: first per-basis expectation, then average over bases ──
    // mu0Sums[m] / numSamples  ≈  E_x[ B_m(x)^2 ]      for each m
    // mu0Total = Σ_m E_x[ B_m(x)^2 ]
    // mu0Avg   = mu0Total / D  =  μ_B^(0)               (scalar used in init formulas)
    let mu0Total = 0;
    let mu1Total = 0;
    for (let m = 0; m < numControlPoints; m++) {
      mu0Total += mu0Sums[m] / numSamples;
      mu1Total += mu1Sums[m] / numSamples;
    }
    const mu0Avg = mu0Total / numControlPoints;  // μ_B^(0)
    const mu1Avg = mu1Total / numControlPoints;  // μ_B^(1)

    return { mu0Avg, mu1Avg };
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
  histogramBins: number = 20;
  histogramRange: [number, number] = [-6, 6];
  histogramDecayFactor: number = 0.995; // Decay old counts: 0.99 = faster decay, 0.999 = slower decay

  // Track observed ranges for adaptive histograms
  private observedInputMin: number = Infinity;
  private observedInputMax: number = -Infinity;
  private useAdaptiveRanges: boolean = false;

  constructor(
    source: KANNode,
    dest: KANNode,
    gridSize: number = 5,
    degree: number = 3,
    initNoise: number | "identity" | "lecun" | "glorot" = 0.3,
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
    return this.learnableFunction.evaluate(input);
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

  /** Get normalized histogram for visualization */
  getNormalizedHistogram(): number[] {
    const maxCount = Math.max(...this.activationHistogram, 1);
    return this.activationHistogram.map(count => count / maxCount);
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

  /**
   * Calculate standard deviation of output activations by pushing the input
   * histogram through the current spline, so it always reflects the current function
   */
  getOutputActivationStd(): number {
    const totalCount = this.activationHistogram.reduce((sum, count) => sum + count, 0);
    if (totalCount === 0) return 0;

    const [min, max] = this.histogramRange;
    const binWidth = (max - min) / this.histogramBins;
    const samplesPerBin = 10;

    // Accumulate weighted sums of f(x) and f(x)^2 over samples within each input bin
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < this.histogramBins; i++) {
      const count = this.activationHistogram[i];
      if (count === 0) continue;
      const weight = count / samplesPerBin;
      for (let s = 0; s < samplesPerBin; s++) {
        const x = min + (i + (s + 0.5) / samplesPerBin) * binWidth;
        const y = this.learnableFunction.evaluate(x);
        sum += weight * y;
        sumSq += weight * y * y;
      }
    }

    const mean = sum / totalCount;
    const variance = Math.max(0, sumSq / totalCount - mean * mean);
    return Math.sqrt(variance);
  }

  /** Get observed input activation range */
  getObservedInputRange(): [number, number] {
    return [this.observedInputMin, this.observedInputMax];
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
  }

  /** Reset histogram */
  resetHistogram(): void {
    this.activationHistogram = [];
    for (let i = 0; i < this.histogramBins; i++) {
      this.activationHistogram.push(0);
    }

    // Reset observed ranges
    this.observedInputMin = Infinity;
    this.observedInputMax = -Infinity;
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
  initNoise: number | "identity" | "lecun" | "glorot" = 0.3
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