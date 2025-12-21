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

import * as dataset from "./dataset";

/** Suffix added to the state when storing if a control is hidden or not. */
const HIDE_STATE_SUFFIX = "_hide";

/** A map between dataset names and functions that generate classification data. */
export let datasets: {[key: string]: dataset.DataGenerator} = {
  "circle": dataset.classifyCircleData,
  "xor": dataset.classifyXORData,
  "gauss": dataset.classifyTwoGaussData,
  "spiral": dataset.classifySpiralData,
};

/** A map between dataset names and functions that generate regression data. */
export let regDatasets: {[key: string]: dataset.DataGenerator} = {
  "reg-plane": dataset.regressPlane,
  "reg-gauss": dataset.regressGaussian
};

export function getKeyFromValue(obj: any, value: any): string {
  for (let key in obj) {
    if (obj[key] === value) {
      return key;
    }
  }
  return undefined;
}

function endsWith(s: string, suffix: string): boolean {
  return s.substr(-suffix.length) === suffix;
}

function getHideProps(obj: any): string[] {
  let result: string[] = [];
  for (let prop in obj) {
    if (endsWith(prop, HIDE_STATE_SUFFIX)) {
      result.push(prop);
    }
  }
  return result;
}

/**
 * The data type of a state variable. Used for determining the
 * (de)serialization method.
 */
export enum Type {
  STRING,
  NUMBER,
  ARRAY_NUMBER,
  ARRAY_STRING,
  BOOLEAN,
  OBJECT
}

export enum Problem {
  CLASSIFICATION,
  REGRESSION
}

export let problems = {
  "classification": Problem.CLASSIFICATION,
  "regression": Problem.REGRESSION
};

export interface Property {
  name: string;
  type: Type;
  keyMap?: {[key: string]: any};
};

// Add the GUI state.
export class State {

  private static PROPS: Property[] = [
    {name: "batchSize", type: Type.NUMBER},
    {name: "dataset", type: Type.OBJECT, keyMap: datasets},
    {name: "regDataset", type: Type.OBJECT, keyMap: regDatasets},
    {name: "learningRate", type: Type.NUMBER},
    {name: "noise", type: Type.NUMBER},
    {name: "networkShape", type: Type.ARRAY_NUMBER},
    {name: "seed", type: Type.STRING},
    {name: "showTestData", type: Type.BOOLEAN},
    {name: "discretize", type: Type.BOOLEAN},
    {name: "percTrainData", type: Type.NUMBER},
    {name: "x", type: Type.BOOLEAN},
    {name: "y", type: Type.BOOLEAN},
    {name: "xTimesY", type: Type.BOOLEAN},
    {name: "xSquared", type: Type.BOOLEAN},
    {name: "ySquared", type: Type.BOOLEAN},
    {name: "cosX", type: Type.BOOLEAN},
    {name: "sinX", type: Type.BOOLEAN},
    {name: "cosY", type: Type.BOOLEAN},
    {name: "sinY", type: Type.BOOLEAN},
    {name: "collectStats", type: Type.BOOLEAN},
    {name: "tutorial", type: Type.STRING},
    {name: "problem", type: Type.OBJECT, keyMap: problems},
    {name: "initZero", type: Type.BOOLEAN},
    {name: "hideText", type: Type.BOOLEAN},
    {name: "numControlPoints", type: Type.NUMBER}, // KAN number of control points parameter
    {name: "degree", type: Type.NUMBER},   // KAN B-spline degree parameter
    {name: "initNoise", type: Type.STRING}, // KAN initial control point noise parameter
  ];

  [key: string]: any;
  learningRate = 0.03;
  showTestData = false;
  noise = 0;
  batchSize = 10;
  discretize = false;
  tutorial: string = null;
  percTrainData = 50;
  problem = Problem.CLASSIFICATION;
  initZero = false;
  hideText = false;
  collectStats = false;
  numHiddenLayers = 1;
  hiddenLayerControls: any[] = [];
  networkShape: number[] = [1];
  numControlPoints = 5; // KAN number of control points for spline functions
  degree = 3;   // KAN B-spline degree (1=linear, 3=cubic, etc.)
  initNoise: number | "xavier" | "linear" = 0.3; // Allow strategies
  x = true;
  y = true;
  xTimesY = false;
  xSquared = false;
  ySquared = false;
  cosX = false;
  sinX = false;
  cosY = false;
  sinY = false;
  dataset: dataset.DataGenerator = dataset.classifyCircleData;
  regDataset: dataset.DataGenerator = dataset.regressPlane;
  seed: string;

  /**
   * Deserializes the state from the url hash.
   */
  static deserializeState(): State {
    let map: {[key: string]: string} = {};
    for (let keyvalue of window.location.hash.slice(1).split("&")) {
      let [name, value] = keyvalue.split("=");
      map[name] = value;
    }
    const state = new State();

    function hasKey(name: string): boolean {
      return name in map && map[name] != null && map[name].trim() !== "";
    }

    function parseArray(value: string): string[] {
      return value.trim() === "" ? [] : value.split(",");
    }

    // Deserialize regular properties.
    State.PROPS.forEach(({name, type, keyMap}) => {
      switch (type) {
        case Type.OBJECT:
          if (keyMap == null) {
            throw Error("A key-value map must be provided for state " +
                "variables of type Object");
          }
          if (hasKey(name) && map[name] in keyMap) {
            state[name] = keyMap[map[name]];
          }
          break;
        case Type.NUMBER:
          if (hasKey(name)) {
            // The + operator is for converting a string to a number.
            state[name] = +map[name];
          }
          break;
        case Type.STRING:
          if (hasKey(name)) {
            state[name] = map[name];
          }
          break;
        case Type.BOOLEAN:
          if (hasKey(name)) {
            state[name] = (map[name] === "false" ? false : true);
          }
          break;
        case Type.ARRAY_NUMBER:
          if (name in map) {
            state[name] = parseArray(map[name]).map(Number);
          }
          break;
        case Type.ARRAY_STRING:
          if (name in map) {
            state[name] = parseArray(map[name]);
          }
          break;
        default:
          throw Error("Encountered an unknown type for a state variable");
      }
    });

    // Deserialize state properties that correspond to hiding UI controls.
    getHideProps(map).forEach(prop => {
      state[prop] = (map[prop] === "true") ? true : false;
    });
    state.numHiddenLayers = state.networkShape.length;
    if (state.seed == null) {
      state.seed = Math.random().toFixed(5);
    }
    Math.seedrandom(state.seed);

    // Coercion helpers
    const toNum = (v: any, fallback: number): number => {
      const n = typeof v === "string" ? parseFloat(v) :
                typeof v === "number" ? v : NaN;
      return isNaN(n) ? fallback : n;
    };
    const toInt = (v: any, fallback: number): number =>
      Math.floor(toNum(v, fallback));

    // Coerce numeric fields
    state.learningRate = toNum((state as any).learningRate, 0.03);
    state.batchSize = Math.max(1, toInt((state as any).batchSize, 10));
    state.percTrainData = Math.min(100, Math.max(0, toNum((state as any).percTrainData, 50)));
    state.noise = Math.max(0, toNum((state as any).noise, 0));
    state.degree = Math.max(1, toInt((state as any).degree, 3));
    state.numControlPoints = Math.max(state.degree + 1, toInt((state as any).numControlPoints, 6));

    // Init noise can be strategy or number
    const rawInit = (state as any).initNoise;
    if (rawInit === "xavier" || rawInit === "kaiming" || rawInit === "lecun") {
      state.initNoise = rawInit;
    } else {
      const n = typeof rawInit === "string" ? parseFloat(rawInit) :
                typeof rawInit === "number" ? rawInit : NaN;
      state.initNoise = isNaN(n) ? 0.3 : n;
    }

    return state;
  }

  /**
   * Serializes the state into the url hash.
   */
  serialize() {
    // Serialize regular properties.
    let props: string[] = [];
    State.PROPS.forEach(({name, type, keyMap}) => {
      let value = this[name];
      // Don't serialize missing values.
      if (value == null) {
        return;
      }
      if (type === Type.OBJECT) {
        value = getKeyFromValue(keyMap, value);
      } else if (type === Type.ARRAY_NUMBER ||
          type === Type.ARRAY_STRING) {
        value = value.join(",");
      }
      props.push(`${name}=${value}`);
    });
    // Serialize properties that correspond to hiding UI controls.
    getHideProps(this).forEach(prop => {
      props.push(`${prop}=${this[prop]}`);
    });
    window.location.hash = props.join("&");
  }

  /** Returns all the hidden properties. */
  getHiddenProps(): string[] {
    let result: string[] = [];
    for (let prop in this) {
      if (endsWith(prop, HIDE_STATE_SUFFIX) && String(this[prop]) === "true") {
        result.push(prop.replace(HIDE_STATE_SUFFIX, ""));
      }
    }
    return result;
  }

  setHideProperty(name: string, hidden: boolean) {
    this[name + HIDE_STATE_SUFFIX] = hidden;
  }
}

// If you have a ParamSpec/PROP_SPECS list, keep initNoise as STRING:
//   {name: "batchSize", type: Type.NUMBER},
//   {name: "dataset", type: Type.OBJECT, keyMap: datasets},
//   {name: "regDataset", type: Type.OBJECT, keyMap: regDatasets},
//   {name: "learningRate", type: Type.NUMBER},
//   {name: "noise", type: Type.NUMBER},
//   {name: "networkShape", type: Type.ARRAY_NUMBER},
//   {name: "seed", type: Type.STRING},
//   {name: "showTestData", type: Type.BOOLEAN},
//   {name: "discretize", type: Type.BOOLEAN},
//   {name: "percTrainData", type: Type.NUMBER},
//   {name: "x", type: Type.BOOLEAN},
//   {name: "y", type: Type.BOOLEAN},
//   {name: "xTimesY", type: Type.BOOLEAN},
//   {name: "xSquared", type: Type.BOOLEAN},
//   {name: "ySquared", type: Type.BOOLEAN},
//   {name: "cosX", type: Type.BOOLEAN},
//   {name: "sinX", type: Type.BOOLEAN},
//   {name: "cosY", type: Type.BOOLEAN},
//   {name: "sinY", type: Type.BOOLEAN},
//   {name: "collectStats", type: Type.BOOLEAN},
//   {name: "tutorial", type: Type.STRING},
//   {name: "problem", type: Type.OBJECT, keyMap: problems},
//   {name: "initZero", type: Type.BOOLEAN},
//   {name: "hideText", type: Type.BOOLEAN},
//   {name: "numControlPoints", type: Type.NUMBER}, // KAN number of control points parameter
//   {name: "degree", type: Type.NUMBER},   // KAN B-spline degree parameter
//   {name: "initNoise", type: Type.STRING}, // KAN initial control point noise parameter
