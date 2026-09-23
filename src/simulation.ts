import type { Circuit } from "./data.ts";

export type Experiment = "threat" | "motion" | "light" | "lesion";
export type Lesion = "none" | "visual" | "descending" | "motor";
export interface Parameters {
  experiment: Experiment;
  direction: number;
  speed: number;
  size: number;
  intensity: number;
  lesion: Lesion;
}
export interface Result {
  onsets: Float32Array;
  parents: Int32Array;
  motorAt: number;
  duration: number;
  stimulusAt: number;
  path: number[];
  blocked: number[];
}
export const CONDUCTION = 0.85; // Presentation/model seconds, not biological latency.

export function simulate(circuit: Circuit, parameters: Parameters): Result {
  const { neurons, edges } = circuit;
  const n = neurons.length;
  if (
    !n ||
    n > 500 ||
    !edges.every(
      ([a, b, weight]) =>
        Number.isInteger(a) &&
        Number.isInteger(b) &&
        a >= 0 &&
        b >= 0 &&
        a < n &&
        b < n &&
        Number.isFinite(weight) &&
        weight > 0,
    )
  ) {
    throw new Error("Invalid circuit graph");
  }
  if (
    !["threat", "motion", "light", "lesion"].includes(parameters.experiment) ||
    !["none", "visual", "descending", "motor"].includes(parameters.lesion) ||
    ![
      parameters.direction,
      parameters.speed,
      parameters.size,
      parameters.intensity,
    ].every(Number.isFinite)
  ) {
    throw new Error("Invalid experiment parameters");
  }
  const speed = Math.max(0.25, Math.min(3, parameters.speed));
  const intensity = Math.max(0, Math.min(1, parameters.intensity));
  const size = Math.max(0.2, Math.min(2, parameters.size));
  const direction = Math.max(-90, Math.min(90, parameters.direction));
  const onsets = new Float32Array(n).fill(Infinity);
  const parents = new Int32Array(n).fill(-1);
  const visited = new Uint8Array(n);
  const blocked = neurons.flatMap((neuron, index) => {
    const disabled =
      parameters.lesion === "descending"
        ? neuron.type === "DNp01"
        : parameters.lesion === "motor"
          ? neuron.type === "TTMn"
          : parameters.lesion === "visual" &&
            ["LC4", "LPLC2"].includes(neuron.type);
    return disabled ? [index] : [];
  });
  const disabled = new Set(blocked);
  const stimulusAt = 1.3 / speed;
  for (let i = 0; i < n; i++) {
    const neuron = neurons[i];
    if (disabled.has(i) || !["LC4", "LPLC2"].includes(neuron.type)) continue;
    const lateral = neuron.side === "L" ? -1 : 1;
    const sideGain = 1 - Math.max(0, (-lateral * direction) / 90) * 0.8;
    const typeGain =
      parameters.experiment === "motion" && neuron.type === "LPLC2" ? 0.35 : 1;
    const drive = intensity * sideGain * typeGain * Math.min(1, size);
    if (drive >= 0.18) onsets[i] = stimulusAt + (1 - drive) * 0.5;
  }
  // ponytail: O(n²) shortest-path scan is bounded to 500 circuit neurons; use a heap for larger graphs.
  for (let iteration = 0; iteration < n; iteration++) {
    let current = -1;
    for (let i = 0; i < n; i++) {
      if (
        !visited[i] &&
        !disabled.has(i) &&
        Number.isFinite(onsets[i]) &&
        (current < 0 || onsets[i] < onsets[current])
      )
        current = i;
    }
    if (current < 0) break;
    visited[current] = 1;
    for (const [from, to, weight] of edges) {
      if (from !== current || disabled.has(to) || weight < 5) continue;
      const arrival =
        onsets[current] + CONDUCTION + 0.16 + 0.5 / Math.sqrt(weight);
      if (arrival < onsets[to]) {
        onsets[to] = arrival;
        parents[to] = current;
      }
    }
  }
  let motor = -1;
  for (let i = 0; i < n; i++) {
    if (
      neurons[i].type === "TTMn" &&
      Number.isFinite(onsets[i]) &&
      (motor < 0 || onsets[i] < onsets[motor])
    )
      motor = i;
  }
  const motorAt = motor < 0 ? Infinity : onsets[motor] + CONDUCTION;
  const path = [];
  let cursor = motor;
  while (cursor >= 0 && path.length < n) {
    path.unshift(cursor);
    cursor = parents[cursor];
  }
  const latest = onsets.reduce(
    (max, v) => (Number.isFinite(v) ? Math.max(max, v) : max),
    stimulusAt,
  );
  return {
    onsets,
    parents,
    motorAt,
    duration: Math.max(6, latest + CONDUCTION + 2),
    stimulusAt,
    path,
    blocked,
  };
}
