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
  responseAt: number;
  action: Circuit["action"];
  duration: number;
  stimulusAt: number;
  path: number[];
  blocked: number[];
}
export const CONDUCTION = 0.85; // Presentation/model seconds, not biological latency.

/** An illustrative leap, evaluated from the shared clock so scrubbing is reversible. */
export function escapePose(time: number, motorAt: number) {
  const elapsed =
    Number.isFinite(time) && Number.isFinite(motorAt)
      ? Math.max(0, time - motorAt)
      : 0;
  const clamp = (value: number) => Math.max(0, Math.min(1, value));
  const smooth = (value: number) => {
    const x = clamp(value);
    return x * x * (3 - 2 * x);
  };
  const loaded = smooth(elapsed / 0.12);
  const u = clamp((elapsed - 0.12) / 1.35);
  const lift = 4 * u * (1 - u);
  const recovery = smooth((elapsed - 1.47) / 0.2);
  const wingOpen = smooth((u - 0.035) / 0.1) * (1 - smooth((u - 0.84) / 0.16));
  return {
    travel: u,
    height: lift * 0.55,
    fold: smooth((u - 0.04) / 0.15) * (1 - smooth((u - 0.72) / 0.24)),
    pitch: -lift * 0.22,
    // Display-slow wing beats in model seconds, not a biological wingbeat frequency.
    wing: wingOpen * (0.35 + Math.sin((elapsed - 0.12) * 48) * 0.22),
    push: 4 * loaded * (1 - loaded),
    settle: 4 * recovery * (1 - recovery),
    airborne: smooth(u / 0.045) * (1 - smooth((u - 0.94) / 0.06)),
  };
}

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
  const expectedKind =
    parameters.experiment === "motion" || parameters.experiment === "light"
      ? parameters.experiment
      : "escape";
  if (
    circuit.kind !== expectedKind ||
    !circuit.inputTypes?.length ||
    !circuit.outputTypes?.length ||
    !neurons.every((neuron) => neuron.polarity === 1 || neuron.polarity === -1)
  )
    throw new Error(
      `The ${expectedKind} experiment requires its own verified circuit`,
    );
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
    if (disabled.has(i) || !circuit.inputTypes.includes(neuron.type)) continue;
    const lateral = neuron.side === "L" ? -1 : 1;
    const sideGain = 1 - Math.max(0, (-lateral * direction) / 90) * 0.8;
    const drive = intensity * sideGain * Math.min(1, size);
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
      circuit.outputTypes.includes(neurons[i].type) &&
      Number.isFinite(onsets[i]) &&
      (motor < 0 || onsets[i] < onsets[motor])
    )
      motor = i;
  }
  const responseAt = motor < 0 ? Infinity : onsets[motor] + CONDUCTION;
  const motorAt = circuit.action === "none" ? Infinity : responseAt;
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
    responseAt,
    action: circuit.action,
    duration: Math.max(6, latest + CONDUCTION + 2),
    stimulusAt,
    path,
    blocked,
  };
}
