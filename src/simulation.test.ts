import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { decode, type Circuit } from "./data.ts";
import {
  simulate,
  escapePose,
  CONDUCTION,
  type Parameters,
} from "./simulation.ts";

const circuit: Circuit = JSON.parse(
  readFileSync(new URL("../public/brain/escape.json", import.meta.url), "utf8"),
);
const parameters: Parameters = {
  experiment: "threat",
  direction: 0,
  speed: 1,
  size: 1,
  intensity: 0.8,
  lesion: "none",
};

test("the fly stays grounded before output, leaps, and lands; scrubbing is deterministic", () => {
  const rest = escapePose(0, 3);
  assert.equal(rest.height, 0);
  assert.deepEqual(escapePose(3, 3), rest);
  assert.deepEqual(escapePose(100, Infinity), rest);
  const push = escapePose(3.06, 3);
  assert.ok(push.push > 0.99);
  assert.equal(push.height, 0);
  assert.equal(push.wing, 0);
  assert.equal(push.fold, 0);
  assert.ok(escapePose(3.7, 3).height > 0.5);
  assert.ok(Math.abs(escapePose(3.795, 3).height - 0.55) < 1e-12);
  assert.ok(Math.abs(escapePose(3.4575, 3).travel - 0.25) < 1e-12);
  assert.equal(escapePose(4.44, 3).fold, 0);
  const settling = escapePose(4.57, 3);
  assert.ok(settling.settle > 0.99);
  assert.equal(settling.height, 0);
  assert.equal(settling.wing, 0);
  const landed = escapePose(6, 3);
  assert.deepEqual(landed, { ...rest, travel: 1 });
  for (const boundary of [3, 3.12, 4.47, 4.67]) {
    const before = escapePose(boundary - 1e-7, 3);
    const after = escapePose(boundary + 1e-7, 3);
    for (const key of Object.keys(before) as (keyof typeof before)[])
      assert.ok(
        Math.abs(after[key] - before[key]) < 1e-5,
        `${key} discontinuity at ${boundary}`,
      );
  }
  const samples = Array.from({ length: 501 }, (_, i) => escapePose(i / 100, 3));
  for (const pose of samples) {
    assert.ok(Object.values(pose).every(Number.isFinite));
    assert.ok(pose.height >= 0 && pose.height <= 0.55);
    assert.ok(pose.travel >= 0 && pose.travel <= 1);
  }
  for (let i = samples.length - 1; i >= 0; i--)
    assert.deepEqual(escapePose(i / 100, 3), samples[i]);
  assert.deepEqual(escapePose(0, 3), rest);
});

test("real circuit drives motor output; lesions block it; inputs remain deterministic", () => {
  const normal = simulate(circuit, parameters);
  assert.ok(Number.isFinite(normal.motorAt));
  assert.ok(normal.path.length >= 3);
  assert.equal(circuit.neurons[normal.path.at(-1)!].type, "TTMn");
  for (let i = 1; i < normal.path.length; i++) {
    const previous = normal.path[i - 1],
      current = normal.path[i];
    assert.ok(circuit.edges.some(([a, b]) => a === previous && b === current));
    assert.ok(normal.onsets[current] >= normal.onsets[previous] + CONDUCTION);
  }
  assert.equal(normal.motorAt, normal.onsets[normal.path.at(-1)!] + CONDUCTION);
  for (const lesion of ["visual", "descending", "motor"] as const) {
    assert.equal(
      simulate(circuit, { ...parameters, lesion }).motorAt,
      Infinity,
    );
  }
  assert.equal(
    simulate(circuit, { ...parameters, intensity: 0 }).motorAt,
    Infinity,
  );
  assert.deepEqual(simulate(circuit, parameters).onsets, normal.onsets);
  assert.throws(() => simulate(circuit, { ...parameters, intensity: NaN }));
  const left = simulate(circuit, { ...parameters, direction: -90 });
  const right = simulate(circuit, { ...parameters, direction: 90 });
  assert.notDeepEqual(left.onsets, right.onsets);
});

test("binary exports have valid coordinates, lengths and circuit owners", () => {
  for (const filename of [
    "overview.bin",
    "background.bin",
    "escape.bin",
    "motion.bin",
    "light.bin",
    "region-0.bin",
    "region-1.bin",
    "region-2.bin",
  ]) {
    const file = gunzipSync(
      readFileSync(new URL(`../public/brain/${filename}.gz`, import.meta.url)),
    );
    const buffer = file.buffer.slice(
      file.byteOffset,
      file.byteOffset + file.byteLength,
    ) as ArrayBuffer;
    const data = decode(buffer);
    assert.ok(data.count > 0);
    if (filename === "escape.bin")
      assert.ok(data.owners.every((owner) => owner < circuit.neurons.length));
    assert.throws(() => decode(buffer.slice(0, -4)));
  }
});

test("different stimuli use distinct measured circuits and light has no invented motor response", () => {
  const ids = new Set(circuit.neurons.map((neuron) => neuron.id));
  for (const kind of ["motion", "light"] as const) {
    const graph: Circuit = JSON.parse(
      readFileSync(
        new URL(`../public/brain/${kind}.json`, import.meta.url),
        "utf8",
      ),
    );
    assert.equal(graph.kind, kind);
    assert.ok(graph.neurons.every((neuron) => !ids.has(neuron.id)));
    graph.neurons.forEach((neuron) => ids.add(neuron.id));
    assert.throws(
      () => simulate(circuit, { ...parameters, experiment: kind }),
      /own verified circuit/,
    );
    const response = simulate(graph, { ...parameters, experiment: kind });
    assert.ok(Number.isFinite(response.responseAt));
    assert.equal(response.action, kind === "motion" ? "turn" : "none");
    if (kind === "light") {
      assert.equal(response.motorAt, Infinity);
      assert.ok(
        graph.neurons
          .filter((neuron) => neuron.type === "L1")
          .every((neuron) => neuron.polarity === -1),
      );
      assert.ok(
        graph.neurons
          .filter((neuron) => neuron.type !== "L1")
          .every((neuron) => neuron.polarity === 1),
      );
    } else assert.equal(response.motorAt, response.responseAt);
    for (let i = 1; i < response.path.length; i++) {
      assert.ok(
        graph.edges.some(
          ([from, to]) =>
            from === response.path[i - 1] && to === response.path[i],
        ),
      );
    }
    const silent = simulate(graph, {
      ...parameters,
      experiment: kind,
      intensity: 0,
    });
    assert.equal(silent.responseAt, Infinity);
  }
});
