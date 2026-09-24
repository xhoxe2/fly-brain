import assert from "node:assert/strict";
import test from "node:test";
import { FingerStimulus, fingerTarget, type FingerEnvironment } from "./finger.ts";
import { FeedingWorld } from "./embodiment.ts";
import { SpikingBrain } from "./neural.ts";
import type { FeedingProtocol, FingerProtocol } from "./neural-data.ts";

// Artificial identities and an empty graph exercise adapter contracts only.
// The full anatomical graph is exercised separately by tools/check-finger.ts.
const protocol: FingerProtocol = {
  inputs: {
    lc4Left: { ids: ["fixture-0"], indices: [0] },
    lc4Right: { ids: ["fixture-1"], indices: [1] },
    lplc2Left: { ids: ["fixture-2"], indices: [2] },
    lplc2Right: { ids: ["fixture-3"], indices: [3] },
  },
  outputs: {
    gfLeft: { id: "fixture-4", index: 4, label: "fixture GF left" },
    gfRight: { id: "fixture-5", index: 5, label: "fixture GF right" },
  },
};
const feeding: FeedingProtocol = {
  inputs: {
    sugar: { ids: ["fixture-6"], indices: [6] },
    bitter: { ids: ["fixture-7"], indices: [7] },
  },
  outputs: {
    primary: { id: "fixture-8", index: 8, label: "fixture MN9 left" },
    additional: [{ id: "fixture-9", index: 9, label: "fixture MN9 right" }],
  },
};
const far: FingerEnvironment = { enabled: true, lateral: -1, proximity: 0 };
const near: FingerEnvironment = { ...far, proximity: 1 };
const zeroRates = [0, 0, 0, 0];

test("approach drive requires positive optical expansion; insertion, rest, withdrawal and disabled input do not drive vision", () => {
  const stimulus = new FingerStimulus(protocol, 10);
  assert.deepEqual(stimulus.advance(far, 10), []);
  assert.deepEqual(stimulus.state.sensoryRates, zeroRates);
  const before = stimulus.state.position!;
  const drives = stimulus.advance(near, 10);
  assert.ok(drives.length > 0);
  assert.ok(stimulus.state.sensoryRates[0] > 0);
  assert.ok(stimulus.state.sensoryRates[2] > 0);
  assert.ok(Math.hypot(...stimulus.state.position!.map((v, i) => v - before[i])) <= 0.015 + 1e-12);
  assert.ok(drives.every((drive) => drive.indices.every((i) => i < 4)), "No direct GF/motor injection");
  for (let i = 0; i < 200; i++) stimulus.advance(near, 10);
  assert.deepEqual(stimulus.state.position, fingerTarget(near));
  assert.deepEqual(stimulus.state.sensoryRates, zeroRates);
  for (let i = 0; i < 20; i++) {
    assert.deepEqual(stimulus.advance(far, 10), []);
    assert.deepEqual(stimulus.state.sensoryRates, zeroRates);
  }
  assert.deepEqual(stimulus.advance({ ...near, enabled: false }, 10), []);
  assert.equal(stimulus.state.position, null);
  assert.deepEqual(stimulus.state.sensoryRates, zeroRates);
});

test("the same approach on opposite sides swaps optical input weights, without supplying GF spikes", () => {
  const left = new FingerStimulus(protocol, 10);
  const right = new FingerStimulus(protocol, 10);
  for (let i = 0; i < 100; i++) {
    left.advance(near, 10);
    right.advance({ ...near, lateral: 1 }, 10);
  }
  const a = left.state.sensoryRates, b = right.state.sensoryRates;
  assert.ok(a[0] > a[1] && a[2] > a[3]);
  assert.ok(b[1] > b[0] && b[3] > b[2]);
  assert.ok(Math.abs(a[0] - b[1]) < 1e-12 && Math.abs(a[2] - b[3]) < 1e-12);
  assert.deepEqual(left.state.gfRates, [0, 0]);
  assert.deepEqual(right.state.gfRates, [0, 0]);
  const counts = new Uint16Array(10);
  counts[4] = 2; // Synthetic observed spikes test readout plumbing, not biology.
  left.observe(counts, 10);
  assert.ok(left.state.gfRates[0] > 0);
  assert.equal(left.state.gfRates[1], 0);
  left.reset();
  assert.deepEqual(left.state, { position: null, sensoryRates: zeroRates, gfRates: [0, 0] });
});

test("visual protocols reject empty, duplicate, out-of-range and sensory-overlapping GF identities", () => {
  const bad = (change: (value: FingerProtocol) => void) => {
    const value = structuredClone(protocol);
    change(value);
    assert.throws(() => new FingerStimulus(value, 10), RangeError);
  };
  bad((p) => { p.inputs.lc4Left.indices = []; });
  bad((p) => { p.inputs.lc4Left.indices = [1]; });
  bad((p) => { p.inputs.lc4Left.indices = [10]; });
  bad((p) => { p.outputs.gfLeft.index = 0; });
  bad((p) => { p.outputs.gfRight.index = p.outputs.gfLeft.index; });
});

test("world input validation is atomic; bounded finger motion resets reproducibly and cannot move an undriven mouth", () => {
  const brain = new SpikingBrain({ count: 10, offsets: new Uint32Array(11), targets: new Uint32Array(), weights: new Float32Array() }, 37);
  const world = new FeedingWorld(brain, feeding, protocol);
  const environment = { sugar: 0, bitter: 0, foodDistance: 0.7, finger: near };
  const rest = world.advance(0, environment);
  for (const finger of [{ ...near, lateral: NaN }, { ...near, proximity: Infinity }])
    assert.throws(() => world.advance(10, { ...environment, finger }), RangeError);
  assert.throws(() => world.advance(NaN, environment), RangeError);
  assert.throws(() => world.advance(-1, environment), RangeError);
  assert.deepEqual(world.advance(0, environment), rest);
  const first = world.advance(100, environment);
  assert.ok(first.finger!.position!.every(Number.isFinite));
  assert.ok(first.finger!.sensoryRates.every((rate) => Number.isFinite(rate) && rate >= 0 && rate <= 100));
  assert.deepEqual(first.finger!.gfRates, [0, 0]);
  assert.equal(first.extension, 0);
  assert.equal(first.motorRate, 0);
  world.reset(37);
  assert.deepEqual(world.advance(100, environment), first);
  const bounded = world.advance(2000, { ...environment, finger: { enabled: true, lateral: 20, proximity: 20 } });
  assert.ok(bounded.finger!.position!.every(Number.isFinite));
  assert.ok(Math.abs(bounded.finger!.position![0]) <= 1.4);
  assert.ok(bounded.finger!.position![2] >= -4 && bounded.finger!.position![2] <= -1.65 + 1e-12);
});
