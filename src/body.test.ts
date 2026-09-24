import assert from "node:assert/strict";
import test from "node:test";
import { EscapeBody } from "./body.ts";
import { FeedingWorld } from "./embodiment.ts";
import { FingerStimulus } from "./finger.ts";
import { feedingContact, foodPosition, mouthProbePosition } from "./feeding-geometry.ts";
import { SpikingBrain } from "./neural.ts";
import type { FeedingProtocol, FingerProtocol } from "./neural-data.ts";

// Synthetic identities/edges test the mechanical boundary, not anatomy.
const group = (index: number) => ({ ids: [`fixture-${index}`], indices: [index] });
const output = (index: number) => ({ id: `fixture-${index}`, index, label: "synthetic output" });
const finger: FingerProtocol = {
  inputs: { lc4Left: group(0), lc4Right: group(1), lplc2Left: group(2), lplc2Right: group(3) },
  outputs: { gfLeft: output(4), gfRight: output(5) },
};
const feeding: FeedingProtocol = {
  inputs: { sugar: group(6), bitter: group(7) },
  outputs: { primary: output(8), additional: [output(9)] },
};
const environment = { sugar: 0, bitter: 0, foodDistance: 0.7, finger: { enabled: true, lateral: -1, proximity: 1 } };
function world(connected = true) {
  const brain = new SpikingBrain({
    count: 10,
    offsets: new Uint32Array(connected ? [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2] : 11),
    targets: new Uint32Array(connected ? [4, 5] : []),
    weights: new Float32Array(connected ? [250, 250] : []),
  }, 37);
  return { brain, body: new FeedingWorld(brain, feeding, finger) };
}

test("body excitation uses GF counts alone; rest and an open motor link remain still", () => {
  const body = new EscapeBody();
  const rest = body.state;
  assert.deepEqual(body.advance([0, 0], 1000), rest);
  const excited = body.advance([1, 0], 10);
  assert.ok(excited.activation[0] > 0);
  assert.equal(excited.activation[1], 0);
  assert.ok(excited.height > 0 && excited.velocity > 0);
  // No GF threshold, stimulus label or preselected jump phase enters the API.
  const detached = new EscapeBody();
  for (let i = 0; i < 100; i++) detached.advance([0, 0], 10);
  assert.deepEqual(detached.state, rest);
  body.reset();
  assert.deepEqual(body.state, rest);
  assert.deepEqual(body.advance([1, 0], 10), excited);
  const invalid = body.state;
  for (const args of [[[NaN, 0], 10], [[-1, 0], 10], [[0.5, 0], 10], [[0, 0], Infinity], [[0, 0], -1]] as const)
    assert.throws(() => body.advance(args[0], args[1]), RangeError);
  assert.deepEqual(body.state, invalid);
});

test("airborne GF spikes can excite a muscle but cannot change vertical momentum without contact", () => {
  const active = new EscapeBody(), passive = new EscapeBody();
  // One initial impulse leaves room below activation saturation, so the new
  // airborne impulses actually change excitation and cannot be a vacuous test.
  active.advance([1, 1], 10);
  passive.advance([1, 1], 10);
  active.advance([0, 0], 90);
  passive.advance([0, 0], 90);
  assert.ok(active.state.height > 0.3 && active.state.velocity > 0 && !active.state.grounded);
  const initialVelocity = active.state.velocity;
  const a = active.advance([10, 10], 10), b = passive.advance([0, 0], 10);
  assert.ok(a.activation[0] > b.activation[0]);
  assert.equal(a.height, b.height);
  assert.equal(a.velocity, b.velocity);
  assert.ok(a.velocity < initialVelocity, "Gravity reduces upward speed even with fresh spikes");
});

test("after spikes cease ballistic energy does not grow and the body settles without floor penetration", () => {
  const body = new EscapeBody();
  for (let i = 0; i < 10; i++) body.advance([2, 2], 10);
  let previous = body.state;
  for (let i = 0; i < 600; i++) {
    const state = body.advance([0, 0], 10);
    assert.ok([state.height, state.velocity, ...state.activation, ...state.legExtension].every(Number.isFinite));
    assert.ok(state.height >= 0);
    assert.ok(state.activation.every(v => v >= 0 && v <= 1));
    if (previous.height > 0.31 && state.height > 0.31) {
      const energy = (s: typeof state) => 20 * s.height + s.velocity ** 2 / 2;
      assert.ok(energy(state) <= energy(previous) + 1e-10, "No external contact: mechanical energy must not increase");
    }
    previous = state;
  }
  assert.ok(body.state.height < 1e-10 && Math.abs(body.state.velocity) < 1e-10);
  assert.ok(body.state.grounded);
  assert.equal(body.state.takeoffs, 1, "Passive landing must not create another reported takeoff");
});

test("sustained GF trains produce bounded tonic contraction rather than repeated propulsive strokes", () => {
  for (const hz of [10, 50, 100]) {
    const body = new EscapeBody();
    let low = Infinity, high = -Infinity;
    let lastSecondTakeoffs = 0;
    for (let tick = 0; tick < 500; tick++) {
      const state = body.advance(tick % (100 / hz) === 0 ? [1, 1] : [0, 0], 10);
      if (tick === 400) lastSecondTakeoffs = state.takeoffs;
      if (tick >= 400) {
        low = Math.min(low, state.height);
        high = Math.max(high, state.height);
      }
      assert.ok(state.height >= 0 && Number.isFinite(state.velocity));
      assert.ok(state.activation.every(value => value >= 0 && value <= 1));
    }
    // Physical height is tested as well as the reporting counter: changing a
    // contact tolerance alone cannot hide oscillation. 0.003 is 1% of the
    // adapter's authored 0.3-unit maximum stroke, not a biological calibration.
    assert.ok(high - low < 0.003, `${hz} Hz: sustained contraction must settle`);
    assert.equal(body.state.takeoffs, lastSecondTakeoffs, `${hz} Hz: no late repeated takeoffs`);
    assert.equal(body.state.takeoffs, 1, `${hz} Hz: passive landing must absorb the initial takeoff`);
  }
});

test("closed-loop body and neural state are reproducible across render chunks and reset", () => {
  const whole = world(), sliced = world();
  const expected = whole.body.advance(1000, environment);
  const counts = new Uint32Array(10);
  let spikes = 0;
  for (let i = 0; i < 50; i++) {
    for (const duration of [7, 13]) {
      const state = sliced.body.advance(duration, environment);
      state.counts.forEach((count, index) => counts[index] += count);
      spikes += state.spikes;
    }
  }
  const actual = sliced.body.advance(0, environment);
  assert.deepEqual({ ...actual, counts, spikes }, expected);
  assert.deepEqual(sliced.brain.voltage, whole.brain.voltage);
  assert.deepEqual(sliced.brain.current, whole.brain.current);
  assert.ok(expected.body.peakHeight > 0 && expected.body.takeoffs > 0);
  whole.body.reset(37);
  assert.deepEqual(whole.body.advance(1000, environment), expected);
  const detached = world(false).body.advance(1000, environment);
  assert.ok(detached.spikes > 0, "Sensory cells must still fire in the control");
  assert.equal(detached.body.height, 0);
  assert.equal(detached.body.takeoffs, 0);
  assert.deepEqual(detached.finger!.gfRates, [0, 0]);
});

test("world-anchored food and finger use the body's translated mouth and eyes", () => {
  const food = foodPosition(0);
  for (const extension of [0, 0.5, 1]) {
    const ground = mouthProbePosition(extension), raised = mouthProbePosition(extension, 1);
    assert.equal(raised[0], ground[0]);
    assert.ok(Math.abs(raised[1] - ground[1] - 1) < 1e-12);
    assert.equal(raised[2], ground[2]);
    assert.equal(feedingContact(extension, 0), 1);
    assert.equal(feedingContact(extension, 0, 1), 0);
  }
  assert.deepEqual(foodPosition(0), food, "Food must not move with the body");
  const visual = new FingerStimulus(finger, 10);
  for (let i = 0; i < 200; i++) visual.advance(environment.finger, 10, 1);
  const position = visual.state.position;
  assert.deepEqual(visual.state.sensoryRates, [0, 0, 0, 0]);
  visual.advance(environment.finger, 10, 0.9);
  assert.deepEqual(visual.state.position, position, "Finger must remain world-anchored");
  assert.ok(visual.state.sensoryRates.some(rate => rate > 0), "Descending eyes see positive expansion from a stationary nearby finger");
  visual.advance(environment.finger, 10, 1);
  assert.deepEqual(visual.state.sensoryRates, [0, 0, 0, 0], "Moving away produces no positive expansion");
});
