import assert from "node:assert/strict";
import test from "node:test";
import { SpikingBrain, type Network, type ExplicitInput } from "./neural.ts";

// These tiny networks test numerical behavior only; they make no anatomical claim.
function network(
  count: number,
  edges: [number, number, number][] = [],
): Network {
  const ordered = edges.slice().sort((a, b) => a[0] - b[0]);
  const offsets = new Uint32Array(count + 1);
  for (const [source] of ordered) offsets[source + 1]++;
  for (let i = 1; i <= count; i++) offsets[i] += offsets[i - 1];
  return {
    count,
    offsets,
    targets: Uint32Array.from(ordered.map((edge) => edge[1])),
    weights: Float32Array.from(ordered.map((edge) => edge[2])),
  };
}

function close(actual: number, expected: number): void {
  assert.ok(Math.abs(actual - expected) < 1e-11, `${actual} != ${expected}`);
}

test("an undriven network remains at rest without invented background activity", () => {
  const brain = new SpikingBrain(
    network(3, [
      [0, 1, 1000],
      [1, 2, 1000],
    ]),
  );
  const result = brain.advance(100, []);
  assert.equal(result.spikes, 0);
  assert.deepEqual([...brain.voltage], [-52, -52, -52]);
  assert.deepEqual([...brain.current], [0, 0, 0]);
});

test("input precedes source spike, delayed current, and downstream spike", () => {
  const brain = new SpikingBrain(network(2, [[0, 1, 1000]]));
  const drives = [{ indices: [0], rateHz: 0 }];
  const events = [{ step: 0, index: 0 }];
  assert.equal(brain.advance(0.1, drives, [], events).spikes, 0);
  close(brain.voltage[0], 16.75);
  assert.deepEqual([...brain.advance(0.1, drives, [], events).counts], [1, 0]);
  close(brain.voltage[0], -52);
  brain.advance(1.7, drives, [], events);
  close(brain.current[1], 0);
  brain.advance(0.1, drives, [], events);
  close(brain.current[1], 275); // Source spike at step 1 + 18-step delay.
  close(brain.voltage[1], -52); // Synapses follow groups on the delivery step.
  brain.advance(0.1, drives, [], events);
  close(brain.current[1], 275 * Math.exp(-0.1 / 5));
  close(
    brain.voltage[1],
    -52 + (275 / 3) * (Math.exp(-0.1 / 20) - Math.exp(-0.1 / 5)),
  );
  assert.equal(brain.advance(1, drives, [], events).counts[1], 1);
});

test("signed synapses inhibit and equal opposite inputs cancel", () => {
  const drives = [{ indices: [0, 1], rateHz: 0 }];
  const events = [
    { step: 0, index: 0 },
    { step: 0, index: 1 },
  ];
  const inhibitory = new SpikingBrain(network(3, [[1, 2, -1000]]));
  assert.equal(inhibitory.advance(4, drives, [], events).counts[2], 0);
  assert.ok(inhibitory.voltage[2] < -52);
  assert.ok(inhibitory.current[2] < 0);

  const balanced = new SpikingBrain(
    network(3, [
      [0, 2, 1000],
      [1, 2, -1000],
    ]),
  );
  assert.equal(balanced.advance(4, drives, [], events).counts[2], 0);
  close(balanced.voltage[2], -52);
  close(balanced.current[2], 0);
});

test("silencing zeros outgoing transmission while preserving sensor spikes", () => {
  const graph = network(3, [
    [0, 2, 1000],
    [1, 2, -1000],
  ]);
  const events = [
    { step: 0, index: 0 },
    { step: 0, index: 1 },
  ];
  const drives = [{ indices: [0, 1], rateHz: 0 }];
  const intact = new SpikingBrain(graph).advance(4, drives, [], events);
  const lesion = new SpikingBrain(graph).advance(4, drives, [1], events);
  assert.deepEqual([...intact.counts], [1, 1, 0]);
  assert.deepEqual([...lesion.counts], [1, 1, 1]);
  assert.deepEqual([...graph.weights], [1000, -1000]);
});

test("a lesion applied before delivery suppresses already queued output", () => {
  const brain = new SpikingBrain(network(2, [[0, 1, 1000]]));
  const drives = [{ indices: [0], rateHz: 0 }];
  const events = [{ step: 0, index: 0 }];
  assert.equal(brain.advance(1, drives, [], events).counts[0], 1);
  assert.equal(brain.advance(4, drives, [0], events).counts[1], 0);
  close(brain.current[1], 0);
});

test("refractory cells reject incoming g writes as Brian's unless-refractory flag requires", () => {
  const brain = new SpikingBrain(network(2, [[0, 1, 1000]]));
  const drives = [{ indices: [0], rateHz: 0 }];
  const events = [
    { step: 0, index: 0 },
    { step: 10, index: 0 },
  ];
  const result = brain.advance(5, drives, [], events);
  assert.deepEqual([...result.counts], [2, 1]);
  close(brain.current[1], 0);
  close(brain.voltage[1], -52);
});

test("resets discard synapse-phase kicks on a neuron's firing step", () => {
  const brain = new SpikingBrain(network(1));
  const events = [
    { step: 0, index: 0 },
    { step: 1, index: 0 },
  ];
  const result = brain.advance(0.3, [{ indices: [0], rateHz: 0 }], [], events);
  assert.equal(result.spikes, 1);
  close(brain.voltage[0], -52);
});

test("seeded Poisson input and queued synapses are invariant to advance chunking", () => {
  const graph = network(3, [
    [0, 1, 1000],
    [1, 2, 800],
    [2, 1, -200],
  ]);
  const drives = [{ indices: [0], rateHz: 150 }];
  const whole = new SpikingBrain(graph, 314);
  const chunks = new SpikingBrain(graph, 314);
  const expected = whole.advance(100, drives);
  const actualCounts = new Uint16Array(3);
  let actualSpikes = 0;
  for (let i = 0; i < 100; i++) {
    const part = chunks.advance(1, drives);
    actualSpikes += part.spikes;
    for (let cell = 0; cell < 3; cell++)
      actualCounts[cell] += part.counts[cell];
  }
  assert.deepEqual(actualCounts, expected.counts);
  assert.equal(actualSpikes, expected.spikes);
  assert.equal(chunks.timeMs, whole.timeMs);
  assert.deepEqual(chunks.voltage, whole.voltage);
  assert.deepEqual(chunks.current, whole.current);
  assert.ok(actualSpikes > 0);
  chunks.reset(314);
  assert.deepEqual(chunks.advance(100, drives), expected);
});

test("fractional durations retain a stable grid and explicit fixtures do not replay", () => {
  const whole = new SpikingBrain(network(1));
  const chunks = new SpikingBrain(network(1));
  const events: ExplicitInput[] = [
    { step: 0, index: 0 },
    { step: 5, index: 0 },
  ];
  const drives = [{ indices: [0], rateHz: 9999 }];
  const expected = whole.advance(1, drives, [], events);
  let spikes = 0;
  for (let i = 0; i < 10; i++) {
    spikes += chunks.advance(0.03, drives, [], events).spikes;
    spikes += chunks.advance(0.07, drives, [], events).spikes;
  }
  assert.equal(chunks.timeMs, 1);
  assert.equal(spikes, expected.spikes);
  assert.equal(spikes, 2); // Explicit events suppress the high-rate Poisson drive.
  assert.deepEqual(chunks.voltage, whole.voltage);
});

test("invalid networks and inputs fail without advancing state", () => {
  assert.throws(
    () => new SpikingBrain({ ...network(2), offsets: new Uint32Array(2) }),
  );
  assert.throws(() => new SpikingBrain(network(2, [[0, 2, 1]])));
  const brain = new SpikingBrain(network(2));
  assert.throws(() => brain.advance(-1, []));
  assert.throws(() => brain.advance(10, [{ indices: [2], rateHz: 100 }]));
  assert.throws(() => brain.advance(10, [{ indices: [0], rateHz: Infinity }]));
  assert.throws(() => brain.advance(6554, []));
  assert.throws(() =>
    brain.advance(
      1,
      [],
      [],
      [
        { step: 2, index: 0 },
        { step: 1, index: 0 },
      ],
    ),
  );
  assert.equal(brain.timeMs, 0);
  assert.deepEqual([...brain.voltage], [-52, -52]);
});

test("fixed residuals and reactivation match an always-dense reference exactly", () => {
  const graph = network(5, [[0, 1, -10], [0, 2, 10], [3, 4, 1000]]);
  const brain = new SpikingBrain(graph);
  const driven = [0, 1, 3];
  const drives = [{ indices: driven, rateHz: 0 }];
  const events = [
    { step: 0, index: 0 },
    { step: 40_000, index: 1 }, // Voltage input wakes the inhibitory residual.
    { step: 40_010, index: 3 },
    { step: 40_020, index: 3 }, // Also exercise ordinary refractory cells.
    { step: 40_035, index: 3 },
    { step: 42_000, index: 0 }, // Delayed synapse wakes the positive residual.
  ];

  // Independent dense schedule: update every cell at every grid step, including
  // nonzero subnormal fixed points. No active-set bookkeeping or pruning.
  const voltage = new Float64Array(5).fill(-52);
  const current = new Float64Array(5);
  const lastSpike = new Float64Array(5).fill(-Infinity);
  const pending = new Map<number, number[]>();
  const membraneDecay = Math.exp(-0.1 / 20);
  const currentDecay = Math.exp(-0.1 / 5);
  const currentToVoltage = (membraneDecay - currentDecay) / 3;
  let eventIndex = 0;
  let sawFixedResiduals = false;

  for (let step = 0; step < 43_000; step++) {
    const counts = new Uint16Array(5);
    const fired: number[] = [];
    const refractory = (index: number) =>
      !driven.includes(index) && step - lastSpike[index] < 22;
    for (let index = 0; index < 5; index++) {
      if (refractory(index)) continue;
      voltage[index] = -52 + (voltage[index] + 52) * membraneDecay +
        current[index] * currentToVoltage;
      current[index] *= currentDecay;
      if (voltage[index] > -45) {
        fired.push(index);
        counts[index]++;
        lastSpike[index] = step;
      }
    }
    if (fired.length) pending.set(step + 18, fired);
    for (const source of pending.get(step) ?? []) {
      for (let edge = graph.offsets[source]; edge < graph.offsets[source + 1]; edge++) {
        const target = graph.targets[edge];
        if (!refractory(target)) current[target] += graph.weights[edge] * 0.275;
      }
    }
    pending.delete(step);
    while (eventIndex < events.length && events[eventIndex].step === step) {
      const index = events[eventIndex++].index;
      if (!refractory(index)) voltage[index] += 68.75;
    }
    for (const index of fired) {
      voltage[index] = -52;
      current[index] = 0;
    }

    const actual = brain.advance(0.1, drives, [], events);
    assert.deepEqual(actual.counts, counts, `Spike mismatch at step ${step}`);
    assert.deepEqual(brain.voltage, voltage, `Voltage mismatch at step ${step}`);
    assert.deepEqual(brain.current, current, `Current mismatch at step ${step}`);
    if (step === 39_000) {
      for (const index of [1, 2]) {
        assert.ok(current[index] !== 0 && Math.abs(current[index]) < 2.2250738585072014e-308);
        assert.equal(current[index] * currentDecay, current[index]);
        assert.equal(-52 + (voltage[index] + 52) * membraneDecay +
          current[index] * currentToVoltage, voltage[index]);
      }
      sawFixedResiduals = true;
    }
  }
  assert.ok(sawFixedResiduals, "Fixture must reach nonzero floating-point fixed points");
  assert.ok(Math.abs(current[2]) > 1e-100, "Delayed input must wake the positive residual");
});
