/**
 * Full-network causal check of the exact bilateral looming VPN populations.
 * Run: npm run test:finger
 * Uniform 100 Hz VPN stimulation is an authored probe, not retinal vision or
 * a biological response calibration. No motor action is asserted or selected.
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { SpikingBrain, type Drive, type Network } from "../src/neural.ts";
import { FeedingWorld } from "../src/embodiment.ts";
import { EscapeBody } from "../src/body.ts";
import { fingerTarget, type FingerEnvironment } from "../src/finger.ts";
import {
  decodeNetwork,
  type DynamicsManifest,
  type FeedingProtocol,
} from "../src/neural-data.ts";

type Group = { ids: string[]; indices: number[] };
type Readout = { id: string; index: number; label: string };
type FingerProtocol = {
  datasetVersion: number;
  inputs: Record<"lc4Left" | "lc4Right" | "lplc2Left" | "lplc2Right", Group>;
  outputs: Record<"gfLeft" | "gfRight", Readout>;
  provenance: { graphSha256: string; idsSha256: string };
};
const assets = new URL("../public/dynamics/", import.meta.url);
const json = (file: string) => JSON.parse(readFileSync(new URL(file, assets), "utf8"));
const manifest: DynamicsManifest = json("manifest.json");
const finger: FingerProtocol = json("finger.json");
const feeding: FeedingProtocol = json("feeding.json");
const hash = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
assert.equal(finger.datasetVersion, 630);
assert.equal(finger.provenance.graphSha256, manifest.graph.sha256);
assert.equal(finger.provenance.idsSha256, manifest.ids.sha256);

const idBytes = gunzipSync(readFileSync(new URL(manifest.ids.file, assets)));
assert.equal(idBytes.byteLength, manifest.neuronCount * 8);
assert.equal(hash(idBytes), manifest.ids.sha256);
const allInputs = Object.values(finger.inputs).flatMap((group) => {
  assert.equal(group.ids.length, group.indices.length);
  group.indices.forEach((index, i) => {
    assert.ok(Number.isInteger(index) && index >= 0 && index < manifest.neuronCount);
    assert.equal(idBytes.readBigUInt64LE(index * 8).toString(), group.ids[i]);
  });
  return group.indices;
});
assert.deepEqual(Object.values(finger.inputs).map((group) => group.indices.length), [54, 50, 108, 102]);
assert.equal(new Set(allInputs).size, 314);
const readouts = [finger.outputs.gfLeft, finger.outputs.gfRight];
assert.deepEqual(readouts.map((row) => row.id), ["720575940622838154", "720575940632499757"]);
const mn9 = [feeding.outputs.primary, feeding.outputs.additional[0]];
for (const row of [...readouts, ...mn9]) {
  assert.equal(idBytes.readBigUInt64LE(row.index * 8).toString(), row.id);
  assert.ok(!allInputs.includes(row.index), "GF and MN9 must never be direct sensory inputs");
}

assert.equal(manifest.format, "LIF1");
assert.ok(manifest.graph.bytes > 0 && manifest.graph.bytes <= 260_000_000);
const buffer = new ArrayBuffer(manifest.graph.bytes);
const destination = new Uint8Array(buffer);
let offset = 0;
for (const chunk of manifest.graph.chunks) {
  assert.equal(chunk.offset, offset, "Graph chunks must be contiguous");
  const compressed = readFileSync(new URL(chunk.file, assets));
  assert.equal(compressed.byteLength, chunk.compressedBytes);
  const raw = gunzipSync(compressed);
  assert.equal(raw.byteLength, chunk.bytes);
  assert.equal(hash(raw), chunk.sha256);
  destination.set(raw, offset);
  offset += raw.byteLength;
}
assert.equal(offset, destination.byteLength);
assert.equal(hash(destination), manifest.graph.sha256);
const network = decodeNetwork(buffer);
assert.equal(network.count, manifest.neuronCount);
assert.equal(network.targets.length, manifest.edgeCount);
console.log(`Verified ${network.count} neurons / ${network.targets.length} unchanged edges and all exact v630 input/readout IDs.`);

const left: Drive[] = [finger.inputs.lc4Left, finger.inputs.lplc2Left]
  .map((group) => ({ indices: group.indices, rateHz: 100 }));
const right: Drive[] = [finger.inputs.lc4Right, finger.inputs.lplc2Right]
  .map((group) => ({ indices: group.indices, rateHz: 100 }));
const DURATION_MS = 1000;
function phase(brain: SpikingBrain, label: string, drives: Drive[]) {
  const started = performance.now();
  const startMs = brain.timeMs;
  const result = brain.advance(DURATION_MS, drives);
  assert.equal(result.timeMs, startMs + DURATION_MS);
  const gf = readouts.map((row) => result.counts[row.index]);
  const feedingSpikes = mn9.map((row) => result.counts[row.index]);
  assert.deepEqual(feedingSpikes, [0, 0], "This isolated visual probe unexpectedly recruited MN9");
  const sensorySpikes = allInputs.reduce((sum, index) => sum + result.counts[index], 0);
  assert.ok(brain.voltage.every(Number.isFinite));
  assert.ok(brain.current.every(Number.isFinite));
  console.log(`${label}: GF L/R ${gf.join(" / ")} Hz; MN9 ${feedingSpikes.join(" / ")}; ${sensorySpikes} VPN spikes; ${(performance.now() - started).toFixed(0)} ms wall / ${DURATION_MS} ms model.`);
  return { gf, sensorySpikes, spikes: result.spikes };
}

// One brain and clock; changing input side does not reset neural state.
const brain = new SpikingBrain(network, 37);
const baseline = phase(brain, "No input", []);
assert.equal(baseline.spikes, 0);
const leftResult = phase(brain, "Left LC4/LPLC2, 100 Hz", left);
const rightResult = phase(brain, "Right LC4/LPLC2, 100 Hz", right);
assert.ok(leftResult.sensorySpikes > 0 && rightResult.sensorySpikes > 0);
assert.ok([...leftResult.gf, ...rightResult.gf].every((value) => value > 0));
// These assert different neural responses in identified cells, not steering.
assert.ok(leftResult.gf[0] > rightResult.gf[0]);
assert.ok(rightResult.gf[1] > leftResult.gf[1]);

const disconnected: Network = {
  count: network.count,
  offsets: new Uint32Array(network.count + 1),
  targets: new Uint32Array(),
  weights: new Float32Array(),
};
const control = phase(new SpikingBrain(disconnected, 37), "Disconnected, bilateral 100 Hz", [...left, ...right]);
assert.ok(control.sensorySpikes > 0);
assert.deepEqual(control.gf, [0, 0]);
console.log("PASS: continuous side-dependent GF computation, undriven rest, no direct motor stimulation, and disconnected causal control.");

function motionPhase(world: FeedingWorld, label: string, fingerEnvironment: FingerEnvironment, durationMs: number, detachedMotor: EscapeBody) {
  const environment = { sugar: 0, bitter: 0, foodDistance: 0.7, finger: fingerEnvironment };
  const startMs = world.advance(0, environment).timeMs;
  const started = performance.now();
  const gfSpikes = [0, 0];
  let sensorySpikes = 0;
  const maximumDrive = [0, 0, 0, 0];
  let state = world.advance(0, environment);
  let minBodyHeight = state.body.height, maxBodyHeight = state.body.height, maxBodySpeed = Math.abs(state.body.velocity);
  let sourceSpikesSeen = false;
  let previousPosition = state.finger!.position;
  for (let elapsed = 0; elapsed < durationMs; elapsed += 100) {
    const dtMs = Math.min(100, durationMs - elapsed);
    state = world.advance(dtMs, environment);
    const fingerState = state.finger!;
    assert.ok(fingerState.position!.every(Number.isFinite));
    assert.ok(fingerState.gfRates.every((rate) => Number.isFinite(rate) && rate >= 0));
    assert.ok(fingerState.sensoryRates.every((rate) => Number.isFinite(rate) && rate >= 0 && rate <= 100));
    if (previousPosition)
      assert.ok(Math.hypot(...fingerState.position!.map((v, i) => v - previousPosition![i])) <= 1.5 * dtMs / 1000 + 1e-10, "Displayed finger must follow bounded physical motion");
    previousPosition = fingerState.position;
    for (let side = 0; side < 2; side++) gfSpikes[side] += state.counts[readouts[side].index];
    sourceSpikesSeen ||= readouts.some(row => state.counts[row.index] > 0);
    // A true motor-link ablation: the same run still computes GF spikes, but
    // the isolated actuator receives none. This does not silence graph edges.
    const detached = detachedMotor.advance([0, 0], dtMs);
    assert.equal(detached.height, 0);
    assert.equal(detached.takeoffs, 0);
    assert.ok([state.body.height, state.body.velocity, ...state.body.activation, ...state.body.legExtension].every(Number.isFinite));
    assert.ok(state.body.height >= 0 && state.body.activation.every(a => a >= 0 && a <= 1));
    minBodyHeight = Math.min(minBodyHeight, state.body.height);
    maxBodyHeight = Math.max(maxBodyHeight, state.body.height);
    maxBodySpeed = Math.max(maxBodySpeed, Math.abs(state.body.velocity));
    fingerState.sensoryRates.forEach((rate, i) => { maximumDrive[i] = Math.max(maximumDrive[i], rate); });
    sensorySpikes += allInputs.reduce((sum, index) => sum + state.counts[index], 0);
    assert.deepEqual(state.sensoryRates, { sugar: 0, bitter: 0 });
    assert.deepEqual(mn9.map((row) => state.counts[row.index]), [0, 0]);
    assert.equal(state.extension, 0, "Visual stimulation must not directly actuate the mouth");
  }
  assert.equal(state.timeMs, startMs + durationMs);
  const rates = gfSpikes.map((value) => value * 1000 / durationMs);
  console.log(`${label}: model ${startMs}–${state.timeMs}ms; position [${state.finger!.position!.map(v => v.toFixed(3)).join(", ")}]; GF L/R ${rates.map(v => v.toFixed(1)).join(" / ")}Hz; peak sampled LC4/LPLC2 input [${maximumDrive.map(v => v.toFixed(1)).join(", ")}]; body height/peak ${state.body.height.toFixed(3)}/${state.body.peakHeight.toFixed(3)}, takeoffs ${state.body.takeoffs}; detached motor0; ${(performance.now() - started).toFixed(0)}ms wall.`);
  return { state, gfSpikes, sensorySpikes, maximumDrive, sourceSpikesSeen, bodyStill: maxBodyHeight - minBodyHeight < 1e-10 && maxBodySpeed < 1e-10 };
}

function trajectory(world: FeedingWorld, label: string, connected: boolean) {
  const detachedMotor = new EscapeBody();
  const run = (phaseLabel: string, env: FingerEnvironment, durationMs: number) => motionPhase(world, `${label} ${phaseLabel}`, env, durationMs, detachedMotor);
  const leftFar = { enabled: true, lateral: -1, proximity: 0 };
  const leftNear = { ...leftFar, proximity: 1 };
  const rightFar = { ...leftFar, lateral: 1 };
  const rightNear = { ...rightFar, proximity: 1 };
  const far = run("visible at far plane", leftFar, 100);
  assert.deepEqual(far.gfSpikes, [0, 0]);
  assert.deepEqual(far.maximumDrive, [0, 0, 0, 0]);
  const left = run("left approach", leftNear, 1600);
  assert.deepEqual(left.state.finger!.position, fingerTarget(leftNear));
  const stillLeft = run("stationary left", leftNear, 200);
  // A stationary world object can expand on a moving retina as the fly falls.
  if (stillLeft.bodyStill) assert.deepEqual(stillLeft.maximumDrive, [0, 0, 0, 0]);
  const withdrawal = run("withdraw left", leftFar, 1600);
  if (withdrawal.bodyStill) assert.deepEqual(withdrawal.maximumDrive, [0, 0, 0, 0]);
  assert.deepEqual(withdrawal.state.finger!.position, fingerTarget(leftFar));
  run("reposition at far plane", rightFar, 1900);
  const right = run("right approach", rightNear, 1600);
  assert.deepEqual(right.state.finger!.position, fingerTarget(rightNear));
  const stillRight = run("stationary right", rightNear, 200);
  if (stillRight.bodyStill) assert.deepEqual(stillRight.maximumDrive, [0, 0, 0, 0]);
  assert.ok(left.sensorySpikes > 0 && right.sensorySpikes > 0);
  if (connected) {
    assert.ok(left.gfSpikes.some(count => count > 0) && right.gfSpikes.some(count => count > 0));
    assert.notDeepEqual(left.gfSpikes, right.gfSpikes, "Computed response changes with the optical/mechanical history; no direction is prescribed");
    assert.ok(left.state.body.peakHeight > 0 && right.state.body.peakHeight > 0);
    assert.ok(left.state.body.takeoffs > 0 && right.state.body.takeoffs > 0);
    assert.ok(left.sourceSpikesSeen && right.sourceSpikesSeen, "Detached motor control must be paired with genuine GF spikes");
  } else {
    assert.deepEqual(left.gfSpikes, [0, 0]);
    assert.deepEqual(right.gfSpikes, [0, 0]);
    assert.equal(stillRight.state.body.peakHeight, 0);
    assert.equal(stillRight.state.body.takeoffs, 0);
  }
}

// An independent run now verifies the production geometry adapter, including
// its returned physical trajectory and continuous, unreset neural clock.
brain.reset(37);
trajectory(new FeedingWorld(brain, feeding, finger), "Geometry", true);
trajectory(new FeedingWorld(new SpikingBrain(disconnected, 37), feeding, finger), "Disconnected geometry", false);
console.log("PASS: bounded physical approach drives computed GF/body activity and optical feedback; disconnected graph and detached motor link remain still. Stationary external objects are silent only while the body is also still.");
console.log("These numerical probes do not validate retinal transduction, natural firing rates, steering, or escape behavior.");
