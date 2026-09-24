/**
 * Offline causal regression against the complete local model assets.
 * Run: node --experimental-strip-types tools/check-feeding.ts
 *
 * One seeded run checks direction of response, not biological calibration or
 * the paper's multi-trial statistics. The mouth/contact adapter is authored.
 * Durations are reported, never used as machine-dependent pass criteria.
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { FeedingWorld, type FeedingEnvironment } from "../src/embodiment.ts";
import { SpikingBrain, type Network } from "../src/neural.ts";
import {
  decodeNetwork,
  type DynamicsManifest,
  type FeedingProtocol,
  type FingerProtocol,
} from "../src/neural-data.ts";

const assets = new URL("../public/dynamics/", import.meta.url);
const manifest: DynamicsManifest = JSON.parse(
  readFileSync(new URL("manifest.json", assets), "utf8"),
);
const protocol: FeedingProtocol = JSON.parse(
  readFileSync(new URL(manifest.protocol.file, assets), "utf8"),
);
const fingerProtocol: FingerProtocol = JSON.parse(
  readFileSync(new URL("finger.json", assets), "utf8"),
);
const giantFibers = [fingerProtocol.outputs.gfLeft.index, fingerProtocol.outputs.gfRight.index];
const hash = (bytes: Uint8Array) =>
  createHash("sha256").update(bytes).digest("hex");

function localNetwork(): Network {
  assert.equal(manifest.format, "LIF1", "Unsupported graph format");
  assert.ok(manifest.graph.bytes > 0 && manifest.graph.bytes <= 260_000_000);
  const buffer = new ArrayBuffer(manifest.graph.bytes);
  const destination = new Uint8Array(buffer);
  let offset = 0;
  for (const chunk of manifest.graph.chunks) {
    assert.equal(chunk.offset, offset, "Graph chunks must be contiguous");
    const compressed = readFileSync(new URL(chunk.file, assets));
    assert.equal(compressed.byteLength, chunk.compressedBytes, chunk.file);
    const bytes = gunzipSync(compressed);
    assert.equal(bytes.byteLength, chunk.bytes, chunk.file);
    assert.equal(hash(bytes), chunk.sha256, `Chunk hash: ${chunk.file}`);
    destination.set(bytes, offset);
    offset += bytes.byteLength;
  }
  assert.equal(offset, destination.byteLength, "Incomplete graph");
  assert.equal(hash(destination), manifest.graph.sha256, "Whole graph hash");
  const network = decodeNetwork(buffer);
  assert.equal(network.count, manifest.neuronCount);
  assert.equal(network.targets.length, manifest.edgeCount);
  return network;
}

const loadStarted = performance.now();
const network = localNetwork();
console.log(
  `Verified ${network.count.toLocaleString("en-US")} neurons / ` +
    `${network.targets.length.toLocaleString("en-US")} signed edges ` +
    `in ${(performance.now() - loadStarted).toFixed(0)} ms.`,
);

const SEED = 37;
const DURATION_MS = 1000;
const motors = [
  protocol.outputs.primary.index,
  protocol.outputs.additional[0].index,
];
const inputs = [
  ...protocol.inputs.sugar.indices,
  ...protocol.inputs.bitter.indices,
];
assert.ok(
  motors.every((index) => !inputs.includes(index)),
  "MN9 must not be a driven sensory cell",
);

function phase(
  label: string,
  brain: SpikingBrain,
  world: FeedingWorld,
  environment: FeedingEnvironment,
) {
  const started = performance.now();
  const startTime = brain.timeMs;
  const motorSpikes = [0, 0];
  const lateMotorSpikes = [0, 0];
  const gfSpikes = [0, 0];
  let sensorySpikes = 0;
  let state = world.advance(0, environment);
  for (let elapsed = 0; elapsed < DURATION_MS; elapsed += 100) {
    state = world.advance(100, environment);
    for (let side = 0; side < 2; side++) {
      motorSpikes[side] += state.counts[motors[side]];
      if (elapsed >= DURATION_MS - 500)
        lateMotorSpikes[side] += state.counts[motors[side]];
      gfSpikes[side] += state.counts[giantFibers[side]];
    }
    for (const index of inputs) sensorySpikes += state.counts[index];
  }
  const wallMs = performance.now() - started;
  assert.equal(
    state.timeMs,
    startTime + DURATION_MS,
    "Model clock must remain continuous",
  );
  assert.ok(
    brain.voltage.every(Number.isFinite),
    `${label}: non-finite voltage`,
  );
  assert.ok(
    brain.current.every(Number.isFinite),
    `${label}: non-finite current`,
  );
  assert.ok(
    Number.isFinite(state.extension) &&
      state.extension >= 0 &&
      state.extension <= 1,
  );
  assert.ok(
    Number.isFinite(state.contact) && state.contact >= 0 && state.contact <= 1,
  );
  const rates = motorSpikes.map((count) => (count * 1000) / DURATION_MS);
  console.log(
    `${label.padEnd(25)} MN9 L/R ${rates.join(" / ")} Hz; ` +
      `extension ${state.extension.toFixed(3)}; contact ${state.contact.toFixed(3)}; ` +
      `GF ${gfSpikes.join("/")}; body peak ${state.body.peakHeight.toFixed(6)}; takeoffs ${state.body.takeoffs}; ` +
      `${wallMs.toFixed(0)} ms wall / ${DURATION_MS} ms model`,
  );
  assert.deepEqual(gfSpikes, [0, 0], `${label}: feeding unexpectedly recruits GF`);
  assert.equal(state.body.peakHeight, 0, `${label}: the active GF/body link must not spuriously move the body`);
  return { ...state, motorSpikes, lateMotorSpikes, sensorySpikes };
}

// Keep the same brain, queued events and body state through all interventions.
const brain = new SpikingBrain(network, SEED);
const world = new FeedingWorld(brain, protocol, fingerProtocol);
const food = { sugar: 0.75, bitter: 0, foodDistance: 0 };
const sugar = phase("Sugar contact", brain, world, food);
const bitter = phase("Add bitter", brain, world, { ...food, bitter: 1 });
const recovery = phase("Remove bitter", brain, world, food);
const away = phase("Move beyond reach", brain, world, {
  ...food,
  foodDistance: 0.7,
});
const totalMotor = (result: ReturnType<typeof phase>) =>
  result.motorSpikes[0] + result.motorSpikes[1];

assert.ok(
  sugar.sensorySpikes > 0 && sugar.motorSpikes[0] > 0 && sugar.extension > 0,
  "Contact with sugar must produce sensory, primary MN9 and actuator response",
);
assert.ok(
  bitter.motorSpikes[0] < sugar.motorSpikes[0] &&
    totalMotor(bitter) < totalMotor(sugar) &&
    bitter.extension < sugar.extension,
  "Bitter co-stimulation must suppress the seeded sugar response",
);
assert.ok(
  recovery.motorSpikes[0] > bitter.motorSpikes[0] &&
    totalMotor(recovery) > totalMotor(bitter) &&
    recovery.extension > bitter.extension,
  "Removing bitter must recover neural and actuator response without resetting",
);
assert.equal(away.contact, 0);
assert.deepEqual(away.sensoryRates, { sugar: 0, bitter: 0 });
assert.deepEqual(
  away.lateMotorSpikes,
  [0, 0],
  "MN9 must fall silent after contact is lost",
);
assert.ok(
  away.extension < recovery.extension,
  "The actuator must relax after losing contact",
);
console.log(
  `Out-of-reach final 500 ms: MN9 L/R ${away.lateMotorSpikes.map((n) => n * 2).join(" / ")} Hz.`,
);

// Same cells and stimuli, with every connection removed: food cannot directly
// command the actuator. Sensor spikes alone must not become movement.
const disconnected: Network = {
  count: network.count,
  offsets: new Uint32Array(network.count + 1),
  targets: new Uint32Array(),
  weights: new Float32Array(),
};
const controlBrain = new SpikingBrain(disconnected, SEED);
const controlWorld = new FeedingWorld(controlBrain, protocol, fingerProtocol);
const control = phase("Disconnected control", controlBrain, controlWorld, food);
assert.ok(
  control.sensorySpikes > 0,
  "Control must still stimulate sensory cells",
);
assert.deepEqual(control.motorSpikes, [0, 0]);
assert.equal(control.extension, 0);

console.log(
  "PASS: full-graph sugar response, bitter suppression, continuous recovery, contact loss and disconnected control.",
);
console.log(
  "This is a numerical/causal regression, not validation of biological movement or adapter parameters.",
);
