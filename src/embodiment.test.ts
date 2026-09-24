import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { Object3D, Vector3 } from "three";
import { FeedingWorld } from "./embodiment.ts";
import type { FeedingEnvironment } from "./embodiment.ts";
import type { FeedingProtocol } from "./neural-data.ts";
import { SpikingBrain } from "./neural.ts";
import {
  FEEDING_BODY_POSE,
  FOOD_RADII,
  feedingContact,
  foodPosition,
  mouthProbePosition,
} from "./feeding-geometry.ts";

// Artificial four-cell fixture tests the adapter contract only. These edges,
// labels and weights are not a biological circuit or scientific validation.
const protocol: FeedingProtocol = {
  inputs: {
    sugar: { ids: ["fixture-sugar"], indices: [0] },
    bitter: { ids: ["fixture-bitter"], indices: [1] },
  },
  outputs: {
    primary: { id: "fixture-left", index: 2, label: "fixture output left" },
    additional: [
      { id: "fixture-right", index: 3, label: "fixture output right" },
    ],
  },
};
const food: FeedingEnvironment = { sugar: 0.75, bitter: 0, foodDistance: 0 };

function fixture(connected = true, seed = 123) {
  const brain = new SpikingBrain(
    {
      count: 4,
      offsets: new Uint32Array(connected ? [0, 2, 2, 2, 2] : [0, 0, 0, 0, 0]),
      targets: new Uint32Array(connected ? [2, 3] : []),
      weights: new Float32Array(connected ? [200, 200] : []),
    },
    seed,
  );
  return { brain, world: new FeedingWorld(brain, protocol) };
}

test("without physical contact the world supplies no taste input or motor response", () => {
  const { world } = fixture();
  const state = world.advance(1000, { sugar: 1, bitter: 1, foodDistance: 0.7 });
  assert.equal(state.timeMs, 1000);
  assert.equal(state.contact, 0);
  assert.deepEqual(state.sensoryRates, { sugar: 0, bitter: 0 });
  assert.equal(state.spikes, 0);
  assert.equal(state.extension, 0);
  assert.deepEqual(state.motorRates, [0, 0]);
});

test("food cannot command movement without neural output; output drives extension and contact", () => {
  const disconnected = fixture(false).world.advance(1000, food);
  assert.ok(
    disconnected.counts[0] > 0,
    "Contact must activate the sensory fixture cell",
  );
  assert.equal(disconnected.motorRate, 0);
  assert.equal(disconnected.extension, 0);

  const { brain, world } = fixture();
  const active = world.advance(1000, food);
  assert.ok(active.motorRate > 0);
  assert.ok(active.extension > 0 && active.extension < 1);
  const voltage = brain.voltage.slice();
  let near = 0,
    far = 0.7;
  for (let i = 0; i < 50; i++) {
    const middle = (near + far) / 2;
    if (feedingContact(active.extension, middle) > 0.5) near = middle;
    else far = middle;
  }
  const distance = (near + far) / 2;
  const changed = world.advance(0, {
    sugar: 0.4,
    bitter: 0.8,
    foodDistance: distance,
  });
  assert.equal(changed.timeMs, active.timeMs);
  assert.equal(changed.extension, active.extension);
  assert.deepEqual(
    brain.voltage,
    voltage,
    "Moving food must not reset the brain",
  );
  assert.ok(Math.abs(changed.contact - 0.5) < 1e-12);
  assert.ok(Math.abs(changed.sensoryRates.sugar - 40) < 1e-10);
  assert.ok(Math.abs(changed.sensoryRates.bitter - 80) < 1e-10);
  const away = world.advance(2000, { ...food, foodDistance: 0.7 });
  assert.equal(away.timeMs, 3000);
  assert.ok(away.extension < active.extension / 100);
  assert.equal(away.contact, 0);
});

test("fixed feedback ticks and accumulated counts are independent of render call sizes", () => {
  const whole = fixture();
  const sliced = fixture();
  const expected = whole.world.advance(1000, food);
  const counts = new Uint32Array(4);
  let spikes = 0;
  let actual = sliced.world.advance(0, food);
  for (let i = 0; i < 60; i++) {
    actual = sliced.world.advance(1000 / 60, food);
    actual.counts.forEach((count, cell) => {
      counts[cell] += count;
    });
    spikes += actual.spikes;
  }
  assert.deepEqual(counts, expected.counts);
  assert.equal(spikes, expected.spikes);
  assert.equal(actual.timeMs, expected.timeMs);
  assert.deepEqual(actual.motorRates, expected.motorRates);
  assert.equal(actual.extension, expected.extension);
  assert.deepEqual(sliced.brain.voltage, whole.brain.voltage);
  assert.deepEqual(sliced.brain.current, whole.brain.current);
});

test("reset restores seeded brain and body state; environment changes retain the continuous clock", () => {
  const { world } = fixture();
  const first = world.advance(700, food);
  world.advance(237, { sugar: 0, bitter: 1, foodDistance: 0.02 });
  world.reset(123);
  assert.deepEqual(world.advance(700, food), first);
  const continued = world.advance(7, { ...food, bitter: 1 });
  assert.equal(continued.timeMs, 700);
  assert.equal(continued.extension, first.extension);
  assert.equal(world.advance(3, food).timeMs, 710);
  world.reset(123);
  const reset = world.advance(0, food);
  assert.equal(reset.timeMs, 0);
  assert.equal(reset.extension, 0);
  assert.equal(reset.motorRate, 0);
  assert.equal(reset.spikes, 0);
});

test("non-finite inputs fail before changing state and physical bounds stay finite", () => {
  const { world } = fixture();
  const rest = world.advance(0, food);
  assert.throws(() => world.advance(NaN, food), RangeError);
  assert.throws(
    () => world.advance(10, { ...food, foodDistance: NaN }),
    RangeError,
  );
  assert.deepEqual(world.advance(0, food), rest);
  const clipped = world.advance(0, { sugar: 2, bitter: -1, foodDistance: -10 });
  assert.deepEqual(clipped.sensoryRates, { sugar: 200, bitter: 0 });
  assert.equal(clipped.contact, 1);
});

test("contact probe follows the real GLB mouth hierarchy and the rendered food ellipsoid", () => {
  const bytes = readFileSync(new URL("../public/fly/fly.glb", import.meta.url));
  const jsonLength = bytes.readUInt32LE(12);
  const gltf = JSON.parse(bytes.subarray(20, 20 + jsonLength).toString());
  const binaryStart = 28 + jsonLength;
  const fly = new Object3D();
  const objects: Object3D[] = gltf.nodes.map(
    (node: { name: string; translation: number[] }) => {
      const object = new Object3D();
      object.name = node.name;
      object.position.fromArray(node.translation);
      return object;
    },
  );
  gltf.nodes.forEach((node: { children?: number[] }, i: number) => {
    node.children?.forEach((child) => objects[i].add(objects[child]));
  });
  gltf.scenes[0].nodes.forEach((index: number) => fly.add(objects[index]));
  fly.updateMatrixWorld(true);
  const feet = gltf.nodes.flatMap(
    (node: { name: string; extras: { footContact?: number[] } }, i: number) =>
      node.extras.footContact
        ? [
            {
              name: node.name,
              point: objects[i].localToWorld(
                new Vector3().fromArray(node.extras.footContact),
              ),
            },
          ]
        : [],
  );
  const front = feet.find(
    (foot: { name: string }) => foot.name === "leg_LF_tibia",
  )!.point;
  const rear = feet.find(
    (foot: { name: string }) => foot.name === "leg_LH_tibia",
  )!.point;
  const pitch = Math.atan((front.y - rear.y) / (front.z - rear.z));
  const height = -Math.min(
    ...feet.map(
      (foot: { point: Vector3 }) =>
        foot.point.clone().applyAxisAngle(new Vector3(1, 0, 0), pitch).y,
    ),
  );
  assert.ok(Math.abs(pitch - FEEDING_BODY_POSE.pitch) < 1e-12);
  assert.ok(Math.abs(height - FEEDING_BODY_POSE.height) < 1e-12);
  fly.rotation.x = pitch;
  fly.position.y = height;

  const rostrum = fly.getObjectByName("mouth_rostrum")!;
  const tip = fly.getObjectByName("mouth_tip")!;
  const node = gltf.nodes.find(
    (value: { name: string }) => value.name === "mouth_tip",
  );
  const accessor =
    gltf.accessors[gltf.meshes[node.mesh].primitives[0].attributes.POSITION];
  const view = gltf.bufferViews[accessor.bufferView];
  const offset = binaryStart + view.byteOffset + 106 * 12;
  const probe = new Vector3(
    bytes.readFloatLE(offset),
    bytes.readFloatLE(offset + 4),
    bytes.readFloatLE(offset + 8),
  );
  const axis = new Vector3().fromArray(node.extras.pitchAxis);
  for (let i = 0; i <= 20; i++) {
    const extension = i / 20;
    rostrum.quaternion.setFromAxisAngle(axis, (-extension * Math.PI) / 2);
    tip.quaternion.setFromAxisAngle(axis, (extension * Math.PI * 2) / 3);
    fly.updateMatrixWorld(true);
    const actual = tip.localToWorld(probe.clone());
    assert.ok(
      actual.distanceTo(new Vector3(...mouthProbePosition(extension))) < 1e-9,
    );
    const delta = actual.clone().sub(new Vector3(...foodPosition(0)));
    const inside = Math.hypot(
      delta.x / FOOD_RADII[0],
      delta.y / FOOD_RADII[1],
      delta.z / FOOD_RADII[2],
    );
    assert.ok(
      inside <= 1,
      "Default food must contain the probe throughout articulation",
    );
    assert.equal(feedingContact(extension, 0), 1);
    assert.equal(feedingContact(extension, 0.7), 0);
  }
  // Formerly the 1D approximation said no contact at this visibly touching pose.
  assert.equal(feedingContact(0, 0.05), 1);
  assert.equal(feedingContact(-1, -1), feedingContact(0, 0));
  assert.equal(feedingContact(2, 2), feedingContact(1, 0.7));
});
