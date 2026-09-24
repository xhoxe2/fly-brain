import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { Object3D, Vector3 } from "three";
import { FEEDING_BODY_POSE } from "./feeding-geometry.ts";

test("the actual middle-leg mesh supports the body adapter's downward stroke", () => {
  const bytes = readFileSync(new URL("../public/fly/fly.glb", import.meta.url));
  const gltf = JSON.parse(bytes.subarray(20, 20 + bytes.readUInt32LE(12)).toString());
  const nodes: Object3D[] = gltf.nodes.map((node: {
    name: string; translation?: number[]; rotation?: number[];
    extras?: Record<string, unknown>;
  }) => {
    const object = new Object3D();
    object.name = node.name;
    if (node.translation) object.position.fromArray(node.translation);
    if (node.rotation) object.quaternion.fromArray(node.rotation);
    object.userData = node.extras ?? {};
    return object;
  });
  gltf.nodes.forEach((node: { children?: number[] }, index: number) => {
    node.children?.forEach(child => nodes[index].add(nodes[child]));
  });
  const fly = new Object3D();
  gltf.scenes[0].nodes.forEach((index: number) => fly.add(nodes[index]));
  fly.position.y = FEEDING_BODY_POSE.height;
  fly.rotation.x = FEEDING_BODY_POSE.pitch;
  const reachBySide: number[][] = [];
  for (const side of ["LM", "RM"]) {
    const femur = fly.getObjectByName(`leg_${side}_femur`)!;
    const tibia = fly.getObjectByName(`leg_${side}_tibia`)!;
    const femurRest = femur.quaternion.clone();
    const tibiaRest = tibia.quaternion.clone();
    const femurAxis = new Vector3().fromArray(femur.userData.pitchAxis);
    const tibiaAxis = new Vector3().fromArray(tibia.userData.pitchAxis);
    const contact = new Vector3().fromArray(tibia.userData.footContact);
    fly.updateMatrixWorld(true);
    assert.ok(Math.abs(contact.clone().applyMatrix4(tibia.matrixWorld).y) < 1e-12);
    const reaches: number[] = [];
    // These are pinned asset/actuator assumptions, not measured biomechanics:
    // both mirrored pitch axes extend the foot with the same positive sign.
    for (let step = 0; step <= 100; step++) {
      const angle = 0.35 * step / 100;
      femur.quaternion.copy(femurRest);
      tibia.quaternion.copy(tibiaRest);
      femur.rotateOnAxis(femurAxis, angle);
      tibia.rotateOnAxis(tibiaAxis, angle * 0.3);
      fly.updateMatrixWorld(true);
      const reach = -contact.clone().applyMatrix4(tibia.matrixWorld).y;
      if (step) assert.ok(reach > reaches[step - 1]);
      reaches.push(reach);
    }
    assert.ok(reaches.at(-1)! > 0.3, "The actual rig must cover the maximum body stroke");
    reachBySide.push(reaches);
    femur.quaternion.copy(femurRest);
    tibia.quaternion.copy(tibiaRest);
  }
  assert.deepEqual(reachBySide[0], reachBySide[1]);
});
