type Point = readonly [number, number, number];
const clamp = (value: number, low: number, high: number) =>
  Math.max(low, Math.min(high, value));

// Geometry of the pinned NeuroMechFly derivative in public/fly/fly.glb.
// Tests compare this compact worker-side rig with the actual GLB hierarchy.
export const FEEDING_BODY_POSE = {
  pitch: -0.10272848264169568,
  height: -0.08006998966175488,
} as const;
const ROSTRUM_PIVOT: Point = [0, 0.8919758014876785, -1.0479483610523148];
const TIP_OFFSET: Point = [0, -0.0157756402551541, 0.2986103334011304];
// Distal surface vertex 106 of mouth_tip; a contact probe, not a full collision mesh.
const TIP_PROBE: Point = [
  0.030392799526453018, -0.10040166974067688, -0.2814559042453766,
];
export const FOOD_RADII: Point = [0.22, 0.374, 0.22];
const FOOD_ORIGIN: Point = [0, 0.47, -1.19];
const SOFT_CONTACT_DISTANCE = 0.04;

function rotateX(point: Point, angle: number): [number, number, number] {
  const c = Math.cos(angle),
    s = Math.sin(angle);
  return [point[0], point[1] * c - point[2] * s, point[1] * s + point[2] * c];
}

export function feedingJointAngles(extension: number) {
  const amount = clamp(extension, 0, 1);
  return { rostrum: (-amount * Math.PI) / 2, tip: (amount * Math.PI * 2) / 3 };
}

/** The same two joint rotations and standing transform used by the renderer. */
export function mouthProbePosition(
  extension: number,
  bodyHeight = 0,
): [number, number, number] {
  const angles = feedingJointAngles(extension);
  // Both anatomical pitch axes are [-1, 0, 0], hence the negated X angles.
  const tip = rotateX(TIP_PROBE, -angles.tip);
  const rostrum = rotateX(
    [tip[0] + TIP_OFFSET[0], tip[1] + TIP_OFFSET[1], tip[2] + TIP_OFFSET[2]],
    -angles.rostrum,
  );
  const body = rotateX(
    [
      rostrum[0] + ROSTRUM_PIVOT[0],
      rostrum[1] + ROSTRUM_PIVOT[1],
      rostrum[2] + ROSTRUM_PIVOT[2],
    ],
    FEEDING_BODY_POSE.pitch,
  );
  body[1] += FEEDING_BODY_POSE.height + bodyHeight;
  return body;
}

export function foodPosition(distance: number): [number, number, number] {
  return [
    FOOD_ORIGIN[0],
    FOOD_ORIGIN[1],
    FOOD_ORIGIN[2] - clamp(distance, 0, 0.7),
  ];
}

/**
 * Approximate contact between one anatomical mouth probe and the visible food
 * ellipsoid. Inside means full contact; a 0.04-unit radial boundary softens loss
 * of contact. This is neither measured taste transduction nor surface physics.
 */
export function feedingContact(extension: number, distance: number, bodyHeight = 0): number {
  const probe = mouthProbePosition(extension, bodyHeight),
    food = foodPosition(distance);
  const delta = probe.map((value, axis) => value - food[axis]);
  const normalized = Math.hypot(
    ...delta.map((value, axis) => value / FOOD_RADII[axis]),
  );
  if (normalized <= 1) return 1;
  const radialGap = Math.hypot(...delta) * (1 - 1 / normalized);
  return clamp(1 - radialGap / SOFT_CONTACT_DISTANCE, 0, 1);
}
