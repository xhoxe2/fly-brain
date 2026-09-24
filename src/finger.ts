import type { FingerProtocol } from "./neural-data.ts";
import type { Drive } from "./neural.ts";

export type FingerPoint = [number, number, number];
export interface FingerEnvironment {
  enabled: boolean;
  lateral: number;
  proximity: number;
}
export interface FingerState {
  position: FingerPoint | null;
  sensoryRates: [number, number, number, number];
  gfRates: [number, number];
}
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
// Geometry uses the displayed standing fly; anatomical left is negative world X.
const EYES: readonly FingerPoint[] = [[-0.242, 0.943, -0.999], [0.242, 0.943, -0.999]];
const PAD_RADIUS = 0.3;
const SPEED = 1.5; // scene units per model second; an environmental trajectory
export const FINGER_GROUPS = ["lc4Left", "lc4Right", "lplc2Left", "lplc2Right"] as const;

export function validateFinger(input: FingerEnvironment | undefined): void {
  if (input && (typeof input.enabled !== "boolean" ||
      ![input.lateral, input.proximity].every(Number.isFinite)))
    throw new RangeError("Finger conditions must be finite");
}

export function fingerTarget(input: FingerEnvironment): FingerPoint {
  return [clamp(input.lateral, -1, 1) * 1.4, 0.95, -4 + clamp(input.proximity, 0, 1) * 2.35];
}

function opticalSize(tip: FingerPoint, side: number, bodyHeight: number) {
  const eye = EYES[side];
  const dx = tip[0] - eye[0], dy = tip[1] - eye[1] - bodyHeight, dz = tip[2] - PAD_RADIUS - eye[2];
  const distance = Math.hypot(dx, dy, dz);
  // Spherical pad approximation of the visible fingertip. The broad outward
  // eye axes are authored binocular weights, not measured receptive fields.
  const weight = Math.max(0, ((side ? 1 : -1) * Math.sqrt(3) / 2 * dx - 0.5 * dz) / distance);
  return { angle: 2 * Math.asin(Math.min(1, PAD_RADIUS / distance)), weight };
}

/** Feature-level sensory adapter, NOT a retina simulation or a behavior policy.
 * LC4 gets positive angular expansion; LPLC2 gets size gated by expansion.
 * Rate gains, pad shape and population-wide input are authored approximations.
 * This sensory adapter never injects GF spikes or supplies a motor command.
 */
export class FingerStimulus {
  readonly drives: Drive[];
  private readonly outputs: number[];
  private position: FingerPoint | null = null;
  private gfRates: [number, number] = [0, 0];
  private previousBodyHeight = 0;

  constructor(protocol: FingerProtocol, count: number) {
    const valid = (i: number) => Number.isInteger(i) && i >= 0 && i < count;
    this.drives = FINGER_GROUPS.map(name => ({ indices: protocol.inputs[name].indices.slice(), rateHz: 0 }));
    this.outputs = [protocol.outputs.gfLeft.index, protocol.outputs.gfRight.index];
    const inputs = this.drives.flatMap(drive => [...drive.indices]);
    if (this.drives.some(d => !d.indices.length) || !inputs.every(valid) ||
        new Set(inputs).size !== inputs.length || !this.outputs.every(valid) ||
        this.outputs[0] === this.outputs[1] || this.outputs.some(i => inputs.includes(i)))
      throw new RangeError("Invalid visual sensory cells or GF readouts");
  }

  reset() {
    this.position = null;
    this.gfRates = [0, 0];
    this.previousBodyHeight = 0;
    this.drives.forEach(d => d.rateHz = 0);
  }

  advance(input: FingerEnvironment | undefined, dtMs: number, bodyHeight = 0): readonly Drive[] {
    validateFinger(input);
    if (!Number.isFinite(dtMs) || dtMs < 0) throw new RangeError("Invalid sensory interval");
    if (!Number.isFinite(bodyHeight) || bodyHeight < 0) throw new RangeError("Invalid body height");
    if (dtMs === 0) return this.drives.filter(drive => drive.rateHz > 0);
    this.drives.forEach(d => d.rateHz = 0);
    if (!input?.enabled) {
      this.position = null;
      this.previousBodyHeight = bodyHeight;
      return [];
    }
    const target = fingerTarget(input);
    // Insertion starts at the far plane; visibility alone is not a looming cue.
    if (!this.position) {
      this.position = [target[0], target[1], -4];
      this.previousBodyHeight = bodyHeight;
    }
    const before = this.position.slice() as FingerPoint;
    const delta = target.map((value, i) => value - before[i]);
    const distance = Math.hypot(...delta);
    const fraction = distance ? Math.min(1, SPEED * dtMs / 1000 / distance) : 0;
    this.position = before.map((value, i) => value + delta[i] * fraction) as FingerPoint;
    for (let side = 0; side < 2; side++) {
      const previous = opticalSize(before, side, this.previousBodyHeight), current = opticalSize(this.position, side, bodyHeight);
      const expansion = Math.max(0, (current.angle - previous.angle) * 1000 / dtMs);
      this.drives[side].rateHz = 100 * Math.min(1, expansion / 0.8) * current.weight;
      this.drives[side + 2].rateHz = 100 * Math.min(1, current.angle / 0.8) *
        Math.min(1, expansion / 0.08) * current.weight;
    }
    this.previousBodyHeight = bodyHeight;
    // Zero drive leaves these cells' ordinary refractory dynamics unchanged.
    return this.drives.filter(drive => drive.rateHz > 0);
  }

  observe(counts: Uint16Array, dtMs: number) {
    if (!Number.isFinite(dtMs) || dtMs < 0) throw new RangeError("Invalid readout interval");
    if (dtMs === 0) return;
    const decay = Math.exp(-dtMs / 120);
    this.outputs.forEach((index, side) => {
      this.gfRates[side] = this.gfRates[side] * decay + counts[index] * 1000 / dtMs * (1 - decay);
    });
  }

  get state(): FingerState {
    return {
      position: this.position ? [...this.position] : null,
      sensoryRates: this.drives.map(d => d.rateHz) as FingerState["sensoryRates"],
      gfRates: [...this.gfRates],
    };
  }
}
