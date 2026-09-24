import type { FeedingProtocol, FingerProtocol } from "./neural-data.ts";
import type { Drive, SpikingBrain } from "./neural.ts";
import { feedingContact } from "./feeding-geometry.ts";
import { FingerStimulus, validateFinger, type FingerEnvironment, type FingerState } from "./finger.ts";
import { EscapeBody, type BodyState } from "./body.ts";

export interface FeedingEnvironment {
  sugar: number;
  bitter: number;
  foodDistance: number;
  finger?: FingerEnvironment;
}

export interface FeedingState {
  timeMs: number;
  counts: Uint32Array;
  spikes: number;
  motorRates: [number, number];
  motorRate: number;
  extension: number;
  contact: number;
  sensoryRates: { sugar: number; bitter: number };
  finger: FingerState | null;
  body: BodyState;
}

const FEEDBACK_MS = 10;
const RATE_DECAY = Math.exp(-FEEDBACK_MS / 120);
const MUSCLE_DECAY = Math.exp(-FEEDBACK_MS / 60);
const clamp = (value: number, low: number, high: number) =>
  Math.max(low, Math.min(high, value));

/**
 * Approximate mouth/environment adapter, not published
 * whole-animal dynamics. Contact encodes taste as sensory Poisson rates; only
 * computed MN9 spikes drive a filtered muscle activation and mouth extension.
 * The point-contact geometry, rate encoding and actuator transfer are authored
 * assumptions. There is no navigation, stimulus-to-action policy or trial reset.
 */
export class FeedingWorld {
  private readonly brain: SpikingBrain;
  private readonly motors: [number, number];
  private readonly drives: [Drive, Drive];
  private motorRates: [number, number] = [0, 0];
  private extension = 0;
  private remainderMs = 0;
  private readonly finger: FingerStimulus | null;
  private readonly body = new EscapeBody();
  private readonly giantFibers: [number, number] | null;

  constructor(brain: SpikingBrain, protocol: FeedingProtocol, fingerProtocol?: FingerProtocol) {
    this.brain = brain;
    const left = protocol.outputs.primary.index;
    const right = protocol.outputs.additional[0]?.index;
    const validIndex = (index: number) =>
      Number.isInteger(index) && index >= 0 && index < brain.voltage.length;
    if (!validIndex(left) || !validIndex(right) || left === right)
      throw new RangeError("Feeding protocol requires two distinct MN9 cells");
    this.motors = [left, right];
    const drive = (name: "sugar" | "bitter"): Drive => {
      const indices = protocol.inputs[name].indices.slice();
      if (!indices.length || !indices.every(validIndex))
        throw new RangeError("Feeding protocol has invalid sensory cells");
      return { indices, rateHz: 0 };
    };
    this.drives = [drive("sugar"), drive("bitter")];
    this.finger = fingerProtocol ? new FingerStimulus(fingerProtocol, brain.voltage.length) : null;
    this.giantFibers = fingerProtocol ? [fingerProtocol.outputs.gfLeft.index, fingerProtocol.outputs.gfRight.index] : null;
  }

  reset(seed = 1): void {
    this.brain.reset(seed);
    this.motorRates = [0, 0];
    this.extension = 0;
    this.remainderMs = 0;
    this.finger?.reset();
    this.body.reset();
    for (const drive of this.drives) drive.rateHz = 0;
  }

  /**
   * Retains sub-tick time and samples the supplied environment at each completed
   * 10 ms feedback tick, independently of rendering. Changing the environment
   * preserves all neural and muscle state. Returned contact/rates describe the
   * current geometry, ready for the next neural tick; counts cover this call.
   */
  advance(durationMs: number, environment: FeedingEnvironment): FeedingState {
    if (!Number.isFinite(durationMs) || durationMs < 0)
      throw new RangeError("Duration must be finite and nonnegative");
    if (
      ![environment.sugar, environment.bitter, environment.foodDistance].every(
        Number.isFinite,
      )
    )
      throw new RangeError("Environment values must be finite");
    validateFinger(environment.finger);
    const sugar = clamp(environment.sugar, 0, 1);
    const bitter = clamp(environment.bitter, 0, 1);
    const distance = clamp(environment.foodDistance, 0, 0.7);
    const accumulated = this.remainderMs + durationMs;
    const ticks = Math.floor(accumulated / FEEDBACK_MS + 1e-9);
    if (!Number.isSafeInteger(ticks))
      throw new RangeError("Duration exceeds the simulation clock range");
    const remainder = accumulated - ticks * FEEDBACK_MS;
    this.remainderMs = Math.abs(remainder) < 1e-8 ? 0 : remainder;

    const counts = new Uint32Array(this.brain.voltage.length);
    let spikes = 0;
    for (let tick = 0; tick < ticks; tick++) {
      const contact = this.contact(distance);
      this.drives[0].rateHz = sugar * contact * 200;
      this.drives[1].rateHz = bitter * contact * 200;
      const visualDrives = this.finger?.advance(environment.finger, FEEDBACK_MS, this.body.state.height) ?? [];
      const interval = this.brain.advance(FEEDBACK_MS, visualDrives.length ? [...this.drives, ...visualDrives] : this.drives);
      this.finger?.observe(interval.counts, FEEDBACK_MS);
      for (let i = 0; i < counts.length; i++) counts[i] += interval.counts[i];
      spikes += interval.spikes;

      // The actuator receives neural output alone, never environment labels.
      for (let side = 0; side < 2; side++) {
        const rate = (interval.counts[this.motors[side]] * 1000) / FEEDBACK_MS;
        this.motorRates[side] =
          this.motorRates[side] * RATE_DECAY + rate * (1 - RATE_DECAY);
      }
      const motorRate = (this.motorRates[0] + this.motorRates[1]) / 2;
      const activation = motorRate / (motorRate + 20);
      this.extension =
        this.extension * MUSCLE_DECAY + activation * (1 - MUSCLE_DECAY);
      this.body.advance(this.giantFibers
        ? [interval.counts[this.giantFibers[0]], interval.counts[this.giantFibers[1]]]
        : [0, 0], FEEDBACK_MS);
    }

    const contact = this.contact(distance);
    return {
      timeMs: this.brain.timeMs,
      counts,
      spikes,
      motorRates: [...this.motorRates],
      motorRate: (this.motorRates[0] + this.motorRates[1]) / 2,
      extension: this.extension,
      contact,
      sensoryRates: {
        sugar: sugar * contact * 200,
        bitter: bitter * contact * 200,
      },
      finger: this.finger?.state ?? null,
      body: this.body.state,
    };
  }

  private contact(distance: number): number {
    return feedingContact(this.extension, distance, this.body.state.height);
  }
}
