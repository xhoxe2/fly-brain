/** A sequence of environmental changes. It never commands or resets the brain. */
export const TASTE_STEPS = [
  {
    title: "Sweet food",
    note: "Watch the mouth beneath the head. Only sweet taste is present.",
    bitter: 0,
  },
  {
    title: "Bitter added",
    note: "The food stays sweet. Watch how the mouth and motor signal change.",
    bitter: 1,
  },
  {
    title: "Sweet again",
    note: "Bitterness is removed. The same brain keeps running.",
    bitter: 0,
  },
] as const;

export class TasteIntroduction {
  step = 0;
  private elapsedMs = 0;
  private modelStartMs: number;

  constructor(modelTimeMs: number) {
    this.modelStartMs = modelTimeMs;
  }

  /** Call only while visible and playing. Slow devices get time to compute. */
  advance(visibleMs: number, modelTimeMs: number): boolean {
    if (this.done) return false;
    this.elapsedMs += visibleMs;
    if (this.progress(modelTimeMs) < 1) return false;
    this.step++;
    this.elapsedMs = 0;
    this.modelStartMs = modelTimeMs;
    return true;
  }

  progress(modelTimeMs: number): number {
    return Math.min(1, this.elapsedMs / 8000, (modelTimeMs - this.modelStartMs) / 1000);
  }

  get done(): boolean {
    return this.step === TASTE_STEPS.length;
  }
}
