export interface BodyState {
  height: number;
  velocity: number;
  activation: [number, number];
  legExtension: [number, number];
  grounded: boolean;
  peakHeight: number;
  takeoffs: number;
}

/** Phenomenological GF-to-muscle bypass of the absent thoracic motor network.
 * All mechanical parameters are authored in scene units, not biological SI.
 * A vertical constraint deliberately omits steering, flight and body rotation.
 * No stimulus, action timer, animation curve or jump command enters this class.
 */
export class EscapeBody {
  private excitation: [number, number] = [0, 0];
  private activation: [number, number] = [0, 0];
  private stroke: [number, number] = [0, 0];
  private height = 0;
  private velocity = 0;
  private peakHeight = 0;
  private takeoffs = 0;
  private grounded = true;

  reset() {
    this.excitation = [0, 0];
    this.activation = [0, 0];
    this.stroke = [0, 0];
    this.height = this.velocity = this.peakHeight = this.takeoffs = 0;
    this.grounded = true;
  }

  /** Counts are real GF spikes in the preceding 10 ms neural interval.
   * Timing within that bin is not reconstructed. The HUD's smoothed Hz is not
   * used for actuation. Ground forces are integrated at 1 ms, even during rest.
   */
  advance(spikes: readonly [number, number], durationMs: number): BodyState {
    if (!Number.isInteger(durationMs) || durationMs < 0 || durationMs > 1000 ||
        !spikes.every(n => Number.isSafeInteger(n) && n >= 0 && n <= 65535))
      throw new RangeError("Invalid motor interval");
    if (!durationMs) return this.state;
    for (let side = 0; side < 2; side++)
      this.excitation[side] += spikes[side];

    const dt = 0.001;
    const gravity = 20, stiffness = 1200;
    for (let step = 0; step < durationMs; step++) {
      let support = 0;
      for (let side = 0; side < 2; side++) {
        // Summed excitation saturates contraction under a train of spikes;
        // it does not restart an independent leg stroke for every impulse.
        this.excitation[side] *= Math.exp(-dt / 0.2);
        this.activation[side] = 1 - Math.exp(-3 * this.excitation[side]);
        this.stroke[side] += (0.3 * this.activation[side] - this.stroke[side]) * (1 - Math.exp(-dt / 0.008));
        // A leg can push against the platform but cannot pull it or propel the
        // airborne body. Passive spring/damper forces dissipate landing energy.
        if (this.height <= this.stroke[side]) {
          const damping = this.velocity < 0 ? 140 : 28;
          support += Math.max(0, (gravity + stiffness * (this.stroke[side] - this.height) - damping * this.velocity) / 2);
        }
      }
      this.velocity += (support - gravity) * dt;
      this.height += this.velocity * dt;
      if (this.height < 0) {
        this.height = 0;
        this.velocity = Math.max(0, this.velocity);
      }
      // The small contact tolerance avoids counting numerical micro-separations
      // as takeoffs. It affects reporting only, never force or movement.
      const grounded = this.height <= Math.max(...this.stroke) + 0.01;
      if (this.grounded && !grounded && this.velocity > 0) this.takeoffs++;
      this.grounded = grounded;
      this.peakHeight = Math.max(this.peakHeight, this.height);
    }
    return this.state;
  }

  get state(): BodyState {
    return {
      height: this.height,
      velocity: this.velocity,
      activation: [...this.activation],
      // Joint compliance keeps a supporting foot on the platform. In flight
      // it follows its freely relaxing actuator; no renderer-only floor snap.
      legExtension: this.stroke.map(length => Math.min(this.height, length)) as [number, number],
      grounded: this.grounded,
      peakHeight: this.peakHeight,
      takeoffs: this.takeoffs,
    };
  }
}
