/** Compressed sparse rows of signed synapse counts, indexed by presynaptic cell. */
export interface Network {
  count: number;
  offsets: Uint32Array;
  targets: Uint32Array;
  weights: Float32Array;
}

/** Each target receives an independent N=1 PoissonInput, as in Shiu et al. */
export interface Drive {
  indices: readonly number[];
  rateHz: number;
}

/** Deterministic replacement for Poisson events, on the absolute 0.1 ms grid. */
export interface ExplicitInput {
  step: number;
  index: number;
}

export interface IntervalActivity {
  timeMs: number;
  counts: Uint16Array;
  spikes: number;
}

export const NEURAL_DT_MS = 0.1;

// Shiu's published whole-brain LIF parameterization; units here are ms and mV.
// These are model assumptions, not measured individual-cell electrophysiology.
const REST = -52;
const THRESHOLD = -45;
const REFRACTORY_STEPS = 22;
const DELAY_STEPS = 18;
const SYNAPSE_MV = 0.275;
const INPUT_MV = SYNAPSE_MV * 250;
const MEMBRANE_DECAY = Math.exp(-NEURAL_DT_MS / 20);
const CURRENT_DECAY = Math.exp(-NEURAL_DT_MS / 5);
const CURRENT_TO_VOLTAGE = (MEMBRANE_DECAY - CURRENT_DECAY) / 3;
const MAX_INTERVAL_STEPS = 65_535;
// Detection cadence changes CPU work only; sleeping states are exact fixed points.
const FIXED_POINT_SCAN_STEPS = 100;

/**
 * Fixed-step port of the Shiu model's linear LIF equations and Brian2 schedule.
 * Network arrays and public probe arrays must not be mutated by the caller.
 * No baseline noise, behavioral policy, or extra connections are introduced.
 */
export class SpikingBrain {
  readonly voltage: Float64Array;
  readonly current: Float64Array;

  private readonly network: Network;
  private readonly lastSpike: Float64Array;
  private readonly driven: Uint8Array;
  private readonly silenced: Uint8Array;
  private readonly active: Uint32Array;
  private readonly activePosition: Int32Array;
  private readonly fired: number[] = [];
  private readonly pending: number[][] = Array.from(
    { length: DELAY_STEPS + 1 },
    () => [],
  );
  private readonly drivenIndices: number[] = [];
  private readonly silencedIndices: number[] = [];
  private activeCount = 0;
  private step = 0;
  private fractionalSteps = 0;
  private randomState = 1;

  constructor(network: Network, seed = 1) {
    validateNetwork(network);
    this.network = network;
    this.voltage = new Float64Array(network.count);
    this.current = new Float64Array(network.count);
    this.lastSpike = new Float64Array(network.count);
    this.driven = new Uint8Array(network.count);
    this.silenced = new Uint8Array(network.count);
    this.active = new Uint32Array(network.count);
    this.activePosition = new Int32Array(network.count);
    this.reset(seed);
  }

  get timeMs(): number {
    return this.step * NEURAL_DT_MS;
  }

  reset(seed = 1): void {
    if (!Number.isSafeInteger(seed))
      throw new RangeError("Seed must be an integer");
    this.randomState = seed >>> 0 || 0x9e3779b9;
    this.voltage.fill(REST);
    this.current.fill(0);
    this.lastSpike.fill(-Infinity);
    this.driven.fill(0);
    this.silenced.fill(0);
    this.activePosition.fill(-1);
    this.activeCount = 0;
    this.step = 0;
    this.fractionalSteps = 0;
    this.fired.length = 0;
    this.drivenIndices.length = 0;
    this.silencedIndices.length = 0;
    for (const bucket of this.pending) bucket.length = 0;
  }

  /**
   * Advance whole 0.1 ms steps, retaining a fractional remainder across calls.
   * A call is limited to 6,553.5 ms so interval Uint16 counts cannot overflow.
   * Silencing removes outgoing transmission, including already queued spikes.
   * Supplying explicitInputs disables Poisson sampling; events must be sorted by
   * absolute step. Include their targets in drives (rate 0 is valid) to give them
   * the reference model's zero refractory period for stimulated neurons.
   */
  advance(
    durationMs: number,
    drives: readonly Drive[],
    silenced: readonly number[] = [],
    explicitInputs?: readonly ExplicitInput[],
  ): IntervalActivity {
    if (!Number.isFinite(durationMs) || durationMs < 0)
      throw new RangeError("Duration must be finite and nonnegative");
    const accumulated = this.fractionalSteps + durationMs / NEURAL_DT_MS;
    const steps = Math.floor(accumulated + 1e-9);
    if (steps > MAX_INTERVAL_STEPS || !Number.isSafeInteger(this.step + steps))
      throw new RangeError("Advance at most 6553.5 ms per call");
    this.validateInputs(drives, silenced, explicitInputs);
    this.setMasks(drives, silenced);
    const remainder = accumulated - steps;
    this.fractionalSteps = Math.abs(remainder) < 1e-9 ? 0 : remainder;

    const counts = new Uint16Array(this.network.count);
    const end = this.step + steps;
    let spikes = 0;
    let inputCursor = 0;
    if (explicitInputs) {
      // Reusing the same absolute fixture for multiple advances must not replay it.
      let high = explicitInputs.length;
      while (inputCursor < high) {
        const mid = Math.floor((inputCursor + high) / 2);
        if (explicitInputs[mid].step < this.step) inputCursor = mid + 1;
        else high = mid;
      }
    }

    for (; this.step < end; this.step++) {
      this.fired.length = 0;

      // Brian2 groups: exact coupled linear update, with both v and g clamped
      // during refractoriness. Fixed-point detection is outside this hot loop.
      for (let slot = 0; slot < this.activeCount; slot++) {
        const index = this.active[slot];
        if (this.isRefractory(index)) continue;
        const oldCurrent = this.current[index];
        this.voltage[index] =
          REST +
          (this.voltage[index] - REST) * MEMBRANE_DECAY +
          oldCurrent * CURRENT_TO_VOLTAGE;
        this.current[index] = oldCurrent * CURRENT_DECAY;

        // Brian2 thresholds run after all groups. These independent cells can be
        // checked here because all synaptic writes happen in the next phase.
        if (this.voltage[index] > THRESHOLD) {
          this.fired.push(index);
          this.lastSpike[index] = this.step;
          counts[index]++;
          spikes++;
        }
      }

      // Brian emits threshold events in cell-index order. Keep that order for
      // deterministic accumulation when several sources share a target.
      this.fired.sort((a, b) => a - b);
      const later =
        this.pending[(this.step + DELAY_STEPS) % this.pending.length];
      for (const source of this.fired) later.push(source);

      // Brian2 synapses: on_pre uses the current outgoing weight at delivery.
      // In Brian2, `(unless refractory)` also makes g read-only to synaptic
      // writes during the refractory window (not just frozen during groups).
      const due = this.pending[this.step % this.pending.length];
      for (const source of due) {
        if (this.silenced[source]) continue;
        for (
          let edge = this.network.offsets[source];
          edge < this.network.offsets[source + 1];
          edge++
        ) {
          const target = this.network.targets[edge];
          if (this.isRefractory(target)) continue;
          const increment = this.network.weights[edge] * SYNAPSE_MV;
          if (increment !== 0) {
            this.current[target] += increment;
            this.activate(target);
          }
        }
      }
      due.length = 0;

      if (explicitInputs) {
        while (
          inputCursor < explicitInputs.length &&
          explicitInputs[inputCursor].step === this.step
        ) {
          this.input(explicitInputs[inputCursor++].index);
        }
      } else {
        // N=1 PoissonInput is a Bernoulli draw per target per step. The seeded
        // generator is reproducible, but does not duplicate NumPy's RNG stream.
        for (const drive of drives) {
          if (drive.rateHz === 0) continue;
          const probability = (drive.rateHz * NEURAL_DT_MS) / 1000;
          for (const index of drive.indices) {
            if (this.random() < probability) this.input(index);
          }
        }
      }

      // Brian2 resets happen last, discarding same-step input to firing cells.
      for (const index of this.fired) {
        this.voltage[index] = REST;
        this.current[index] = 0;
      }
      for (let slot = this.activeCount - 1; slot >= 0; slot--) {
        const index = this.active[slot];
        if (this.voltage[index] === REST && this.current[index] === 0) {
          const replacement = this.active[--this.activeCount];
          this.active[slot] = replacement;
          this.activePosition[replacement] = slot;
          this.activePosition[index] = -1;
        }
      }
      if ((this.step + 1) % FIXED_POINT_SCAN_STEPS === 0)
        this.sleepFixedPoints();
    }
    return { timeMs: this.timeMs, counts, spikes };
  }

  private isRefractory(index: number): boolean {
    return (
      !this.driven[index] &&
      this.step - this.lastSpike[index] < REFRACTORY_STEPS
    );
  }

  private activate(index: number): void {
    if (this.activePosition[index] !== -1) return;
    this.activePosition[index] = this.activeCount;
    this.active[this.activeCount++] = index;
  }

  private sleepFixedPoints(): void {
    // A non-firing, non-refractory cell whose NEXT full recurrence leaves both
    // stored values exactly unchanged can sleep until the next incoming write.
    // Keep every residual bit: no epsilon cutoff or decay truncation. Scanning
    // less often only performs some redundant updates before finding a fixed
    // point; it never changes the simulation's fixed 0.1 ms integration step.
    for (let slot = this.activeCount - 1; slot >= 0; slot--) {
      const index = this.active[slot];
      const voltage = this.voltage[index], current = this.current[index];
      if (this.isRefractory(index) || voltage > THRESHOLD ||
          current * CURRENT_DECAY !== current) continue;
      if (REST + (voltage - REST) * MEMBRANE_DECAY +
          current * CURRENT_TO_VOLTAGE !== voltage) continue;
      const replacement = this.active[--this.activeCount];
      this.active[slot] = replacement;
      this.activePosition[replacement] = slot;
      this.activePosition[index] = -1;
    }
  }

  private input(index: number): void {
    if (this.isRefractory(index)) return;
    this.voltage[index] += INPUT_MV;
    this.activate(index);
  }

  private random(): number {
    let value = this.randomState;
    value ^= value << 13;
    value ^= value >>> 17;
    value ^= value << 5;
    this.randomState = value >>> 0;
    return this.randomState / 0x1_0000_0000;
  }

  private validateInputs(
    drives: readonly Drive[],
    silenced: readonly number[],
    explicitInputs?: readonly ExplicitInput[],
  ): void {
    for (const drive of drives) {
      if (
        !Number.isFinite(drive.rateHz) ||
        drive.rateHz < 0 ||
        drive.rateHz > 1000 / NEURAL_DT_MS
      )
        throw new RangeError("Input rate must be between 0 and 10000 Hz");
      for (const index of drive.indices) this.validateIndex(index);
    }
    for (const index of silenced) this.validateIndex(index);
    let previous = -1;
    if (explicitInputs)
      for (const event of explicitInputs) {
        this.validateIndex(event.index);
        if (
          !Number.isSafeInteger(event.step) ||
          event.step < previous ||
          event.step < 0
        )
          throw new RangeError(
            "Explicit inputs need sorted, nonnegative integer steps",
          );
        previous = event.step;
      }
  }

  private validateIndex(index: number): void {
    if (!Number.isInteger(index) || index < 0 || index >= this.network.count)
      throw new RangeError("Neuron index outside network");
  }

  private setMasks(
    drives: readonly Drive[],
    silenced: readonly number[],
  ): void {
    for (const index of this.drivenIndices) this.driven[index] = 0;
    for (const index of this.silencedIndices) this.silenced[index] = 0;
    this.drivenIndices.length = 0;
    this.silencedIndices.length = 0;
    for (const drive of drives)
      for (const index of drive.indices)
        if (!this.driven[index]) {
          this.driven[index] = 1;
          this.drivenIndices.push(index);
        }
    for (const index of silenced)
      if (!this.silenced[index]) {
        this.silenced[index] = 1;
        this.silencedIndices.push(index);
      }
  }
}

function validateNetwork(network: Network): void {
  if (!Number.isInteger(network.count) || network.count < 1)
    throw new RangeError("Network must contain a positive number of neurons");
  if (
    network.offsets.length !== network.count + 1 ||
    network.targets.length !== network.weights.length ||
    network.offsets[0] !== 0 ||
    network.offsets[network.count] !== network.targets.length
  )
    throw new RangeError("Invalid CSR dimensions");
  for (let index = 0; index < network.count; index++)
    if (network.offsets[index] > network.offsets[index + 1])
      throw new RangeError("CSR offsets must be monotonic");
  for (let edge = 0; edge < network.targets.length; edge++)
    if (
      network.targets[edge] >= network.count ||
      !Number.isFinite(network.weights[edge])
    )
      throw new RangeError("Invalid synaptic target or weight");
}
