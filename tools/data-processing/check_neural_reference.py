#!/usr/bin/env python3
"""Offline numerical check of the browser LIF core against Brian2 2.10.1.

Run with .venv/bin/python tools/data-processing/check_neural_reference.py.
Uses tiny synthetic graphs, not anatomical claims or behavioral validation.
Fixed input events separate numerical parity from the different seeded RNGs.
No data download, web service, or browser is involved.
"""

import json
from pathlib import Path
import subprocess

import brian2 as b
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DT_MS = 0.1
CASES = [
    {"name": "rest", "count": 3, "edges": [[0, 1, 1000]], "driven": [], "events": [], "steps": 60},
    {
        "name": "delay, refractory writes and refractory boundary",
        "count": 2,
        "edges": [[0, 1, 1000]],
        "driven": [0],
        "events": [[0, 0], [10, 0], [25, 0], [28, 0]],
        "steps": 80,
    },
    {
        "name": "signed converging inputs",
        "count": 3,
        "edges": [[0, 2, 1000], [1, 2, -1000]],
        "driven": [0, 1],
        "events": [[0, 0], [0, 1], [20, 1]],
        "steps": 80,
    },
    {
        "name": "outgoing lesion preserves sensor spiking",
        "count": 3,
        "edges": [[0, 2, 1000], [1, 2, -1000]],
        "driven": [0, 1],
        "silenced": [1],
        "events": [[0, 0], [0, 1]],
        "steps": 60,
    },
    {
        "name": "same-step input erased by reset",
        "count": 2,
        "edges": [[0, 1, 1000]],
        "driven": [0],
        "events": [[0, 0], [1, 0], [2, 0]],
        "steps": 60,
    },
    {
        "name": "deterministic N=1 PoissonInput at one event per step",
        "count": 2,
        "edges": [[0, 1, 1000]],
        "driven": [0],
        "events": [],
        "poisson": True,
        "steps": 80,
    },
]


def reference(case):
    b.start_scope()
    b.defaultclock.dt = DT_MS * b.ms
    b.prefs.codegen.target = "numpy"
    constants = {
        "v_0": -52 * b.mV,
        "v_rst": -52 * b.mV,
        "v_th": -45 * b.mV,
        "t_mbr": 20 * b.ms,
        "tau": 5 * b.ms,
    }
    # Equations, threshold, reset and linear integration match the inspected
    # Shiu model.py. Its reset-local w=0 has no stored neuronal-state effect.
    neurons = b.NeuronGroup(
        case["count"],
        """
        dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
        dg/dt = -g / tau : volt (unless refractory)
        rfc : second
        """,
        threshold="v > v_th",
        reset="v = v_rst; w = 0; g = 0 * mV",
        refractory="rfc",
        method="linear",
        namespace=constants,
    )
    neurons.v = -52 * b.mV
    neurons.g = 0 * b.mV
    neurons.rfc = 2.2 * b.ms
    neurons.rfc[np.asarray(case["driven"], dtype=int)] = 0 * b.ms
    synapses = b.Synapses(neurons, neurons, "w : volt", on_pre="g += w", delay=1.8 * b.ms)
    source, target, weights = zip(*case["edges"])
    synapses.connect(i=source, j=target)
    # Browser asset weights are float32 signed counts before promotion to f64.
    signed_counts = np.asarray(weights, dtype=np.float32).astype(np.float64)
    synapses.w = signed_counts * 0.275 * b.mV
    for index in case.get("silenced", []):
        synapses.w[np.flatnonzero(np.asarray(source) == index)] = 0 * b.mV

    objects = [neurons, synapses]
    if case.get("poisson"):
        for index in case["driven"]:
            objects.append(b.PoissonInput(neurons[index], "v", N=1, rate=10000 * b.Hz, weight=68.75 * b.mV))
    else:
        events = case["events"]
        stimulus = b.SpikeGeneratorGroup(
            case["count"],
            np.array([event[1] for event in events], dtype=int),
            np.array([event[0] * DT_MS for event in events]) * b.ms,
        )
        stimulus_synapses = b.Synapses(stimulus, neurons, on_pre="v += 68.75 * mV")
        stimulus_synapses.connect(j="i")
        objects.extend([stimulus, stimulus_synapses])

    trace = b.StateMonitor(neurons, ["v", "g"], record=True, when="end")
    spikes = b.SpikeMonitor(neurons)
    b.Network(*objects, trace, spikes).run(case["steps"] * DT_MS * b.ms)
    counts = np.zeros((case["steps"], case["count"]), dtype=np.uint16)
    for index, time in zip(np.asarray(spikes.i), np.asarray(spikes.t / b.ms)):
        counts[round(time / DT_MS), index] += 1
    return {"voltage": np.asarray(trace.v / b.mV).T, "current": np.asarray(trace.g / b.mV).T, "counts": counts}


def browser_core(cases):
    program = r"""
import { readFileSync } from 'node:fs';
import { SpikingBrain } from './src/neural.ts';
const cases = JSON.parse(readFileSync(0, 'utf8'));
const output = cases.map(test => {
  const edges = test.edges.toSorted((a, b) => a[0] - b[0]);
  const offsets = new Uint32Array(test.count + 1);
  for (const [source] of edges) offsets[source + 1]++;
  for (let i = 1; i <= test.count; i++) offsets[i] += offsets[i - 1];
  const brain = new SpikingBrain({count:test.count, offsets,
    targets: Uint32Array.from(edges.map(e => e[1])),
    weights: Float32Array.from(edges.map(e => e[2]))});
  const events = test.events.map(([step, index]) => ({step, index}));
  const drives = [{indices:test.driven, rateHz:test.poisson ? 10000 : 0}];
  const output = {voltage:[], current:[], counts:[]};
  for (let step = 0; step < test.steps; step++) {
    const result = brain.advance(.1, drives, test.silenced ?? [], test.poisson ? undefined : events);
    output.voltage.push([...brain.voltage]);
    output.current.push([...brain.current]);
    output.counts.push([...result.counts]);
  }
  return output;
});
process.stdout.write(JSON.stringify(output));
"""
    process = subprocess.run(
        ["node", "--experimental-strip-types", "--input-type=module", "-e", program],
        input=json.dumps(cases), text=True, capture_output=True, check=True, cwd=ROOT,
    )
    return json.loads(process.stdout)


def main():
    if b.__version__ != "2.10.1":
        raise RuntimeError(f"Use the pinned Brian2 2.10.1 reference, found {b.__version__}")
    implementations = browser_core(CASES)
    for case, actual in zip(CASES, implementations):
        expected = reference(case)
        np.testing.assert_array_equal(actual["counts"], expected["counts"], err_msg=case["name"])
        errors = []
        for state in ["voltage", "current"]:
            observed = np.asarray(actual[state])
            np.testing.assert_allclose(observed, expected[state], rtol=1e-12, atol=2e-10, err_msg=f'{case["name"]}: {state}')
            errors.append(float(np.max(np.abs(observed - expected[state]))))
        print(f'PASS {case["name"]}: {int(expected["counts"].sum())} spikes; max error {max(errors):.3g} mV')
    print(f"Brian2 {b.__version__}: all {len(CASES)} fixed-step reference checks passed.")


if __name__ == "__main__":
    main()
