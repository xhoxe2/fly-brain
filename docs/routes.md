# Project routes

- Active data, binary formats and scientific scope: `docs/architecture.md`
- Whole-brain preparation and provenance: `tools/data-processing/prepare_dynamics.py`, `public/dynamics/provenance.json`
- Full FlyWire 630 graph and exact feeding IDs: `public/dynamics/manifest.json`, `public/dynamics/feeding.json`
- LIF core and worker: `src/neural.ts`, `src/brain.worker.ts`
- Continuous contact/sensory/motor feedback: `src/embodiment.ts`, `src/feeding-geometry.ts`
- Approximate GF-driven vertical body mechanics and checks: `src/body.ts`, `src/body.test.ts`, `src/body-rig.test.ts`
- Approaching finger sensory encoder and checks: `src/finger.ts`, `src/finger.test.ts`, `tools/check-finger.ts`
- Exact visual inputs/GF readouts: `public/dynamics/finger.json`, `tools/data-processing/prepare_finger.py`
- Finger illustration and CC0 provenance: `public/finger/NOTICE`, `tools/data-processing/prepare_finger_asset.py`
- Graph and anchor loading: `src/neural-data.ts`
- Numerical and feedback tests: `src/neural.test.ts`, `src/neural-data.test.ts`, `src/embodiment.test.ts`, `tools/check-feeding.ts`, `tools/data-processing/check_neural_reference.py`
- Runtime and environment controls: `src/main.ts`, `src/style.css`
- Guided environmental sequence and timing check: `src/introduction.ts`, `src/introduction.test.ts`
- Batched brain activity and fly rendering: `src/scene.ts`
- Anatomical fly mesh and provenance: `tools/data-processing/prepare_fly.py`, `public/fly/NOTICE`
- Run, validation, payload and limitations: `README.md`
- Legacy Male CNS circuit model/assets: `src/simulation.ts`, `src/simulation.worker.ts`, `src/data.ts`, `public/brain/`, `tools/data-processing/prepare.py`, `tools/data-processing/prepare_visual_circuits.py`
