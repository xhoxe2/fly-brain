# FLY//BRAIN architecture

Static Vite + TypeScript application. Three.js renders through WebGPU or WebGL2;
the whole-brain LIF model runs on the CPU in `src/brain.worker.ts`. Python is
used only for offline preparation and numerical verification. No backend,
credentials or scientific API calls are needed at runtime.

## Sources and identity

The active network is the author's **FlyWire 630 adult-female brain model**:
127,400 neurons and 14,687,178 signed directed connections. The source contains
52,793,639 synapse counts. `tools/data-processing/prepare_dynamics.py` preserves
every neuron, source edge, sign and weight; no cutoff, merging or sampling is
applied. Source row order defines the dense neuron indices.

The model repository is pinned to `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`
(MIT). FlyWire annotations are pinned to v1.1.0,
`df6bb136f5b3d91c3992df4e8de2642329e2a384` (CC BY 4.0).
Exact 630 IDs join 127,322 neurons to measured anchor points. The remaining 78
neurons are still simulated; only their point rendering is omitted. Anchor
coordinates are not necessarily somata. No Male CNS positions are reused.

Raw CSV, Parquet and TSV sources remain in ignored `.cache/brain-model`.
`public/dynamics/provenance.json` records URLs, revisions, licenses and SHA-256
hashes. Root IDs exceed JavaScript's safe integer range, so identity uses uint64
or decimal strings; simulation and rendering use dense uint32 indices.

## Loading and binary layouts

`src/neural-data.ts` reads `public/dynamics/manifest.json`. The graph is split
into 15 `.bin.gz` files; each decompresses to a contiguous byte range described
by `offset` and `bytes`. `compressedBytes` is the file size and `sha256` hashes
the **decompressed** bytes. The loader verifies each chunk before assembling
the graph. Native `DecompressionStream` handles gzip, while already-decoded
responses from hosts using `Content-Encoding` are also accepted.

The graph has little-endian layout:

| Byte offset | Type | Meaning |
| --- | --- | --- |
| 0 | 4 ASCII bytes | `LIF1` |
| 4 | uint32 | N = 127,400 |
| 8 | uint32 | E = 14,687,178 |
| 12 | uint32[N+1] | Presynaptic CSR row offsets |
| 12 + 4(N+1) | uint32[E] | Postsynaptic indices |
| 12 + 4(N+1) + 4E | float32[E] | Exact signed synapse counts, before 0.275 mV gain |

`ids.bin.gz` contains uint64[N] in the same neuron order. `positions.bin.gz`
contains float32[3N] XYZ followed by uint8[N] validity flags. Positions are source
4×4×40 nm voxel coordinates converted to nanometers; invalid rows contain zeros
and must be excluded using the mask. The display centers and uniformly scales the central 99.98% of valid
coordinates per axis with 5% padding and reverses the vertical axis. This prevents extreme
source outliers from moving the brain off screen; all valid points retain their
relative geometry and remain in the buffer, even outside the initial view; it does not register them to another
brain. `annotations.bin.gz` is a neuron-major uint16[N×8] categorical table,
with field names and dictionaries in the manifest.

`feeding.json` records the exact author notebook input IDs and indices under
`inputs.sugar` and `inputs.bitter`; MN9 readouts are `outputs.primary` and
`outputs.additional`. All 44 cells are present in the network and anchor table.
The primary published Figure 3b readout is MN9 left.

## Neural computation

`SpikingBrain` in `src/neural.ts` uses 0.1 ms steps and the exact coupled linear
update of the author's membrane/current equations. Rest/reset is −52 mV,
threshold is strictly above −45 mV, membrane decay is 20 ms, current decay is
5 ms, delay is 1.8 ms and refractoriness is 2.2 ms. Both state updates and
synaptic writes respect Brian2's refractory semantics. Stimulated cells have
zero refractory time and independent N=1 Poisson voltage inputs of 68.75 mV.

The full signed CSR graph stays in the worker. An active-index set skips cells
whose complete voltage/current recurrence leaves both stored values exactly
unchanged. A scan every 10 ms detects these floating-point fixed points; their
values are retained and any incoming nonzero write reactivates them. Refractory
or above-threshold cells cannot sleep. No small weights or decaying state are
pruned. Delayed presynaptic spikes use a ring of delivery
queues. Float64 neuronal state avoids changing the reference recurrence;
float32 graph values represent the supplied integer weights exactly. The
seeded application RNG is reproducible but does not reproduce NumPy's stream.

## Continuous environment feedback

`FeedingWorld` in `src/embodiment.ts` maintains neural and actuator state across
environment changes. Every 10 ms it:

1. Estimates contact from food distance, mouth extension and body height.
2. Encodes sugar and bitter as contact-scaled 0–200 Hz sensory drives.
3. Encodes relative finger/eye expansion and advances the full network.
4. Reads MN9 spikes for mouth extension and GF spikes for approximate body mechanics.

The renderer receives the computed extension, not a stimulus-selected movement.
Food does not directly command an action and moving food does not reset the
network. A shared mouth probe and food ellipsoid define contact in both the
renderer and worker (`src/feeding-geometry.ts`). Probe placement, the soft contact
boundary, taste encoding, motor filtering and joint mapping are authored
approximations. They are not a calibrated
biomechanical model or the paper's open-loop trial protocol. There is no walking,
navigation, plasticity, learned policy or automatic approach to distant food.

The worker accepts `init`, `advance` and explicit `reset` messages. It returns
state and transferable sparse activity arrays. The main thread renders the fly
and an inset of real neuron anchors; points flash from reported spikes. These
flashes are interval activity, not measured propagation along neural skeletons.
Changing display quality cannot alter the numerical timestep or graph.

### Approaching finger

`FingerStimulus` (`src/finger.ts`) runs on the same 10 ms feedback ticks. A user
target is bounded to x ±1.4, y 0.95, z −4…−1.65; the tip moves toward it at
1.5 scene units/model-second. It cannot touch the fly. Insertion begins at the
far plane. The rendered finger uses the returned worker position, not the target.
The illustrative CC0 MakeHuman pointing hand and forearm keep the index tip at
the origin, shaft along −Z, and uniform scene scale 3.2. The remaining fingers
are posed behind the tip and the mesh is rolled to clear the floor. The complete
hand is illustrative; the optical adapter approximates the fingertip only.

A sphere of radius 0.3 behind the tip approximates its optical size. Per-eye
angular size is `2 asin(radius / distance)` from approximate eye centers
(±0.242, 0.943 + body height, −0.999). Anatomical left is negative X. Broad outward eye-axis
weights approximate binocular exposure; these are not measured receptive fields.
Positive angular expansion drives LC4 as `100 min(1, expansion / 0.8)` Hz and
LPLC2 as `100 min(1, size / 0.8) min(1, expansion / 0.08)` Hz, each multiplied
by its eye weight. Angles are radians, time is model seconds. Zero-rate visual
groups are omitted from external drives, preserving normal refractoriness.

`finger.json` joins all 54/50 left/right LC4 and 108/102 LPLC2 cells by exact
v630 IDs; no graph weights or edges change. The encoding is an authored feature
approximation, not the paper's visual protocol or a full retina model. Uniform
population drive omits retinotopy and motion computation. GF/DNp01 cells are never
directly stimulated; their 120 ms filtered firing rates are displayed. Actual GF
spike counts, rather than these filtered rates, feed the body adapter below.
Bilateral GF activity is not a steering signal. Constant or decreasing angular
size supplies zero new looming input, but body descent can increase the size of
a stationary finger. Neural state persists after stimulus changes or removal.

### Approximate vertical body mechanics

`EscapeBody` (`src/body.ts`) receives actual bilateral GF counts in each 10 ms
neural interval. It bypasses missing thoracic/VNC motor neurons and integrates
authored muscles and mechanics every 1 ms. It receives no finger proximity,
behavior label, jump command or animation clock. Within-bin spike timing is
not reconstructed. GF involvement in rapid takeoff is supported by
[von Reyn et al. (2014)](https://www.nature.com/articles/nn.3741); this adapter
does not reproduce that paper's precise spike-timing mechanism or flight behavior.

Each side adds raw GF spike counts to an excitation reservoir that decays with
a 200 ms time constant. Activation is `a = 1 − exp(−3 × excitation)`, so sustained
spiking saturates contraction rather than restarting a stroke for every spike.
Activation drives a spring stroke of at most 0.3 scene units through an 8 ms lag.
With normalized mass, gravity is 20 and stiffness 1200; passive damping is 140
during descent and 28 during ascent, dissipating landing energy to suppress
rebound. Each leg supplies only nonnegative support while its stroke reaches
the platform; an airborne leg provides no thrust. These scene-unit gains and
the slower-than-biological timing are authored, not measured muscle parameters.

The returned height, velocity and compliant leg extension determine the rendered
body pose and middle-leg joints. Ground reaction, gravity and landing determine
the motion; the renderer adds no timed takeoff or floor snap. Body height also
updates the mouth contact probe and eye coordinates before the next neural tick.
The body is constrained to vertical motion, with no steering or rotation. Wings
remain folded; there is no flight controller, aerodynamic lift or wingbeat script.
Residual muscle activation and gravity can move the body after new GF spikes stop.

`prepare_finger.py --check` verifies pinned source hashes and exact identities;
`npm run test:finger` checks actual complete-network responses and disconnected
controls. These tests establish implementation causality, not a quantitative
fit to natural vision, intention or autonomous escape behavior.

The separate NeuroMechFly adult-female micro-CT GLB contains 103,369 triangles
and 30 meshes (2,464,636 bytes). Anatomical pivots articulate the mouth; surface
pigmentation, bristles and eye facets are illustrative. This body reference is
not the specimen underlying the FlyWire connectome.

## Cost and verification

The full model replaces the old sub-5 MB target: graph transfer is 50.31 MB;
all dynamics assets total 51.98 MB. The largest gzip chunk is 5.52 MB. The graph
alone occupies 118.01 MB uncompressed, before neuronal state, temporary loading
buffers and graphics. The fly, application bundle and styles add further cost.
One point buffer represents the brain; there is no per-neuron Object3D.

Measure network load, worker compute time per simulated interval and rendering
FPS separately. A high display FPS does not establish real-time whole-brain
throughput. Old small-circuit browser benchmarks are not evidence for this
model; desktop measurements and real-device checks must describe their actual
hardware, graph, input and duration.

`prepare_dynamics.py --check` checks output hashes, complete source-edge equality,
IDs, masks and protocol membership. `npm test` covers numerical and feedback
behavior alongside retained legacy tests; `npm run build` checks TypeScript and
the production bundle. `check_neural_reference.py` compares six fixed-input
fixtures with Brian2 2.10.1: exact spike counts and voltage/current tolerance
2e-10 mV. This is numerical parity on small graphs, not behavioral validation or
a replay of the paper's full statistical experiment. Browser checks independently
verify loading, interaction, rendering and worker failures.

## Legacy material

`src/simulation.ts`, `src/simulation.worker.ts`, the old Male CNS preparation
tools and `public/brain/` retain the earlier Threat/Motion/Light/Lesion circuit
demonstration. Its `FLY1` geometry and earliest-arrival event model are not the
active dynamics or scientific basis for the continuous feeding/finger experiments.
