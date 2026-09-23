# FLY//BRAIN architecture

Static Vite + TypeScript application. Three.js WebGPURenderer uses WebGPU when
available and WebGL2 otherwise. Python runs only during data preparation.
No credentials, neuPrint requests, Python runtime or backend in the website.

## Source

Male CNS v1.0, https://male-cns.janelia.org/download/ (CC BY 4.0).
Annotation and connectivity tables are Apache Arrow Feather; morphology is SWC.
Original inputs remain in ignored `.cache/`. Scientific IDs are preserved as
strings in metadata and mapped to dense integer indices in GPU attributes.

## Data and detail levels

Small JSON manifests describe provenance, counts, bounds and file names.
Numerical geometry uses little-endian float32 and uint32 sections with a magic
header, version and element counts. The reader validates lengths and indices.

1. Overview: batch representative neuron coordinates / simplified skeletons.
2. Region: load and replace a bounded subset of skeleton geometry on demand.
3. Selected circuit: full prepared skeletons, connectivity and path distances.

Cap geometry, not simulation correctness. No Object3D per neuron. Background
geometry and active skeletons each use batched draw calls. Impulse position is
computed in a node shader from a shared simulation clock and path distance.
Workers calculate event timing; main thread updates uniforms and cameras.

## Budgets and verification

Target 16.7 ms frames on high desktop, 33.3 ms on medium hardware. These are
targets, not measured guarantees. First load target < 5 MB compressed;
individual data chunks < 5 MB; brain draw calls < 20; no unbounded GPU cache.
Report observed FPS, frame interval, CPU submission, GPU time when available,
draw calls, primitives and tracked buffer bytes. GPU allocations are not total
VRAM. Benchmark only visible foreground frames; pause when hidden.

Initial calibration chooses detail and pixel ratio. Manual quality overrides
automatic changes. Reduced motion disables idle rotation/camera travel.

## Scientific scope

Real anatomy and measured synapse counts do not establish conduction velocity,
synaptic sign, membrane dynamics or a validated behavioral model. Model delays,
stimulus coupling and virtual fly movements must be explicitly identified as
illustrative. Unverified links must never be presented as measured connections.

## Delivery order

Performance instrumentation → real-data preparation → overview → region detail
→ event propagation → stimulus/motor circuit → fly → synchronized behavior
→ lesions → other stimuli → signal-follow camera → final visual treatment.
Record measured milestones and remaining limitations in README.

## Binary layout (version 1)

Files use explicit `.bin.gz` gzip containers. The browser uses native
`DecompressionStream`, and also accepts already-decoded responses from hosts
that set Content-Encoding. Uncompressed bytes begin with a 16-byte header:

| Offset   | Type                  | Meaning                                                    |
| -------- | --------------------- | ---------------------------------------------------------- |
| 0        | 4 ASCII bytes         | `FLY1`                                                     |
| 4        | uint32 LE             | version, 1                                                 |
| 8        | uint32 LE             | kind: 1 = soma points, 2 = line vertices                   |
| 12       | uint32 LE             | N, vertex count                                            |
| 16       | float32 LE × 3N       | XYZ positions                                              |
| 16 + 12N | uint32 LE × N         | body IDs (points) or dense owner indices (lines)           |
| 16 + 16N | uint32/float32 LE × N | region groups (points) or normalized root distance (lines) |

Line vertices occur in endpoint pairs. A negative root distance marks a
component disconnected from the chosen root; it stays visible without receiving
an invented impulse. ArrayBuffer views avoid copies during decoding. Circuit
onset attributes change only between runs, never once per neuron per frame.

The escape graph is a small JSON manifest of 36 neurons and 250 edges. Camera
routes contain at most 160 samples per neuron, uniformly spaced along geometric
arc length. The camera and line shader therefore share the same progression.

## Model

Dijkstra-style earliest arrival on positive chemical connections of weight ≥ 5.
At most 500 neurons; an O(n²) scan is deliberately sufficient at this scale.
The source onset is derived from stimulus parameters. Each cell conducts for
0.85 model seconds; a weighted inter-neuron delay follows. The jump starts when
the earliest reached TTMn completes conduction. Removing any selected group
removes its vertices from traversal. Baseline and lesion receive identical
stimulus parameters in the same worker request.

One shared clock drives the shader, arena, fly and follow camera. Pausing and
scrubbing affect every view together. Hidden-tab time is excluded. Camera
control, selection and quality do not alter graph results.
