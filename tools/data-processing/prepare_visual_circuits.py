"""Prepare distinct measured motion and light-on motifs from Male CNS v1.0.

.venv/bin/python tools/data-processing/prepare_visual_circuits.py
.venv/bin/python tools/data-processing/prepare_visual_circuits.py --check

Selection is deterministic from the released annotation and weight tables. The
paper-informed response signs and stimulus interpretations are not physiology
inferred from synapse counts. Raw sources remain in ignored .cache/.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "public/brain"
MOTION_TYPES = ("T4a", "T5a", "HSN", "HSE", "HSS", "DNp15")
LIGHT_TYPES = ("L1", "Mi1", "Tm3")
HS_TYPES = ("HSN", "HSE", "HSS")
SOURCES = {
    "dataset": {"title": "Male CNS v1.0 — FlyEM / Janelia", "url": "https://male-cns.janelia.org/download/"},
    "motion": {"title": "Maisak et al. (2013) — ON/OFF directional motion detectors", "url": "https://www.nature.com/articles/nature12320"},
    "steering": {"title": "Erginkaya et al. (2025) — HS and DNp15 optic-flow network", "url": "https://www.nature.com/articles/s41593-025-01948-9"},
    "light": {"title": "Behnia et al. (2014) — ON/OFF contrast responses", "url": "https://www.nature.com/articles/nature13427"},
    "polarity": {"title": "Molina-Obando et al. (2019) — Distributed ON selectivity", "url": "https://elifesciences.org/articles/49373"},
}


def prepare():
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc
    import prepare as base

    annotations = base.download(base.TABLES + "body-annotations-male-cns-v1.0-minconf-0.5.feather", base.CACHE / "annotations.feather")
    weights = base.download(base.TABLES + "connectome-weights-male-cns-v1.0-minconf-0.5.feather", base.CACHE / "weights.feather")
    rows = feather.read_table(annotations).to_pylist()
    meta = {r["bodyId"]: r for r in rows if r["status"] == "Traced" and r["somaLocation"] is not None and r["type"] in MOTION_TYPES + LIGHT_TYPES}
    ids = pa.array(sorted(meta))
    reader = ipc.open_file(str(weights))
    edges = []
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        mask = pc.and_(pc.is_in(batch["body_pre"], value_set=ids), pc.is_in(batch["body_post"], value_set=ids))
        edges.extend(batch.filter(mask).to_pylist())

    def strongest(target, cell_type, count):
        candidates = [e for e in edges if e["body_post"] == target and e["weight"] >= 5 and meta[e["body_pre"]]["type"] == cell_type and meta[e["body_pre"]]["somaSide"] == meta[target]["somaSide"]]
        return sorted(candidates, key=lambda e: (-e["weight"], e["body_pre"]))[:count]

    motion = {i for i, r in meta.items() if r["type"] == "DNp15"}
    assert len(motion) == 2, "Expected bilateral DNp15 cells in Male CNS v1.0"
    for target in sorted(motion):
        for cell_type in HS_TYPES:
            hs_inputs = strongest(target, cell_type, 1)
            assert len(hs_inputs) == 1
            hs = hs_inputs[0]["body_pre"]
            motion.add(hs)
            for motion_type in ("T4a", "T5a"):
                upstream = strongest(hs, motion_type, 2)
                assert len(upstream) == 2
                motion.update(e["body_pre"] for e in upstream)

    light = set()
    for side in ("L", "R"):
        candidates = [r for r in meta.values() if r["type"] == "L1" and r["somaSide"] == side and r["assignedOlHex1"] is not None and r["assignedOlHex2"] is not None]
        candidates.sort(key=lambda r: ((r["assignedOlHex1"] - 15) ** 2 + (r["assignedOlHex2"] - 15) ** 2, r["bodyId"]))
        selected_inputs = 0
        for row in candidates:
            targets = []
            for cell_type, count in (("Mi1", 1), ("Tm3", 2)):
                outgoing = [e for e in edges if e["body_pre"] == row["bodyId"] and e["weight"] >= 5 and meta[e["body_post"]]["type"] == cell_type and meta[e["body_post"]]["somaSide"] == side]
                outgoing.sort(key=lambda e: (-e["weight"], e["body_post"]))
                if len(outgoing) < count:
                    break
                targets.extend(e["body_post"] for e in outgoing[:count])
            if len(targets) != 3:
                continue
            light.add(row["bodyId"])
            light.update(targets)
            selected_inputs += 1
            if selected_inputs == 3:
                break
        assert selected_inputs == 3

    selections = {"motion": motion, "light": light}
    selected = motion | light
    ids = pa.array(sorted(selected))
    counts_in, counts_out = {i: 0 for i in selected}, {i: 0 for i in selected}
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        for column, counts in (("body_pre", counts_out), ("body_post", counts_in)):
            values, frequencies = np.unique(batch.filter(pc.is_in(batch[column], value_set=ids))[column].to_numpy(), return_counts=True)
            for value, count in zip(values, frequencies):
                counts[int(value)] += int(count)
    print(f"Downloading/reading {len(selected)} distinct visual-circuit skeletons", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        geometry = dict(pool.map(lambda body: (body, base.skeleton(body)), sorted(selected)))

    for kind, members in selections.items():
        neurons = [meta[i] for i in sorted(members)]
        local = {r["bodyId"]: i for i, r in enumerate(neurons)}
        descriptor, routes = base.write_lines(kind + ".bin", [(local[r["bodyId"]], r, geometry[r["bodyId"]]) for r in neurons], .012)
        # Retain only the measured forward motif being explained. Recurrent and
        # lateral partners remain represented in the full-graph partner counts.
        def allowed(e):
            pre, post = meta[e["body_pre"]]["type"], meta[e["body_post"]]["type"]
            if kind == "light":
                return pre == "L1" and post in ("Mi1", "Tm3")
            return (pre in ("T4a", "T5a") and post in HS_TYPES) or (pre in HS_TYPES and post == "DNp15")
        circuit_edges = [[local[e["body_pre"]], local[e["body_post"]], e["weight"]] for e in edges if e["body_pre"] in members and e["body_post"] in members and e["weight"] >= 5 and allowed(e)]
        circuit_edges.sort()
        circuit = {
            "kind": kind, "action": "turn" if kind == "motion" else "none",
            "source": base.SOURCE, "dataset": "male-cns:v1.0", "geometry": descriptor,
            "inputTypes": ["T4a", "T5a"] if kind == "motion" else ["L1"],
            "outputTypes": ["DNp15"] if kind == "motion" else ["Mi1", "Tm3"],
            "stages": [
                {"id": "visual", "label": "Motion detectors", "detail": "T4a · T5a", "types": ["T4a", "T5a"]},
                {"id": "descending", "label": "Optic-flow integration", "detail": "HSN · HSE · HSS", "types": list(HS_TYPES)},
                {"id": "motor", "label": "Descending steering", "detail": "DNp15 / DNHS1", "types": ["DNp15"]},
            ] if kind == "motion" else [
                {"id": "visual", "label": "Light-on input decreases", "detail": "L1 · negative response", "types": ["L1"]},
                {"id": "descending", "label": "ON relay increases", "detail": "Mi1 · Tm3 · positive response", "types": ["Mi1", "Tm3"]},
            ],
            "sources": [SOURCES[k] for k in (("dataset", "motion", "steering") if kind == "motion" else ("dataset", "light", "polarity"))],
            "scientificNote": (
                "Measured T4a/T5a→HS→DNp15 chemical-synapse motif. These sampled detectors prefer front-to-back motion; the opposite-direction detectors are not included. HS–DNp15 electrical coupling and the full competitive network are omitted. Activation timing, stimulus tuning and turn amplitude are illustrative; DNp15 is a descending steering signal, not a reconstructed motor output."
                if kind == "motion" else
                "Measured L1→Mi1/Tm3 optic-lobe motif. For a light increment, L1 decreases and Mi1/Tm3 increase, as prescribed from cell-type physiology. ON selectivity is a distributed multisynaptic computation; the drawn edges alone do not establish its mechanism. Highlights show signed response changes, not action potentials or a fitted membrane model. No phototaxis, jump or motor output is modeled."
            ),
            "selection": "Two strongest same-side T4a and T5a inputs per bilateral HSN/HSE/HSS; strongest HS of each type onto each DNp15." if kind == "motion" else "Three L1 per side nearest optic hex (15,15), each with its strongest Mi1 and two strongest Tm3 targets; only traced cells with measured soma coordinates.",
            "edgeSelection": "Measured forward edges among the selected cells with at least five chemical synapses; feedback and other partners omitted.",
            "neurons": [{"id": str(r["bodyId"]), "type": r["type"], "name": r["instance"], "side": r["somaSide"], "region": base.region(r), "group": r["superclass"], "position": base.transform(r["somaLocation"]).tolist(), "inputs": counts_in[r["bodyId"]], "outputs": counts_out[r["bodyId"]], "polarity": -1 if r["type"] == "L1" else 1, "role": {"T4a": "ON motion detector", "T5a": "OFF motion detector", "DNp15": "Descending steering neuron", "L1": "Lamina light-on decrease", "Mi1": "Medulla ON relay", "Tm3": "Medulla ON relay"}.get(r["type"], "Horizontal-system optic-flow neuron"), "path": routes[str(r["bodyId"])]} for r in neurons],
            "edges": circuit_edges,
        }
        circuit["modelNotice"] = circuit["scientificNote"]
        (OUT / f"{kind}.json").write_text(json.dumps(circuit, separators=(",", ":")) + "\n")
        print(f"{kind}: {len(neurons)} cells, {len(circuit_edges)} measured edges, {descriptor['gzipBytes']:,} skeleton bytes", flush=True)
    path = OUT / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["circuits"] = {kind: f"{kind}.json" for kind in ("escape", "motion", "light")}
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    self_check()


def self_check():
    escape = json.loads((OUT / "escape.json").read_text())
    used_ids = {n["id"] for n in escape["neurons"]}
    for kind, allowed_types in (("motion", MOTION_TYPES), ("light", LIGHT_TYPES)):
        circuit = json.loads((OUT / f"{kind}.json").read_text())
        neurons = circuit["neurons"]
        ids = {n["id"] for n in neurons}
        assert len(ids) == len(neurons) and not ids & used_ids
        used_ids |= ids
        assert len(neurons) <= (40 if kind == "motion" else 24)
        assert {n["side"] for n in neurons} == {"L", "R"}
        assert all(n["type"] in allowed_types and n["polarity"] == (-1 if n["type"] == "L1" else 1) for n in neurons)
        assert all(np.isfinite(n["position"]).all() and np.isfinite(n["path"]).all() and len(n["path"]) <= 160 for n in neurons)
        reached = {i for i, n in enumerate(neurons) if n["type"] in circuit["inputTypes"]}
        for _ in neurons:
            reached |= {b for a, b, weight in circuit["edges"] if a in reached}
        assert all(i in reached for i, n in enumerate(neurons) if n["type"] in circuit["outputTypes"])
        assert all(0 <= a < len(neurons) and 0 <= b < len(neurons) and a != b and weight >= 5 for a, b, weight in circuit["edges"])
        descriptor = circuit["geometry"]
        raw = gzip.decompress((OUT / descriptor["file"]).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == descriptor["sha256"]
        magic, version, binary_kind, count = struct.unpack_from("<4sIII", raw)
        assert (magic, version, binary_kind) == (b"FLY1", 1, 2) and count % 2 == 0
        assert len(raw) == descriptor["bytes"] == 16 + count * 20
        position = np.frombuffer(raw, dtype="<f4", count=count * 3, offset=16)
        owner = np.frombuffer(raw, dtype="<u4", count=count, offset=16 + count * 12)
        progress = np.frombuffer(raw, dtype="<f4", count=count, offset=16 + count * 16)
        assert np.isfinite(position).all() and np.isfinite(progress).all()
        assert set(owner.tolist()) == set(range(len(neurons)))
        # prepare.write_lines normalizes the negative disconnected-component
        # sentinel too; any negative value remains deliberately inactive.
        assert np.all(progress <= 1)
        assert np.array_equal(owner[::2], owner[1::2])
        print(f"{kind} self-check passed: distinct IDs, directed path, signed response, skeleton hash and bounds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    self_check() if args.check else prepare()
