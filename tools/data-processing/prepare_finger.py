"""Extract bilateral looming VPNs and giant-fiber readouts from pinned v630 data.

Run with Python 3: prepare_finger.py [--check]. Uses only the stdlib and existing
local assets; never downloads data or modifies the complete neural graph.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "public/dynamics"
SOURCE = ROOT / ".cache/brain-model/annotations-630.tsv"
ANNOTATION_COMMIT = "df6bb136f5b3d91c3992df4e8de2642329e2a384"
ANNOTATION_SHA = "55c99c61eecf8db6cc36f1a684b35e4c4208afbab02197d753ccc8dc2a6e2e76"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def protocol():
    manifest = json.loads((OUT / "manifest.json").read_text())
    provenance = json.loads((OUT / "provenance.json").read_text())
    assert manifest["format"] == "LIF1"
    assert provenance["annotations"]["rootIdVersion"] == 630
    assert provenance["annotations"]["commit"] == ANNOTATION_COMMIT
    assert provenance["sources"]["annotations-630.tsv"]["sha256"] == ANNOTATION_SHA
    source = SOURCE.read_bytes()
    assert digest(source) == ANNOTATION_SHA, "Pinned annotation hash mismatch"
    descriptor = manifest["ids"]
    packed = (OUT / descriptor["file"]).read_bytes()
    assert len(packed) == descriptor["compressedBytes"]
    assert digest(packed) == descriptor["compressedSha256"]
    raw = gzip.decompress(packed)
    assert len(raw) == descriptor["bytes"] == manifest["neuronCount"] * 8
    assert digest(raw) == descriptor["sha256"]
    ids = [str(value[0]) for value in struct.iter_unpack("<Q", raw)]
    lookup = {root_id: index for index, root_id in enumerate(ids)}
    assert len(lookup) == len(ids), "Duplicate graph ID"
    rows = list(csv.DictReader(io.StringIO(source.decode()), delimiter="\t"))

    def cells(cell_type, side):
        selected = [row for row in rows if row["side"] == side and
                    cell_type in (row["cell_type"], row["hemibrain_type"])]
        roots = [row["root_id"] for row in selected]
        assert len(roots) == len(set(roots)), f"Duplicate {cell_type} {side} ID"
        assert all(root_id in lookup for root_id in roots), "Unmapped sensory/readout ID"
        return {"ids": roots, "indices": [lookup[root_id] for root_id in roots]}

    inputs = {}
    for key, cell_type, side, count in [
        ("lc4Left", "LC4", "left", 54), ("lc4Right", "LC4", "right", 50),
        ("lplc2Left", "LPLC2", "left", 108), ("lplc2Right", "LPLC2", "right", 102),
    ]:
        inputs[key] = cells(cell_type, side)
        assert len(inputs[key]["ids"]) == count, f"Unexpected {key} count"
    outputs = {}
    for key, side, expected in [
        ("gfLeft", "left", "720575940622838154"),
        ("gfRight", "right", "720575940632499757"),
    ]:
        selected = cells("DNp01", side)
        assert selected["ids"] == [expected], f"Unexpected {key} identity"
        outputs[key] = {"id": expected, "index": selected["indices"][0],
                        "label": f"Giant fiber / DNp01 ({side})"}
    sensory = [index for group in inputs.values() for index in group["indices"]]
    assert len(sensory) == len(set(sensory)) == 314
    assert all(output["index"] not in sensory for output in outputs.values())
    return {
        "schemaVersion": 1,
        "name": "Approaching object: bilateral LC4/LPLC2 inputs and GF readouts",
        "datasetVersion": 630,
        "inputs": inputs,
        "outputs": outputs,
        "provenance": {
            "model": provenance["model"],
            "annotations": provenance["annotations"],
            "annotationSource": provenance["sources"]["annotations-630.tsv"],
            "graphSha256": manifest["graph"]["sha256"],
            "idsSha256": descriptor["sha256"],
            "selection": "Exact cell_type or hemibrain_type and side annotations; root IDs joined to the unchanged model's neuron order. All annotated LC4/LPLC2 cells on both sides are included.",
            "scientificSources": [
                {"url": "https://doi.org/10.1016/j.cub.2019.01.079",
                 "supports": "LC4 angular-expansion velocity and LPLC2 angular-size contributions to the giant-fiber looming response."},
                {"url": "https://doi.org/10.1038/nature24626",
                 "supports": "LPLC2 selectivity for local radial expansion and direct output to giant fibers."},
                {"url": "https://doi.org/10.7554/eLife.34272",
                 "supports": "DNp01 is the giant fiber descending neuron."},
                {"url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10263144/",
                 "supports": "Giant fibers integrate bilateral looming information; their activity is not a left/right steering command."},
            ],
        },
        "encoder": {
            "status": "Authored, simplified population-level sensory encoder; not a published Shiu visual-stimulus protocol.",
            "inputMeaning": "Independent Poisson stimulation of identified visual projection neurons. Angular expansion and size may set the input rates; these neurons are not photoreceptors.",
            "limits": [
                "No retinal image, photoreceptor transduction, T4/T5 motion computation, or per-cell receptive-field mapping is provided.",
                "LC4/LPLC2 drive approximates increasing angular size relative to the moving eyes. Stationary object size alone does not establish sustained LPLC2 activity; a stationary finger can supply expansion during body descent.",
                "Poisson input rates and their geometry-dependent scaling are authored assumptions, not measured cell firing rates or calibrated natural looming responses.",
                "GF cells are never directly stimulated. Their actual computed spike counts feed an authored muscle/spring/ground-reaction adapter that bypasses missing thoracic/VNC motor neurons and constrains movement to vertical hops without steering or flight; wings remain folded.",
                "The body adapter uses 10 ms neural count bins and 1 ms mechanical steps with uncalibrated scene-unit parameters and slower-than-biological timing. It does not reproduce the precise takeoff spike-timing mechanism of von Reyn et al. (2014), https://www.nature.com/articles/nn.3741.",
                "The unchanged Shiu LIF graph computes every downstream neural response. No GF spikes means no new muscle activation; residual neural/muscle state and gravity can still produce activity or motion. Body height feeds back into mouth contact and eye geometry.",
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate existing output without writing or downloading")
    args = parser.parse_args()
    expected = protocol()
    path = OUT / "finger.json"
    if args.check:
        assert json.loads(path.read_text()) == expected, "finger.json differs from pinned annotations/graph IDs"
    else:
        path.write_text(json.dumps(expected, separators=(",", ":"), ensure_ascii=False) + "\n")
    print("PASS: v630 LC4 54L/50R + LPLC2 108L/102R; exact bilateral GF IDs; unchanged graph; offline extraction verified.")


if __name__ == "__main__":
    main()
