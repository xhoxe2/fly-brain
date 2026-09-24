"""Losslessly package Shiu et al.'s complete FlyWire 630 LIF graph for the browser.

Run with .venv/bin/python tools/data-processing/prepare_dynamics.py [--check].
Preparation needs numpy and pyarrow; validation needs numpy and the stdlib.
No neuron, signed connection, or source weight is filtered or rescaled. Raw
sources stay in ignored .cache/brain-model. Anchor positions are not somata.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache/brain-model"
OUT = ROOT / "public/dynamics"
MODEL_COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
ANNOTATION_COMMIT = "df6bb136f5b3d91c3992df4e8de2642329e2a384"
MODEL_BASE = f"https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{MODEL_COMMIT}/"
ANNOTATION_BASE = f"https://raw.githubusercontent.com/flyconnectome/flywire_annotations/{ANNOTATION_COMMIT}/"
COMP = "2023_03_23_completeness_630_final.csv"
CON = "2023_03_23_connectivity_630_final.parquet"
N, E = 127400, 14687178
CHUNK_BYTES = 8_000_000  # All arrays/sections stay 4-byte aligned.
SOURCES = {
    COMP: (MODEL_BASE + COMP, "e6b71e17671a9bdb05f55e4bc6774640a1418cb7a05125e0fc994ad40f9bfdfb"),
    CON: (MODEL_BASE + CON, "94db8c650533bc36ffa3223f2e62325d5648b8d6bd31c3a4e1c804628c7557b3"),
    "LICENSE": (MODEL_BASE + "LICENSE", "3621f6d6476189190e2960fa43f11b275ba4eca848fdac50a1ba2925de6223e8"),
    "model.py": (MODEL_BASE + "model.py", "fc45837d7122c6ce2a7f3f2f23c515992e4b232aadb919efabb72337fac88e4e"),
    "figures.ipynb": (MODEL_BASE + "figures.ipynb", "33cd73c0b4b4a291c51b7bab639c4fa28978e49e111ff0d6adbca5687addb84c"),
    "annotations-630.tsv": (ANNOTATION_BASE + "supplemental_files/Supplemental_file1_annotations.tsv", "55c99c61eecf8db6cc36f1a684b35e4c4208afbab02197d753ccc8dc2a6e2e76"),
    "annotations-630-columns.md": (ANNOTATION_BASE + "supplemental_files/Supplemental_files_columns.md", "ffc633fa1e5910fad032ade31992645ad4d87737e2a32d530fcefe8cf68a404b"),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source(name):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    url, expected = SOURCES[name]
    if not path.exists():
        with urlopen(url, timeout=180) as response, path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    assert digest(path.read_bytes()) == expected, f"Source hash mismatch: {name}"
    return path


def json_bytes(value):
    return (json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def compressed(file, raw):
    packed = gzip.compress(raw, compresslevel=6, mtime=0)
    (OUT / file).write_bytes(packed)
    return {"file": file, "bytes": len(raw), "compressedBytes": len(packed),
            "sha256": digest(raw), "compressedSha256": digest(packed)}


def neuron_ids():
    with source(COMP).open() as handle:
        return np.array([int(row[""]) for row in csv.DictReader(handle)], dtype="<u8")


def protocol(ids):
    """Extract literal lists from the pinned notebook; never execute its cells."""
    values = {}
    for cell in json.loads(source("figures.ipynb").read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:  # Notebook magics are unrelated to these lists.
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in ("neu_sugar", "neu_bitter", "ids_mn9"):
                    value = ast.literal_eval(node.value)
                    assert name not in values or values[name] == value
                    values[name] = value
    lookup = {int(root_id): i for i, root_id in enumerate(ids)}
    def group(name, label):
        roots = values[name]
        return {"label": label, "ids": [str(i) for i in roots], "indices": [lookup[i] for i in roots]}
    assert len(values["neu_sugar"]) == len(values["neu_bitter"]) == 21
    assert len(values["ids_mn9"]) == 2
    return {
        "schemaVersion": 1, "name": "Shiu sugar-bitter competition", "datasetVersion": 630,
        "source": f"https://github.com/philshiu/Drosophila_brain_model/blob/{MODEL_COMMIT}/figures.ipynb",
        "paper": "https://www.nature.com/articles/s41586-024-07763-9",
        "inputs": {"sugar": group("neu_sugar", "Right labellar sugar GRNs"),
                   "bitter": group("neu_bitter", "Right labellar bitter GRNs")},
        "outputs": {"primary": {"id": str(values["ids_mn9"][0]), "index": lookup[values["ids_mn9"][0]],
                                "label": "MN9 left", "role": "Rostrum lifting motor neuron; Figure 3b primary readout"},
                    "additional": [{"id": str(values["ids_mn9"][1]), "index": lookup[values["ids_mn9"][1]],
                                    "label": "MN9 right", "role": "Contralateral rostrum lifting motor neuron"}]},
        "publishedProtocol": {"trialDurationMs": 1000, "trialsPerCondition": 30,
                              "ratesHz": list(range(0, 201, 20)), "stimulus": "Independent Poisson voltage inputs to each listed GRN",
                              "primaryReadout": "MN9-left firing rate; final paper Figure 3b, notebook Figure 3 A"},
        "modelParameters": {"restMv": -52, "resetMv": -52, "thresholdMv": -45,
                            "membraneTauMs": 20, "synapseTauMs": 5, "refractoryMs": 2.2,
                            "synapticDelayMs": 1.8, "synapticGainMv": .275,
                            "poissonGain": 250, "inputRefractoryMs": 0},
        "scientificNote": "The published protocol defines stimulation and neural readouts in the full recurrent network. It is not a calibrated environment-to-sensation or neuron-to-body controller. Any such controller must be identified separately as a model assumption.",
    }


def prepare():
    import pyarrow.parquet as pq
    OUT.mkdir(parents=True, exist_ok=True)
    ids = neuron_ids()
    assert len(ids) == N and len(np.unique(ids)) == N
    table = pq.read_table(source(CON), columns=["Presynaptic_ID", "Postsynaptic_ID", "Presynaptic_Index", "Postsynaptic_Index", "Connectivity", "Excitatory", "Excitatory x Connectivity"])
    arrays = {key: table[key].combine_chunks().to_numpy() for key in table.column_names}
    pre, post = arrays["Presynaptic_Index"], arrays["Postsynaptic_Index"]
    weight = arrays["Excitatory x Connectivity"]
    assert len(pre) == E and np.all(pre[1:] >= pre[:-1])
    assert pre.min() >= 0 and post.min() >= 0 and pre.max() < N and post.max() < N
    assert np.array_equal(ids[pre], arrays["Presynaptic_ID"]) and np.array_equal(ids[post], arrays["Postsynaptic_ID"])
    assert np.array_equal(weight, arrays["Connectivity"] * arrays["Excitatory"])
    assert np.all(np.abs(weight) < 2**24), "float32 must preserve every signed count exactly"
    row = np.zeros(N + 1, dtype="<u4")
    np.cumsum(np.bincount(pre, minlength=N), out=row[1:])
    # Preserve the supplied ordering within every presynaptic row, including all
    # one-synapse connections. No thresholding, graph sampling or edge merging.
    raw_path = CACHE / "graph-LIF1.bin"
    with raw_path.open("wb") as handle:
        handle.write(struct.pack("<4sII", b"LIF1", N, E))
        for array in (row, post.astype("<u4"), weight.astype("<f4")):
            handle.write(array.tobytes())
    graph_hash = hashlib.sha256()
    chunks = []
    with raw_path.open("rb") as handle:
        offset = 0
        while block := handle.read(CHUNK_BYTES):
            graph_hash.update(block)
            chunk = compressed(f"graph-{len(chunks):02}.bin.gz", block)
            chunk["offset"] = offset
            assert chunk["compressedBytes"] < 8_000_000
            chunks.append(chunk)
            offset += len(block)
    print(f"Graph: {N:,} neurons / {E:,} edges / {sum(c['compressedBytes'] for c in chunks):,} compressed bytes", flush=True)

    with source("annotations-630.tsv").open() as handle:
        annotations = {int(r["root_id"]): r for r in csv.DictReader(handle, delimiter="\t")}
    positions = np.zeros((N, 3), dtype="<f4")
    valid = np.zeros(N, dtype="u1")
    fields = ("flow", "super_class", "cell_class", "cell_sub_class", "cell_type", "hemibrain_type", "top_nt", "side")
    # Dictionary encoding preserves annotation text without a 127k-object JSON.
    dictionaries = {field: sorted({""} | {r[field] for r in annotations.values()}) for field in fields}
    indexes = {field: {value: i for i, value in enumerate(dictionaries[field])} for field in fields}
    codes = np.zeros((N, len(fields)), dtype="<u2")
    soma_count = 0
    for i, root_id in enumerate(ids):
        annotation = annotations.get(int(root_id))
        if annotation is None:
            continue
        anchor = [float(annotation["pos_" + c]) for c in "xyz"]
        if np.all(np.isfinite(anchor)):
            positions[i] = np.array(anchor) * (4, 4, 40)
            valid[i] = 1
        soma_count += all(annotation["soma_" + c] not in ("", "NA", "nan") for c in "xyz")
        codes[i] = [indexes[field][annotation[field]] for field in fields]
    assert valid.sum() == 127322 and soma_count == 115562
    inputs = protocol(ids)
    selected = [i for g in inputs["inputs"].values() for i in g["indices"]]
    selected += [inputs["outputs"]["primary"]["index"], *[g["index"] for g in inputs["outputs"]["additional"]]]
    assert np.all(valid[selected]), "All protocol cells need exact source anchor coordinates"
    (OUT / "feeding.json").write_bytes(json_bytes(inputs))
    manifest = {
        "schemaVersion": 1, "format": "LIF1", "byteOrder": "little", "neuronCount": N, "edgeCount": E,
        "dataset": "FlyWire adult female brain", "datasetVersion": 630,
        "graph": {"bytes": raw_path.stat().st_size, "sha256": graph_hash.hexdigest(), "chunks": chunks,
                  "layout": ["magic:4 ASCII bytes", "N:uint32", "E:uint32", "rowPtr:uint32[N+1]", "target:uint32[E]", "signedWeight:float32[E]"],
                  "weightUnit": "signed synapse count before multiplication by 0.275 mV",
                  "positiveEdges": int(np.count_nonzero(weight > 0)), "negativeEdges": int(np.count_nonzero(weight < 0)),
                  "totalSynapses": int(arrays["Connectivity"].sum()), "filteredEdges": 0, "filteredNeurons": 0},
        "ids": {**compressed("ids.bin.gz", ids.tobytes()), "encoding": "uint64[N] in exact author completeness row order"},
        "positions": {**compressed("positions.bin.gz", positions.tobytes() + valid.tobytes()),
                      "layout": ["xyz:float32[N*3]", "valid:uint8[N]"], "unit": "nanometer",
                      "kind": "neuron anchor, typically on backbone; not soma",
                      "validCount": int(valid.sum()), "missingCount": int((valid == 0).sum()),
                      "somaAnnotationCount": soma_count,
                      "bounds": {"min": positions[valid != 0].min(axis=0).tolist(), "max": positions[valid != 0].max(axis=0).tolist()},
                      "sourceTransform": "Original 4×4×40 nm voxel coordinates multiplied by [4,4,40]; no registration to another brain."},
        "annotations": {**compressed("annotations.bin.gz", codes.tobytes()), "layout": "uint16[N*8], neuron-major",
                        "fields": fields, "dictionaries": dictionaries},
        "protocol": {"file": "feeding.json"}, "provenance": "provenance.json",
    }
    (OUT / "manifest.json").write_bytes(json_bytes(manifest))
    provenance = {
        "model": {"repository": "https://github.com/philshiu/Drosophila_brain_model", "commit": MODEL_COMMIT,
                  "license": "MIT", "paper": "https://www.nature.com/articles/s41586-024-07763-9"},
        "annotations": {"repository": "https://github.com/flyconnectome/flywire_annotations", "commit": ANNOTATION_COMMIT,
                        "release": "v1.1.0", "rootIdVersion": 630, "license": "CC BY 4.0",
                        "licenseRecord": "https://zenodo.org/records/8077335",
                        "paper": "https://doi.org/10.1101/2023.06.27.546055"},
        "sources": {name: {"url": url, "sha256": sha, "bytes": source(name).stat().st_size} for name, (url, sha) in SOURCES.items()},
        "preparation": "Lossless index-preserving CSR packing of every author edge; signed weights stored exactly as float32, without applying model gain. Anchor coordinates and categorical annotations join by exact FlyWire 630 root ID. Missing coordinates remain invalid; those neurons remain in dynamics.",
        "limitations": ["Anchor-point rendering is not full neuron morphology or anatomical synapse positions.",
                        "Neurotransmitter signs and homogeneous physiological parameters are model assumptions from the author graph/code, not individually measured membrane dynamics.",
                        "Sensory/body transduction is not supplied by the static connectome and must not be presented as a calibrated biological controller."],
    }
    (OUT / "provenance.json").write_bytes(json_bytes(provenance))
    shutil.copyfile(source("LICENSE"), OUT / "MODEL-LICENSE")
    (OUT / "NOTICE").write_text(
        "Whole-brain model: Philip Shiu et al., A Drosophila computational brain model reveals sensorimotor processing, Nature (2024).\n"
        "Author code and connectivity distribution: MIT; see MODEL-LICENSE and provenance.json.\n"
        "FlyWire 630 annotations: Philipp Schlegel et al. (2023), Whole-brain annotation and multi-connectome cell typing quantifies circuit stereotypy in Drosophila.\n"
        "Annotation dataset released CC BY 4.0: https://creativecommons.org/licenses/by/4.0/ ; https://zenodo.org/records/8077335 .\n"
        "Derived assets preserve all model neurons and signed connections. Positions are source anchor points, not somata.\n"
    )
    check()


def inflate(descriptor):
    packed = (OUT / descriptor["file"]).read_bytes()
    assert len(packed) == descriptor["compressedBytes"] and digest(packed) == descriptor["compressedSha256"]
    raw = gzip.decompress(packed)
    assert len(raw) == descriptor["bytes"] and digest(raw) == descriptor["sha256"]
    return raw


def check():
    manifest = json.loads((OUT / "manifest.json").read_text())
    raw = bytearray()
    for chunk in manifest["graph"]["chunks"]:
        assert chunk["offset"] == len(raw) and chunk["compressedBytes"] < 8_000_000
        raw.extend(inflate(chunk))
    assert len(raw) == manifest["graph"]["bytes"] == 12 + 4 * (N + 1) + 8 * E
    assert digest(raw) == manifest["graph"]["sha256"]
    assert struct.unpack_from("<4sII", raw) == (b"LIF1", N, E)
    row = np.frombuffer(raw, "<u4", N + 1, 12)
    target = np.frombuffer(raw, "<u4", E, 12 + 4 * (N + 1))
    weight = np.frombuffer(raw, "<f4", E, 12 + 4 * (N + 1) + 4 * E)
    assert row[0] == 0 and row[-1] == E and np.all(row[1:] >= row[:-1])
    assert target.max() < N and np.all(np.isfinite(weight)) and np.all(weight != 0)
    assert np.all(weight == np.trunc(weight))
    assert np.count_nonzero(weight < 0) == 5886646 and np.count_nonzero(weight > 0) == 8800532
    assert int(np.abs(weight).sum(dtype=np.float64)) == 52793639
    ids = np.frombuffer(inflate(manifest["ids"]), "<u8")
    assert len(ids) == N and len(np.unique(ids)) == N
    pos_raw = inflate(manifest["positions"])
    positions = np.frombuffer(pos_raw, "<f4", N * 3).reshape(N, 3)
    valid = np.frombuffer(pos_raw, "u1", N, N * 12)
    assert len(pos_raw) == N * 13 and valid.sum() == 127322 and np.all(np.isin(valid, [0, 1]))
    assert np.all(np.isfinite(positions)) and np.all(positions[valid == 0] == 0)
    codes = np.frombuffer(inflate(manifest["annotations"]), "<u2").reshape(N, -1)
    for i, field in enumerate(manifest["annotations"]["fields"]):
        assert codes[:, i].max() < len(manifest["annotations"]["dictionaries"][field])
    p = json.loads((OUT / manifest["protocol"]["file"]).read_text())
    for group in p["inputs"].values():
        assert [str(int(ids[i])) for i in group["indices"]] == group["ids"]
        assert np.all(valid[group["indices"]])
    for output in [p["outputs"]["primary"], *p["outputs"]["additional"]]:
        assert str(int(ids[output["index"]])) == output["id"] and valid[output["index"]]
    # When source data exists, prove every packaged edge equals its source row,
    # rather than only trusting aggregate counts and output hashes.
    if (CACHE / CON).exists():
        import pyarrow.parquet as pq
        source(CON)
        table = pq.read_table(CACHE / CON, columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
        source_pre = table["Presynaptic_Index"].combine_chunks().to_numpy()
        assert np.array_equal(np.diff(row).astype(np.int64), np.bincount(source_pre, minlength=N))
        assert np.array_equal(target, table["Postsynaptic_Index"].combine_chunks().to_numpy())
        assert np.array_equal(weight, table["Excitatory x Connectivity"].combine_chunks().to_numpy())
        assert np.array_equal(ids, neuron_ids())
    print(f"LIF1 self-check passed: {N:,} neurons, {E:,} exact signed edges, {int(valid.sum()):,} real anchors; all 44 protocol cells match.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check() if args.check else prepare()
