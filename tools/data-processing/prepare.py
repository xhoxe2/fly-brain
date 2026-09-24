"""Prepare attributed Male CNS v1.0 anatomy and a measured escape subgraph.

Run: .venv/bin/python tools/data-processing/prepare.py
Raw inputs are cached outside the published assets. No neuPrint token required.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import heapq
import json
from pathlib import Path
import struct
import urllib.request

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / '.cache'
OUT = ROOT / 'public/brain'
BASE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/'
TABLES = BASE + 'connectome-data/flat-connectome/'
SOURCE = 'https://male-cns.janelia.org/download/'


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    temporary = target.with_suffix(target.suffix + '.part')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            temporary.replace(target)
            return target
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise


def transform(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    # One scene unit = 100 micrometres. Source SWC/soma coordinates are 8 nm.
    return np.stack(((xyz[..., 0] - 47000) * .00008,
                     (65000 - xyz[..., 2]) * .00008,
                     (xyz[..., 1] - 40000) * .00008), axis=-1).astype('<f4')


def region(row):
    group = row['superclass'] or ''
    if group.startswith('ol_') or group.startswith('visual_'):
        return 0
    if group.startswith('vnc_') or group in ('ascending_neuron', 'sensory_ascending'):
        return 2
    return 1


def skeleton(body_id):
    path = download(BASE + f'segmentation/skeletons-malecns/skeletons-swc/{body_id}.swc',
                    CACHE / 'skeletons' / f'{body_id}.swc')
    rows = np.loadtxt(path, comments='#', ndmin=2)
    if rows.shape[1] != 7 or not np.isfinite(rows).all():
        raise ValueError(f'Invalid SWC: {body_id}')
    ids = {int(row[0]): i for i, row in enumerate(rows)}
    if len(ids) != len(rows):
        raise ValueError(f'Duplicate SWC node: {body_id}')
    parents = np.array([ids.get(int(row[6]), -1) for row in rows], dtype=np.int32)
    for row in rows:
        if int(row[6]) != -1 and int(row[6]) not in ids:
            raise ValueError(f'Missing SWC parent: {body_id}')
    return transform(rows[:, 2:5]), parents


def paths(xyz, parents, step):
    """Simplify degree-two runs; preserve every endpoint and branch point."""
    children = [[] for _ in parents]
    for i, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(i)
    retained = {i for i, child in enumerate(children) if len(child) != 1 or parents[i] < 0}
    edges = []
    for start in sorted(retained):
        for child in children[start]:
            anchor = start
            current = child
            traversed = 0
            while True:
                traversed += 1
                if traversed > len(parents):
                    raise ValueError('Cyclic SWC')
                if current in retained or np.linalg.norm(xyz[current] - xyz[anchor]) >= step:
                    edges.append((anchor, current))
                    anchor = current
                if current in retained:
                    break
                current = children[current][0]
    return edges


def distances(xyz, parents, root):
    adjacency = [[] for _ in parents]
    for i, parent in enumerate(parents):
        if parent >= 0:
            length = float(np.linalg.norm(xyz[i] - xyz[parent]))
            adjacency[i].append((parent, length))
            adjacency[parent].append((i, length))
    dist = np.full(len(parents), np.inf)
    previous = np.full(len(parents), -1, dtype=np.int32)
    dist[root] = 0
    queue = [(0, int(root))]
    while queue:
        length, i = heapq.heappop(queue)
        if length > dist[i]:
            continue
        for j, weight in adjacency[i]:
            candidate = length + weight
            if candidate < dist[j]:
                dist[j] = candidate
                previous[j] = i
                heapq.heappush(queue, (candidate, j))
    finite = np.isfinite(dist)
    end = int(np.argmax(np.where(finite, dist, -1)))
    route = [end]
    while previous[route[-1]] >= 0:
        route.append(int(previous[route[-1]]))
    route.reverse()
    # Disconnected components remain visible but do not receive invented impulses.
    dist[~finite] = -1
    return dist, route


def write_binary(name, kind, arrays, count):
    raw = struct.pack('<4sIII', b'FLY1', 1, kind, count) + b''.join(a.tobytes() for a in arrays)
    path = OUT / (name + '.gz')
    compressed = gzip.compress(raw, mtime=0)
    path.write_bytes(compressed)
    return {'file': path.name, 'bytes': len(raw), 'gzipBytes': len(compressed),
            'sha256': hashlib.sha256(raw).hexdigest()}


def write_lines(name, entries, detail):
    positions, owners, progress = [], [], []
    routes = {}
    for owner, row, data in entries:
        xyz, parents = data
        edges = paths(xyz, parents, detail)
        kind = row.get('type')
        root = int(np.argmax(np.abs(xyz[:, 0]))) if kind in ('LC4', 'LPLC2') else int(np.argmax(xyz[:, 1]))
        dist, route = distances(xyz, parents, root)
        maximum = max(float(dist.max()), .0001)
        for a, b in edges:
            positions.extend((xyz[a], xyz[b]))
            owners.extend((owner, owner))
            progress.extend((dist[a] / maximum, dist[b] / maximum))
        # Bound camera samples while retaining measured path geometry.
        samples = np.linspace(0, dist[route[-1]], min(160, len(route)))
        routes[str(row['bodyId'])] = np.stack([
            np.interp(samples, dist[route], xyz[route, axis]) for axis in range(3)
        ], axis=1).tolist()
    count = len(owners)
    result = write_binary(name, 2, [np.asarray(positions, dtype='<f4').reshape(-1, 3),
                                    np.asarray(owners, dtype='<u4'),
                                    np.asarray(progress, dtype='<f4')], count)
    return result, routes


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    annotations_path = download(TABLES + 'body-annotations-male-cns-v1.0-minconf-0.5.feather', CACHE / 'annotations.feather')
    rows = feather.read_table(annotations_path).to_pylist()
    traced = [r for r in rows if r['status'] == 'Traced']
    meta = {r['bodyId']: r for r in traced}
    somas = [r for r in traced if r['somaLocation'] is not None]
    # Deterministic shuffle permits spatially unbiased draw-range quality reduction.
    np.random.default_rng(41).shuffle(somas)
    overview = write_binary('overview.bin', 1,
        [transform([r['somaLocation'] for r in somas]),
         np.array([r['bodyId'] for r in somas], dtype='<u4'),
         np.array([region(r) for r in somas], dtype='<u4')], len(somas))
    print(f'Overview: {len(somas):,} real soma locations', flush=True)

    weights_path = download(TABLES + 'connectome-weights-male-cns-v1.0-minconf-0.5.feather', CACHE / 'weights.feather')
    candidates = {r['bodyId'] for r in traced if r['type'] in ('LC4', 'LPLC2', 'DNp01', 'TTMn', 'PSI')}
    candidate_ids = pa.array(sorted(candidates))
    reader = ipc.open_file(str(weights_path))
    edges = []
    # Record-batch scanning keeps working memory bounded for the 1 GB table.
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        mask = pc.and_(pc.is_in(batch['body_pre'], value_set=candidate_ids),
                       pc.is_in(batch['body_post'], value_set=candidate_ids))
        edges.extend(batch.filter(mask).to_pylist())
    selected = {10001, 10010, 800146, 804642}
    for gf in (10001, 10010):
        for cell_type in ('LC4', 'LPLC2'):
            inputs = [e for e in edges if e['body_post'] == gf and meta[e['body_pre']]['type'] == cell_type]
            inputs.sort(key=lambda e: (-e['weight'], e['body_pre']))
            selected.update(e['body_pre'] for e in inputs[:8])
    neurons = [meta[i] for i in sorted(selected)]
    local = {r['bodyId']: i for i, r in enumerate(neurons)}
    circuit_edges = [[local[e['body_pre']], local[e['body_post']], e['weight']]
                     for e in edges if e['body_pre'] in selected and e['body_post'] in selected]

    # Representative, type-diverse skeletons per anatomical group, not a full atlas.
    region_rows = []
    for group in range(3):
        pool = [r for r in traced if region(r) == group and r['type'] and r['somaLocation']]
        by_type = {}
        for row in pool:
            by_type.setdefault(row['type'], row)
        diverse = sorted(by_type.values(), key=lambda r: r['bodyId'])
        indices = np.unique(np.linspace(0, len(diverse) - 1, min(72, len(diverse)), dtype=int))
        region_rows.append([diverse[i] for i in indices])
    all_ids = selected | {r['bodyId'] for group in region_rows for r in group}
    with ThreadPoolExecutor(max_workers=8) as pool:
        geometries = dict(pool.map(lambda body: (body, skeleton(body)), sorted(all_ids)))
    print(f'Prepared {len(geometries)} real SWC skeletons', flush=True)
    circuit_geometry, routes = write_lines('escape.bin',
        [(local[r['bodyId']], r, geometries[r['bodyId']]) for r in neurons], .012)
    counts_in = {i: 0 for i in selected}
    counts_out = {i: 0 for i in selected}
    ids = pa.array(sorted(selected))
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        for name, counts in [('body_pre', counts_out), ('body_post', counts_in)]:
            filtered = batch.filter(pc.is_in(batch[name], value_set=ids))
            values, freq = np.unique(filtered[name].to_numpy(), return_counts=True)
            for value, count in zip(values, freq):
                counts[int(value)] += int(count)
    circuit = {'source': SOURCE, 'dataset': 'male-cns:v1.0', 'geometry': circuit_geometry,
        'kind': 'escape', 'action': 'jump', 'inputTypes': ['LC4', 'LPLC2'], 'outputTypes': ['TTMn'],
        'stages': [{'id': 'visual', 'label': 'Looming input', 'detail': 'LC4 · LPLC2', 'types': ['LC4', 'LPLC2']},
                   {'id': 'descending', 'label': 'Escape command', 'detail': 'Giant fiber · DNp01', 'types': ['DNp01']},
                   {'id': 'motor', 'label': 'Jump output', 'detail': 'TTMn', 'types': ['TTMn']}],
        'sources': [{'title': 'Male CNS source data', 'url': SOURCE}],
        'scientificNote': 'Measured LC4/LPLC2 → giant-fiber → TTMn chemical connections. Timing and jump are illustrative; this is not recorded brain activity or a biophysical simulation.',
        'neurons': [{'id': str(r['bodyId']), 'type': r['type'], 'name': r['instance'],
                     'side': r['somaSide'], 'region': region(r), 'group': r['superclass'], 'polarity': 1,
                     'position': transform(r['somaLocation']).tolist(),
                     'inputs': counts_in[r['bodyId']], 'outputs': counts_out[r['bodyId']],
                     'role': 'Visual projection' if r['type'] in ('LC4', 'LPLC2') else
                             'Descending giant fiber' if r['type'] == 'DNp01' else 'Jump motor neuron',
                     'path': routes[str(r['bodyId'])]} for r in neurons],
        'edges': circuit_edges,
        'modelNotice': 'Measured chemical-synapse graph; illustrative activation delays, signal direction within skeletons and behavior. Electrical synapses are not modeled.'}
    (OUT / 'escape.json').write_text(json.dumps(circuit, separators=(',', ':')))
    regions = []
    background_entries = []
    for group, name in enumerate(['Optic lobes', 'Central brain', 'Ventral nerve cord']):
        entries = [(group, r, geometries[r['bodyId']]) for r in region_rows[group][:64]]
        fine, _ = write_lines(f'region-{group}.bin', entries, .018)
        regions.append({'id': group, 'name': name, 'neurons': len(entries), 'geometry': fine})
        background_entries.extend(entries[:24])
    background, _ = write_lines('background.bin', background_entries, .09)
    manifest = {'version': 1, 'dataset': 'male-cns:v1.0', 'source': SOURCE,
        'license': 'CC BY 4.0', 'attribution': 'FlyEM / HHMI Janelia, Cambridge, MRC LMB and Google Research',
        'tracedNeurons': len(traced), 'mappedSomas': len(somas),
        'omittedWithoutSoma': len(traced) - len(somas), 'units': '100 micrometres',
        'overview': overview, 'background': background, 'regions': regions,
        'circuits': {kind: f'{kind}.json' for kind in ('escape', 'motion', 'light')},
        'circuit': 'escape.json', 'circuitNeurons': len(neurons), 'circuitEdges': len(circuit_edges),
        'note': 'Overview positions are actual cell bodies. Skeleton overview and region detail are representative subsets, not every neuron.'}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'circuitNeurons': len(neurons), 'circuitEdges': len(circuit_edges),
                      'publishedMB': round(sum(p.stat().st_size for p in OUT.iterdir()) / 1e6, 2)}, indent=2))


def self_check():
    xyz = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 1, 0], [3, -1, 0]], dtype=float)
    parents = np.array([-1, 0, 1, 2, 2])
    assert paths(xyz, parents, 10) == [(0, 2), (2, 3), (2, 4)]
    dist, route = distances(xyz, parents, 0)
    assert abs(dist[3] - 2 - 2 ** .5) < 1e-6
    assert route[0] == 0 and route[-1] in (3, 4)
    assert np.allclose(transform([47000, 40000, 65000]), [0, 0, 0])
    print('Data self-check passed: branch preservation, path distance, coordinate transform.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    self_check() if args.check else prepare()
