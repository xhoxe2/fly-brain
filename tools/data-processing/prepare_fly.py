#!/usr/bin/env python3
"""Prepare a posed, small NeuroMechFly GLB with NumPy and installed Three.js.

python tools/data-processing/prepare_fly.py
python tools/data-processing/prepare_fly.py --self-check

Source STLs and rig metadata stay in ignored .cache/fly. No flygym/MuJoCo is
required at runtime. The visual anatomy is a female reference specimen, not the
Male CNS connectome animal. Animation in the app is illustrative, not dynamics.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache/fly"
OUT = ROOT / "public/fly"
REV = "152506d3471646f009480c81f34aefbaec29a6e5"
LICENSE_REV = "38c8ec61034cd59bc5ba0de20688d4a3c0000d60"
BASE = f"https://raw.githubusercontent.com/NeLy-EPFL/fly-svg-maker/{REV}/"
VISUAL_REV = "6fdc6212bcf5b71fcd47bca119876c92092c2cd6"
VISUAL_BASE = f"https://raw.githubusercontent.com/NeLy-EPFL/NeuroMechFly/{VISUAL_REV}/"
VISUAL_MESHES = {
    "c_head.stl": "Head.stl", "c_thorax.stl": "Thorax.stl",
    "l_eye.stl": "LEye.stl", "l_wing.stl": "LWing.stl",
    "c_abdomen12.stl": "A1A2.stl", "c_rostrum.stl": "Rostrum.stl",
    "c_haustellum.stl": "Haustellum.stl", "l_haltere.stl": "LHaltere.stl",
    **{f"c_abdomen{i}.stl": f"A{i}.stl" for i in range(3, 7)},
    **{f"l{limb.lower()}_{part}.stl": f"L{limb}{original}.stl"
       for limb in "FMH" for part, original in
       [("coxa", "Coxa"), ("trochanterfemur", "Femur"), ("tibia", "Tibia")]
       + [(f"tarsus{i}", f"Tarsus{i}") for i in range(1, 6)]},
}


def download(path: str, url: str) -> None:
    dest = CACHE / path
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response:
        data = response.read()
    temp = dest.with_suffix(dest.suffix + ".tmp")
    temp.write_bytes(data)
    temp.replace(dest)


def acquire() -> None:
    download("model.json", BASE + "assets/model.json")
    model = json.loads((CACHE / "model.json").read_text())
    files = sorted({v["file"] for v in model["meshes"].values()})
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda f: download("meshes/" + f, BASE + "assets/meshes/" + f), files))
        list(pool.map(lambda f: download("highres/" + f, VISUAL_BASE + "data/design/meshes/stl/" + f), sorted(set(VISUAL_MESHES.values()))))
    download("ORIGINAL-LICENSE", VISUAL_BASE + "LICENSE")
    download("NOTICE", BASE + "assets/NOTICE")
    download("EDITOR-LICENSE", BASE + "LICENSE")
    download("LICENSE", f"https://raw.githubusercontent.com/NeLy-EPFL/flygym/{LICENSE_REV}/LICENSE")


def build() -> None:
    import numpy as np

    model = json.loads((CACHE / "model.json").read_text())
    (CACHE / "highres-map.json").write_text(json.dumps(VISUAL_MESHES))
    simplifier = CACHE / "simplify.mjs"
    simplifier.write_text('''
import fs from 'node:fs';
import {STLLoader} from 'three/addons/loaders/STLLoader.js';
import {SimplifyModifier} from 'three/addons/modifiers/SimplifyModifier.js';
import {mergeVertices} from 'three/addons/utils/BufferGeometryUtils.js';
const model=JSON.parse(fs.readFileSync(new URL('./model.json',import.meta.url)));
const sources=JSON.parse(fs.readFileSync(new URL('./highres-map.json',import.meta.url)));
const result={};
for(const file of new Set(Object.values(model.meshes).map(m=>m.file))){
  const raw=fs.readFileSync(new URL(sources[file]?'./highres/'+sources[file]:'./meshes/'+file,import.meta.url));
  let geometry=new STLLoader().parse(raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength));
  geometry.deleteAttribute('normal');
  geometry.scale(model.meshScale,model.meshScale,model.meshScale);
  geometry=mergeVertices(geometry,1e-6);
  // Some upstream collision surfaces contain opposite-winding duplicate faces.
  // They cancel normals after welding; retain one face for each vertex triple.
  const seen=new Set(),clean=[];
  for(let i=0;i<geometry.index.count;i+=3){
    const face=Array.from(geometry.index.array.slice(i,i+3));
    if(new Set(face).size!==3)continue;
    const key=face.slice().sort((a,b)=>a-b).join(',');
    if(!seen.has(key)){seen.add(key);clean.push(...face);}
  }
  geometry.setIndex(clean);
  const shell=/thorax|head|abdomen|eye/.test(file);
  const target=file.includes('head')?10000:file.includes('eye')?5000:
    file.includes('wing')?5000:file.includes('thorax')?5000:
    file.includes('abdomen')?2500:file.includes('arista')?2000:1000;
  const ratio=Math.min(1,target/(geometry.index.count/3));
  if(ratio<1){
    geometry=await new SimplifyModifier().modify(geometry,Math.floor(geometry.attributes.position.count*(1-ratio)));
  }
  if(shell){
    // Three small Taubin passes soften segmentation noise without shrinking the
    // body. Antennae, joints, legs and wing outlines retain the source geometry.
    const position=geometry.attributes.position.array, index=geometry.index.array;
    const neighbors=Array.from({length:position.length/3},()=>new Set());
    for(let i=0;i<index.length;i+=3){
      for(let j=0;j<3;j++){
        const a=index[i+j],b=index[i+(j+1)%3];
        neighbors[a].add(b);neighbors[b].add(a);
      }
    }
    const passes=/thorax|head|abdomen/.test(file)?8:2;
    for(let pass=0;pass<passes;pass++)for(const factor of [.45,-.47]){
      const previous=position.slice();
      for(let i=0;i<neighbors.length;i++)for(let axis=0;axis<3;axis++){
        let sum=0;for(const n of neighbors[i])sum+=previous[n*3+axis];
        if(neighbors[i].size)position[i*3+axis]+=factor*(sum/neighbors[i].size-previous[i*3+axis]);
      }
    }
  }
  result[file]={vertices:Array.from(geometry.attributes.position.array),faces:Array.from(geometry.index.array)};
}
fs.writeFileSync(new URL('./simplified.json',import.meta.url),JSON.stringify(result));
''')
    subprocess.run(["node", str(simplifier)], check=True)
    simplified = json.loads((CACHE / "simplified.json").read_text())

    def quaternion_matrix(values):
        w, x, y, z = np.array(values) / np.linalg.norm(values)
        return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

    def axis_rotation(axis, angle):
        x, y, z = axis
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        return np.eye(3) + math.sin(angle)*skew + (1-math.cos(angle))*(skew @ skew)

    rotation = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.float64)
    transforms, axes = {}, {}
    parents = dict((child, parent) for parent, child in model["joints"])
    for name in model["segments"]:
        rest = model["rest"][name]
        matrix = np.eye(4)
        matrix[:3, :3] = quaternion_matrix(rest["quat"])
        matrix[:3, 3] = rest["pos"]
        axes[name] = {}
        for dof in model["dofs"]:
            if dof["child"] == name:
                axis = dof["axis"]
                if isinstance(axis, str):
                    axis = model["axisVector"][axis]
                basis = transforms[parents[name]][:3, :3] if name in parents else np.eye(3)
                axes[name][dof["name"].rsplit("-", 1)[-1] + "Axis"] = (rotation @ basis @ matrix[:3, :3] @ axis).tolist()
                angle = math.radians(model["neutralDeg"].get(dof["name"], 0))
                matrix[:3, :3] = matrix[:3, :3] @ axis_rotation(axis, angle)
        transforms[name] = transforms[parents[name]] @ matrix if name in parents else matrix
    # Check the coordinate convention against the independent upstream FK fixture.
    error = max(abs(transforms[n][i][3] - p[i]) for n, p in model["reference"]["neutralJointPositions"].items() for i in range(3))
    assert error < 2e-6, f"Upstream neutral pose mismatch: {error}"

    vertices, local_vertices, faces = {}, {}, {}
    for name in model["segments"]:
        cfg = model["meshes"][name]
        source = simplified[cfg["file"]]
        points = np.array(source["vertices"], dtype=np.float64).reshape(-1, 3)
        indices = np.array(source["faces"], dtype=np.uint32).reshape(-1, 3)
        # The original visual mouth surfaces use the extended-feeding local
        # frame, whereas the v2 rig and its reference meshes encode a folded
        # resting proboscis. Register the surfaces, not the already folded joint
        # chain. Angles/offsets follow a rigid fit to the pinned v2 mesh reference.
        mouth_frames = {"c_rostrum": (90, [.005, -.0015, -.0076]),
                        "c_haustellum": (25, [.0124, -.0055, -.0094])}
        if name in mouth_frames:
            angle, offset = mouth_frames[name]
            points = points @ axis_rotation([0, 1, 0], math.radians(angle)).T + offset
        # The original extended-wing visual frame also differs from v2. A rigid
        # registration to the official v2 left-wing surface is +15.21 degrees;
        # apply it before mirroring so both folded wings sit over the abdomen.
        if name in ("l_wing", "r_wing"):
            points = points @ axis_rotation([0, 1, 0], math.radians(15.21)).T + [-.0149469, .0080163, .0085825]
        # Reduction can collapse opposite shell faces onto the same triangle.
        # Remove these again, then orient each manifold adjacency coherently.
        _, unique = np.unique(np.sort(indices, axis=1), axis=0, return_index=True)
        indices = indices[np.sort(unique)]
        area = np.cross(points[indices[:, 1]] - points[indices[:, 0]], points[indices[:, 2]] - points[indices[:, 0]])
        indices = indices[np.linalg.norm(area, axis=1) > 1e-14]
        used, inverse = np.unique(indices, return_inverse=True)
        points, indices = points[used], inverse.reshape(-1, 3).astype(np.uint32)
        edge_faces = {}
        for i, face in enumerate(indices):
            for a, b in zip(face, np.roll(face, -1)):
                edge_faces.setdefault((min(a, b), max(a, b)), []).append((i, a < b))
        neighbors = [[] for _ in indices]
        for touching in edge_faces.values():
            if len(touching) == 2:
                (a, direction_a), (b, direction_b) = touching
                same = direction_a == direction_b
                neighbors[a].append((b, same))
                neighbors[b].append((a, same))
        flips = np.full(len(indices), -1, dtype=np.int8)
        for seed in range(len(indices)):
            if flips[seed] != -1:
                continue
            component, pending = [], [seed]
            flips[seed] = 0
            while pending:
                face = pending.pop()
                component.append(face)
                for other, same in neighbors[face]:
                    if flips[other] == -1:
                        flips[other] = flips[face] ^ same
                        pending.append(other)
            # Preserve the majority upstream orientation, including open wings.
            if flips[component].sum() > len(component) / 2:
                flips[component] = 1 - flips[component]
        indices[flips == 1] = indices[flips == 1, ::-1]
        if cfg["mirror"]:
            points[:, 1] *= -1
            indices = indices[:, ::-1]
        local_vertices[name] = points.copy()
        matrix = transforms[name]
        vertices[name] = points @ matrix[:3, :3].T + matrix[:3, 3]
        faces[name] = indices

    # Anatomical +X forward, +Y left, +Z up -> Three +Y up, head toward -Z.
    body = np.concatenate([vertices[n] for n in model["segments"] if n in ("c_head", "c_thorax") or "abdomen" in n])
    scale = 2.2 / (body[:, 0].max() - body[:, 0].min())
    ground = min(v[:, 2].min() for n, v in vertices.items() if "tarsus" in n)
    center = (body[:, 0].max() + body[:, 0].min()) / 2
    origin = np.array([center, 0, ground])
    for name in vertices:
        vertices[name] = (vertices[name] - origin) @ rotation.T * scale
    pivots = {name: (t[:3, 3] - origin) @ rotation.T * scale for name, t in transforms.items()}

    # The source wing is one mesh containing a thick proximal hinge shell and
    # the thin membrane. Keep every triangle but give the proximal 0.30 mm
    # opaque cuticle shading; transparent membrane shading made the bundled
    # hinge look like white paper through the thorax/abdomen junction.
    for side in ("l", "r"):
        name = side + "_wing"
        f = faces[name]
        span = local_vertices[name][:, 1] * (1 if side == "l" else -1) - .0080163
        proximal = span[f].mean(axis=1) < .30
        source_vertices, source_local = vertices[name], local_vertices[name]
        for target, triangles in ((name, f[~proximal]), (name + "_base", f[proximal])):
            assert len(triangles)
            used, indices = np.unique(triangles, return_inverse=True)
            vertices[target] = source_vertices[used]
            local_vertices[target] = source_local[used]
            faces[target] = indices.reshape(-1, 3).astype(np.uint32)
        pivots[name + "_base"] = pivots[name]
        axes[name + "_base"] = {}

    groups = {"body": [], "head": ["c_head"],
              "mouth_rostrum": ["c_rostrum"], "mouth_tip": ["c_haustellum"],
              "antenna_L": ["l_pedicel", "l_funiculus", "l_arista"],
              "antenna_R": ["r_pedicel", "r_funiculus", "r_arista"],
              "eye_L": ["l_eye"], "eye_R": ["r_eye"], "wing_L": ["l_wing"], "wing_R": ["r_wing"],
              "hinge_L": ["l_wing_base"], "hinge_R": ["r_wing_base"]}
    node_parents = {name: "head" for name in ("eye_L", "eye_R", "antenna_L", "antenna_R")}
    node_parents.update(mouth_rostrum="head", mouth_tip="mouth_rostrum")
    node_parents.update(hinge_L="wing_L", hinge_R="wing_R")
    group_pivots = {"body": np.zeros(3)}
    for side in ("L", "R"):
        for limb in ("F", "M", "H"):
            prefix = (side + limb).lower()
            for part, pieces in (("coxa", ["coxa"]), ("femur", ["trochanterfemur"]), ("tibia", ["tibia"] + [f"tarsus{i}" for i in range(1, 6)])):
                group = f"leg_{side}{limb}_{part}"
                groups[group] = [f"{prefix}_{p}" for p in pieces]
                group_pivots[group] = pivots[groups[group][0]]
                if part != "coxa":
                    node_parents[group] = f"leg_{side}{limb}_{'coxa' if part == 'femur' else 'femur'}"
    assigned = {n for names in groups.values() for n in names}
    groups["body"] = [n for n in model["segments"] if n not in assigned]
    for group in ("head", "mouth_rostrum", "mouth_tip", "antenna_L", "antenna_R", "eye_L", "eye_R", "wing_L", "wing_R", "hinge_L", "hinge_R"):
        group_pivots[group] = pivots[groups[group][0]]

    gltf = {"asset": {"version": "2.0", "generator": "FLY//BRAIN prepare_fly.py; NeuroMechFly mesh derivative"}, "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [], "materials": [
        {"name": "Cuticle", "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 0.56}},
        {"name": "Compound eyes", "pbrMetallicRoughness": {"baseColorFactor": [0.42, 0.055, 0.025, 1], "metallicFactor": 0.08, "roughnessFactor": 0.31}},
        {"name": "Wing membrane", "pbrMetallicRoughness": {"baseColorFactor": [0.71, 0.75, 0.69, 0.18], "metallicFactor": 0, "roughnessFactor": 0.4}, "alphaMode": "BLEND", "doubleSided": True},
    ], "buffers": [], "bufferViews": [], "accessors": []}
    binary = bytearray()

    def accessor(array, component, kind, target, bounds=False):
        binary.extend(b"\0" * (-len(binary) % 4))
        offset = len(binary)
        raw = array.tobytes()
        binary.extend(raw)
        view = len(gltf["bufferViews"])
        gltf["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw), "target": target})
        data = {"bufferView": view, "componentType": component, "count": len(array), "type": kind}
        if bounds:
            data.update(min=array.min(axis=0).tolist(), max=array.max(axis=0).tolist())
        gltf["accessors"].append(data)
        return len(gltf["accessors"]) - 1

    names = list(groups)
    for group, parts in groups.items():
        chunks, triangles, colors, count = [], [], [], 0
        for part in parts:
            v = vertices[part] - group_pivots[group]
            f = faces[part]
            chunks.append(v)
            triangles.append(f + count)
            count += len(v)
            # Appearance-only pigmentation, not measured CT color. Posterior
            # abdominal tergite bands are defined per segment in its own frame.
            color = np.tile([.51, .37, .21], (len(v), 1))
            if "abdomen" in part:
                local = local_vertices[part]
                extent = np.maximum(np.ptp(local, axis=0), 1e-9)
                x = (local[:, 0] - local[:, 0].min()) / extent[0]
                z = (local[:, 2] - local[:, 2].min()) / extent[2]
                band = np.clip((.38 - x) / .12, 0, 1)
                dorsal = np.clip((z - .15) / .28, 0, 1)
                pigment = (band * dorsal)[:, None]
                color = np.array([.62, .47, .27]) * (1-pigment) + np.array([.095, .06, .035]) * pigment
            elif part in ("c_head", "c_rostrum", "c_haustellum"):
                color[:] = [.37, .26, .14]
            elif any(word in part for word in ("pedicel", "funiculus", "arista")):
                color[:] = [.26, .18, .10]
            color = np.where(color <= .04045, color / 12.92, ((color + .055) / 1.055) ** 2.4)
            colors.append(color)
        v = np.concatenate(chunks).astype("<f4")
        f = np.concatenate(triangles)
        normals = np.zeros(v.shape, dtype=np.float64)
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        area_normal = np.cross(b - a, c - a)
        for i in range(3):
            np.add.at(normals, f[:, i], area_normal)
        # At a residual pinched seam, use its largest incident face rather than
        # a zero normal produced by exactly cancelling thin surfaces.
        for vertex in np.flatnonzero(np.linalg.norm(normals, axis=1) < 1e-12):
            touching = np.flatnonzero(np.any(f == vertex, axis=1))
            assert len(touching)
            normals[vertex] = area_normal[touching[np.argmax(np.linalg.norm(area_normal[touching], axis=1))]]
        normals = (normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)).astype("<f4")
        attrs = {"POSITION": accessor(v, 5126, "VEC3", 34962, True), "NORMAL": accessor(normals, 5126, "VEC3", 34962)}
        if group.startswith("eye"):
            centered = v - (v.min(axis=0) + v.max(axis=0)) / 2
            spherical = centered / np.maximum(np.ptp(v, axis=0) / 2, 1e-9)
            spherical /= np.maximum(np.linalg.norm(spherical, axis=1, keepdims=True), 1e-12)
            uv = np.stack((np.arctan2(spherical[:, 2], spherical[:, 0]) / (2*np.pi) + .5,
                           np.arccos(np.clip(spherical[:, 1], -1, 1)) / np.pi), axis=1)
            attrs["TEXCOORD_0"] = accessor(uv.astype("<f4"), 5126, "VEC2", 34962)
        elif group.startswith("wing"):
            centered = v - v.mean(axis=0)
            _, _, basis = np.linalg.svd(centered, full_matrices=False)
            uv = centered @ basis[:2].T
            uv = (uv - uv.min(axis=0)) / np.maximum(np.ptp(uv, axis=0), 1e-9)
            attrs["TEXCOORD_0"] = accessor(uv.astype("<f4"), 5126, "VEC2", 34962)
        if not group.startswith(("eye", "wing")):
            attrs["COLOR_0"] = accessor(np.concatenate(colors).astype("<f4"), 5126, "VEC3", 34962)
        assert len(v) < 65536
        indices = accessor(f.reshape(-1).astype("<u2"), 5123, "SCALAR", 34963)
        mat = 1 if group.startswith("eye") else 2 if group.startswith("wing") else 0
        gltf["meshes"].append({"name": group, "primitives": [{"attributes": attrs, "indices": indices, "material": mat}]})
        parent = node_parents.get(group)
        pivot = group_pivots[group] - (group_pivots[parent] if parent else 0)
        extras = dict(axes[parts[0]]) if group != "body" else {}
        if "pitchAxis" in extras:
            extras["hingeAxis"] = extras["pitchAxis"]
        if group.endswith("_tibia"):
            foot = vertices[parts[-1]]
            contact = foot[foot[:, 1] <= foot[:, 1].min() + .02].mean(axis=0)
            extras["footContact"] = (contact - group_pivots[group]).tolist()
        gltf["nodes"].append({"name": group, "mesh": len(gltf["meshes"]) - 1, "translation": pivot.tolist(), "extras": extras})
    for i, name in enumerate(names):
        if name in node_parents:
            gltf["nodes"][names.index(node_parents[name])].setdefault("children", []).append(i)
        else:
            gltf["scenes"][0]["nodes"].append(i)
    gltf["buffers"] = [{"byteLength": len(binary)}]
    encoded = json.dumps(gltf, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary.extend(b"\0" * (-len(binary) % 4))
    total = 12 + 8 + len(encoded) + 8 + len(binary)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fly.glb").write_bytes(struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded + struct.pack("<II", len(binary), 0x004E4942) + binary)
    for file in ("LICENSE", "ORIGINAL-LICENSE", "EDITOR-LICENSE", "NOTICE"):
        shutil.copyfile(CACHE / file, OUT / file)
    with (OUT / "NOTICE").open("a") as notice:
        notice.write("\nFLY//BRAIN derivative: pinned fly-svg-maker " + REV + ".\n"
                     "High-resolution visual surfaces from NeuroMechFly " + VISUAL_REV + ".\n"
                     "Meshes posed, mirrored, decimated and merged; shell segmentation noise lightly smoothed;\n"
                     "duplicate faces removed, normals and materials rebuilt; authored appearance-only\n"
                     "pigmentation and UVs added; anatomical joint-axis metadata exported;\n"
                     "original mouth surfaces registered to the v2 folded rest frames;\n"
                     "rostrum and haustellum retain their anatomical pivots as a separate\n"
                     "head-parented mouth chain; articulation does not alter the resting surfaces;\n"
                     "original wing surfaces registered to the v2 frame (+15.21 degree local pitch);\n"
                     "proximal 0.30 mm wing hinge triangles retain opaque cuticle shading;\n"
                     "coordinates transformed and joints retained for illustrative animation.\n"
                     "The micro-CT adult female morphology is a visual reference, not the Male CNS specimen.\n")
    source_files = sorted(CACHE.glob("meshes/*.stl")) + sorted(CACHE.glob("highres/*.stl")) + [CACHE / "model.json"]
    stats = {"source": "https://github.com/NeLy-EPFL/fly-svg-maker", "sourceCommit": REV, "visualSource": "https://github.com/NeLy-EPFL/NeuroMechFly", "visualSourceCommit": VISUAL_REV, "anatomy": "NeuroMechFly adult female micro-CT reference; not the Male CNS specimen", "coordinates": "+Y up; head toward -Z; feet on y=0; head-abdomen length 2.2", "modifications": "Original visual meshes replace collapsed collision surfaces. Duplicate faces removed; neutral-pose FK baked; right side mirrored; shells decimated and Taubin-smoothed; smooth normals; authored natural-brown pigmentation with abdominal bands; eye/wing UVs. Original mouth and wing geometry registered to the folded v2 rest frames. Rostrum and haustellum form an articulated head-parented mouth chain with original anatomical pivots and unchanged resting surfaces. Proximal 0.30 mm wing hinge triangles use opaque cuticle shading without removing geometry. Head/antenna hierarchy and anatomical pitch/roll/yaw axis metadata preserved for illustrative animation.", "triangles": sum(len(f) for f in faces.values()), "drawCalls": len(groups), "bytes": total, "sha256": hashlib.sha256((OUT / "fly.glb").read_bytes()).hexdigest(), "sourceHashes": {str(p.relative_to(CACHE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}
    (OUT / "provenance.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: stats[k] for k in ("triangles", "drawCalls", "bytes")}, indent=2))


def self_check() -> None:
    import numpy as np

    data = (OUT / "fly.glb").read_bytes()
    magic, version, total = struct.unpack_from("<III", data)
    assert (magic, version, total) == (0x46546C67, 2, len(data))
    json_length, chunk = struct.unpack_from("<II", data, 12)
    assert chunk == 0x4E4F534A
    gltf = json.loads(data[20:20 + json_length])
    binary_length, chunk = struct.unpack_from("<II", data, 20 + json_length)
    assert chunk == 0x004E4942
    binary = data[28 + json_length:]
    assert binary_length == len(binary)
    assert len(data) < 3_000_000, "Fly exceeded the 3MB uncompressed asset budget"
    assert len(gltf["meshes"]) == 30

    def read(index):
        a = gltf["accessors"][index]
        view = gltf["bufferViews"][a["bufferView"]]
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
        dtype = {5126: "<f4", 5123: "<u2"}[a["componentType"]]
        out = np.frombuffer(binary, dtype=dtype, offset=view["byteOffset"], count=a["count"] * width).reshape(-1, width)
        assert out.nbytes == view["byteLength"] and np.isfinite(out).all()
        return out

    count = 0
    for mesh in gltf["meshes"]:
        for primitive in mesh["primitives"]:
            vertices = read(primitive["attributes"]["POSITION"])
            normals = read(primitive["attributes"]["NORMAL"])
            faces = read(primitive["indices"])
            assert faces.max() < len(vertices) and len(faces) % 3 == 0
            lengths = np.linalg.norm(normals, axis=1)
            assert np.all(np.abs(lengths - 1) < 1e-5), "Missing or nonunit vertex normal"
            triangles = faces.reshape(-1, 3)
            assert len(triangles) == len(np.unique(np.sort(triangles, axis=1), axis=0)), "Duplicate or opposed faces"
            if mesh["name"].startswith(("eye", "wing")):
                uv = read(primitive["attributes"]["TEXCOORD_0"])
                assert np.all((uv >= 0) & (uv <= 1))
            count += len(faces) // 3
    assert 30000 < count < 150000
    names = {n["name"] for n in gltf["nodes"]}
    assert {"body", "head", "mouth_rostrum", "mouth_tip", "antenna_L", "antenna_R", "eye_L", "eye_R", "wing_L", "wing_R", "hinge_L", "hinge_R", "leg_LF_tibia", "leg_RH_tibia"} <= names
    for parent_name, child_name in (("head", "mouth_rostrum"), ("mouth_rostrum", "mouth_tip")):
        child = next(i for i, n in enumerate(gltf["nodes"]) if n["name"] == child_name)
        parent = next(n for n in gltf["nodes"] if n["name"] == parent_name)
        assert child in parent["children"], "Mouth must retain its anatomical joint chain"
        node = gltf["nodes"][child]
        primitive = gltf["meshes"][node["mesh"]]["primitives"][0]
        assert primitive["material"] == 0 and "COLOR_0" in primitive["attributes"]
        assert gltf["materials"][primitive["material"]].get("alphaMode", "OPAQUE") == "OPAQUE"
        colors = read(primitive["attributes"]["COLOR_0"])
        assert np.all((colors >= 0) & (colors <= 1))
        assert "hingeAxis" in node["extras"]
    for side in ("L", "R"):
        hinge = next(i for i, n in enumerate(gltf["nodes"]) if n["name"] == f"hinge_{side}")
        wing = next(n for n in gltf["nodes"] if n["name"] == f"wing_{side}")
        assert hinge in wing["children"]
        primitive = gltf["meshes"][gltf["nodes"][hinge]["mesh"]]["primitives"][0]
        assert primitive["material"] == 0 and "COLOR_0" in primitive["attributes"]
    for node in gltf["nodes"]:
        for key, value in node.get("extras", {}).items():
            if key.endswith("Axis"):
                assert abs(np.linalg.norm(value) - 1) < 1e-6
        if node["name"].startswith(("wing", "leg")):
            assert "hingeAxis" in node["extras"]
    world_points, visited = [], set()

    def visit(index, parent):
        assert index not in visited, "Cyclic or shared fly rig node"
        visited.add(index)
        node = gltf["nodes"][index]
        position = parent + np.array(node["translation"])
        assert np.isfinite(position).all()
        mesh = gltf["meshes"][node["mesh"]]["primitives"][0]
        world_points.append(read(mesh["attributes"]["POSITION"]) + position)
        for child in node.get("children", []):
            visit(child, position)

    for index in gltf["scenes"][0]["nodes"]:
        visit(index, np.zeros(3))
    assert len(visited) == 30
    assert abs(np.concatenate(world_points)[:, 1].min()) < 1e-6, "Fly feet must sit on y=0"
    provenance = json.loads((OUT / "provenance.json").read_text())
    assert provenance["sha256"] == hashlib.sha256(data).hexdigest()
    assert provenance["triangles"] == count
    assert provenance["drawCalls"] == len(gltf["meshes"])
    print(f"Fly asset self-check passed: {count:,} triangles, 30 draws, {len(data):,} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if not args.self_check:
        acquire()
        build()
    self_check()
