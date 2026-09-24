#!/usr/bin/env python3
"""Rebuild the CC0 MakeHuman pointing hand with installed Blender; no runtime dependency.

python tools/data-processing/prepare_finger_asset.py [--blender /path/to/blender]
python tools/data-processing/prepare_finger_asset.py --self-check
Raw sources remain in ignored .cache/finger-asset. The result is an authored
human base mesh, not a scan or a measured sensory/biomechanical model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache/finger-asset"
OUT = ROOT / "public/finger"
REV = "a8bc2d54ff0ac92e78ff71431b1023eda42bf482"
REPO = "https://github.com/makehumancommunity/makehuman"
RAW = f"https://raw.githubusercontent.com/makehumancommunity/makehuman/{REV}/"
SOURCES = {
    "base.obj": ("makehuman/data/3dobjs/base.obj", "8e761e6624b8f54536409135d1636da63b32486a90d4897f84e121d144f6fb4c"),
    "default_weights.mhw": ("makehuman/data/rigs/default_weights.mhw", "0f3641d651ae3d00ad6b4ccee43142edb109d3bd909d27d9e4139ef1beed8625"),
    "LICENSE.md": ("LICENSE.md", "edd99571ca62698f78c943fd4fffb159a413c315fa0d4819d3cec5cadaf8b4f1"),
    "LICENSE.ASSETS.md": ("LICENSE.ASSETS.md", "f6089cba01cb570a24712b41ab8a586ccd3cc5ef53dc266ca50b95c288956d2c"),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def acquire() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, (source, expected) in SOURCES.items():
        path = CACHE / name
        if not path.exists():
            with urlopen(RAW + source, timeout=60) as response:
                data = response.read()
            if sha(data) != expected:
                raise ValueError(f"Unexpected source hash: {name}")
            path.write_bytes(data)
        if sha(path.read_bytes()) != expected:
            raise ValueError(f"Unexpected cached source hash: {name}")


def extract() -> None:
    import bpy
    import bmesh
    from mathutils import Vector, Matrix
    from math import radians

    acquire()
    vertices, groups, group = [], {}, ""
    for line in (CACHE / "base.obj").read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append(tuple(map(float, parts[1:4])))
        elif parts[0] == "g":
            group = parts[1]
        elif parts[0] == "f":
            groups.setdefault(group, []).append([int(p.split("/")[0]) - 1 for p in parts[1:]])

    def center(name):
        indices = sorted({i for face in groups[name] for i in face})
        return sum((Vector(vertices[i]) for i in indices), Vector()) / len(indices)

    base, tip = center("joint-l-finger-2-1"), center("joint-l-finger-2-4")
    axis = (tip - base).normalized()
    # Official CC0 skin weights give a static pointing pose. The index stays
    # unchanged; rotations are illustrative, not a biomechanical simulation.
    weights = json.loads((CACHE / "default_weights.mhw").read_text())["weights"]
    source_vertices = [Vector(vertex) for vertex in vertices]
    curled = [vertex.copy() for vertex in source_vertices]
    flex_axis = (center("joint-l-finger-5-1") - base).normalized()
    for digit in (1, 3, 4, 5):
        transform = Matrix.Identity(4)
        for segment, degrees in enumerate((20, 45, 45) if digit == 1 else (58, 85, 45), 1):
            joint = center(f"joint-l-finger-{digit}-{segment}")
            rotation = Matrix.Rotation(radians(degrees), 4, flex_axis)
            transform = transform @ Matrix.Translation(joint) @ rotation @ Matrix.Translation(-joint)
            for index, weight in weights[f"finger{digit}-{segment}.L"]:
                original = source_vertices[index]
                curled[index] += (transform @ original - original) * weight
    vertices = [tuple(vertex) for vertex in curled]
    indices = sorted({i for face in groups["body"] for i in face})
    remap = {original: local for local, original in enumerate(indices)}
    mesh = bpy.data.meshes.new("MakeHuman body source")
    mesh.from_pydata([vertices[i] for i in indices], [], [[remap[i] for i in f] for f in groups["body"]])
    bm = bmesh.new()
    bm.from_mesh(mesh)
    wrist, elbow = center("joint-l-hand"), center("joint-l-elbow")
    # Keep attached hand and forearm.
    bmesh.ops.bisect_plane(
        bm, geom=list(bm.verts) + list(bm.edges) + list(bm.faces), dist=1e-6,
        plane_co=wrist.lerp(elbow, 0.65), plane_no=(wrist - elbow).normalized(), clear_inner=True,
    )
    seed = min(bm.verts, key=lambda vertex: (vertex.co - tip).length)
    connected, stack = {seed}, [seed]
    while stack:
        for edge in stack.pop().link_edges:
            for vertex in edge.verts:
                if vertex not in connected:
                    connected.add(vertex)
                    stack.append(vertex)
    assert 1500 < len(connected) < 2200, "Unexpected hand component"
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v not in connected], context="VERTS")
    bmesh.ops.holes_fill(bm, edges=[edge for edge in bm.edges if edge.is_boundary], sides=0)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    assert all(edge.is_manifold for edge in bm.edges), "Hand must be closed"

    y = -axis
    x = Vector((1, 0, 0))
    x = (x - y * x.dot(y)).normalized()
    z = x.cross(y).normalized()
    for vertex in bm.verts:
        local = vertex.co - tip
        vertex.co = Vector((local.dot(x), local.dot(y), local.dot(z)))
    result = bpy.data.meshes.new("Pointing hand")
    bm.to_mesh(result)
    bm.free()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    obj = bpy.data.objects.new("MakeHuman left pointing hand", result)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    for polygon in result.polygons:
        polygon.use_smooth = True
    subdivision = obj.modifiers.new("Surface subdivision", "SUBSURF")
    subdivision.levels = subdivision.render_levels = 2
    bpy.ops.object.modifier_apply(modifier=subdivision.name)
    # Blender +Y becomes glTF -Z. Put an actual distal surface vertex at the
    # origin so world placement and the projected drag handle share one tip.
    distal = min((vertex for vertex in obj.data.vertices if vertex.co.length < 0.18), key=lambda vertex: vertex.co.y).co.copy()
    # A rigid -120 degree glTF Z roll raises the palm above the ground while
    # preserving the index axis, exact tip origin and spherical stimulus pad.
    roll = Matrix.Rotation(radians(120), 3, "Y")
    for vertex in obj.data.vertices:
        vertex.co = roll @ (vertex.co - distal)
    material = bpy.data.materials.new("Illustrative skin")
    material.diffuse_color = (0.54, 0.29, 0.2, 1)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    shader = nodes.get("Principled BSDF") or nodes.new("ShaderNodeBsdfPrincipled")
    output = nodes.get("Material Output") or nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    shader.inputs["Base Color"].default_value = material.diffuse_color
    shader.inputs["Roughness"].default_value = 0.6
    obj.data.materials.append(material)
    OUT.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(OUT / "finger.glb"), export_format="GLB", use_selection=True)
    provenance = {
        "source": f"{REPO}/blob/{REV}/{SOURCES['base.obj'][0]}",
        "sourceCommit": REV, "sourceSHA256": SOURCES["base.obj"][1],
        "license": "CC0-1.0", "licensePolicy": f"{REPO}/blob/{REV}/LICENSE.md",
        "weightsSource": f"{REPO}/blob/{REV}/{SOURCES['default_weights.mhw'][0]}",
        "weightsSHA256": SOURCES["default_weights.mhw"][1],
        "preparation": "Left hand with attached forearm; crop 65% from wrist toward elbow, capped beyond scene; static pointing pose using official skin weights; two Catmull-Clark levels; index distal surface vertex recentered; -120 degree roll about glTF Z",
        "poseDegrees": {"thumb": [20, 45, 45], "middleRingLittle": [58, 85, 45], "index": [0, 0, 0]},
        "rollDegrees": -120,
        "coordinates": "Y-up glTF, index distal vertex at origin, complete hand behind tip along -Z; runtime scale 3.2 and tip height 0.95",
        "vertices": len(obj.data.vertices),
        "triangles": sum(len(p.vertices) - 2 for p in obj.data.polygons),
        "blenderVersion": bpy.app.version_string,
        "bytes": (OUT / "finger.glb").stat().st_size,
        "sha256": sha((OUT / "finger.glb").read_bytes()),
        "limitations": "Authored MakeHuman base human mesh, not a medical scan or measured physiology. Skin and scale are illustrative.",
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    shutil.copyfile(CACHE / "LICENSE.ASSETS.md", OUT / "LICENSE")
    (OUT / "NOTICE").write_text(
        "Pointing hand and forearm derived from the official MakeHuman hm08 base mesh.\n"
        f"Source: {provenance['source']}\n"
        "Original asset explicitly released under CC0 1.0 Universal in September 2020.\n"
        "Source copyright holders: Data Collection AB; Joel Palmius; Jonas Hauquier (2020).\n"
        "Official skin weights are also CC0, copyright Data Collection AB; Joel Palmius; Jonas Hauquier (2021).\n"
        f"Weights: {provenance['weightsSource']}\n"
        "Derivative: connected left hand and forearm, capped forearm crop, static pointing pose using official weights,\n"
        "two subdivision levels, index tip recentered, rigid -120 degree Z roll to clear the floor.\n"
        "Authored human mesh, NOT a medical scan or physiological measurement. Skin and scale are illustrative.\n"
        "Only graphical asset data is bundled, not MakeHuman application code.\n"
        "See LICENSE and provenance.json.\n"
    )
    self_check()


def self_check() -> None:
    data = (OUT / "finger.glb").read_bytes()
    provenance = json.loads((OUT / "provenance.json").read_text())
    assert sha(data) == provenance["sha256"]
    assert len(data) == provenance["bytes"] < 1_250_000
    magic, version, size = struct.unpack_from("<4sII", data)
    assert (magic, version, size) == (b"glTF", 2, len(data))
    length, kind = struct.unpack_from("<II", data, 12)
    assert kind == 0x4E4F534A
    gltf = json.loads(data[20:20 + length])
    assert len(gltf["meshes"]) == 1
    primitive = gltf["meshes"][0]["primitives"][0]
    positions = gltf["accessors"][primitive["attributes"]["POSITION"]]
    assert positions["count"] == provenance["vertices"]
    # The index is foremost, with no second finger leading the sensory pad.
    assert abs(positions["max"][2]) < 1e-6 and positions["min"][2] < -2.5
    assert positions["min"][1] * 3.2 + 0.95 > 0.3, "Hand intersects the ground"
    view = gltf["bufferViews"][positions["bufferView"]]
    offset = 28 + length + view.get("byteOffset", 0) + positions.get("byteOffset", 0)
    stride = view.get("byteStride", 12)
    assert any(struct.unpack_from("<fff", data, offset + i * stride) == (0, 0, 0) for i in range(positions["count"])), "Exact fingertip anchor missing"
    assert sha((OUT / "LICENSE").read_bytes()) == SOURCES["LICENSE.ASSETS.md"][1]
    print(f"Pointing hand verified: {provenance['triangles']} triangles, {len(data)} bytes; CC0.")


if __name__ == "__main__":
    if "--extract" in sys.argv:
        extract()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--self-check", action="store_true")
        parser.add_argument("--blender", default=shutil.which("blender") or "/Applications/Blender.app/Contents/MacOS/Blender")
        args = parser.parse_args()
        if args.self_check:
            self_check()
        else:
            acquire()
            subprocess.run([args.blender, "--background", "--factory-startup", "--python-exit-code", "1", "--python", str(Path(__file__).resolve()), "--", "--extract"], check=True)
