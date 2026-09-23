import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { decode, type Circuit } from "./data.ts";
import { simulate, CONDUCTION, type Parameters } from "./simulation.ts";

const circuit: Circuit = JSON.parse(
  readFileSync(new URL("../public/brain/escape.json", import.meta.url), "utf8"),
);
const parameters: Parameters = {
  experiment: "threat",
  direction: 0,
  speed: 1,
  size: 1,
  intensity: 0.8,
  lesion: "none",
};

test("real circuit drives motor output; lesions block it; inputs remain deterministic", () => {
  const normal = simulate(circuit, parameters);
  assert.ok(Number.isFinite(normal.motorAt));
  assert.ok(normal.path.length >= 3);
  assert.equal(circuit.neurons[normal.path.at(-1)!].type, "TTMn");
  for (let i = 1; i < normal.path.length; i++) {
    const previous = normal.path[i - 1],
      current = normal.path[i];
    assert.ok(circuit.edges.some(([a, b]) => a === previous && b === current));
    assert.ok(normal.onsets[current] >= normal.onsets[previous] + CONDUCTION);
  }
  assert.equal(normal.motorAt, normal.onsets[normal.path.at(-1)!] + CONDUCTION);
  for (const lesion of ["visual", "descending", "motor"] as const) {
    assert.equal(
      simulate(circuit, { ...parameters, lesion }).motorAt,
      Infinity,
    );
  }
  assert.equal(
    simulate(circuit, { ...parameters, intensity: 0 }).motorAt,
    Infinity,
  );
  assert.deepEqual(simulate(circuit, parameters).onsets, normal.onsets);
  assert.throws(() => simulate(circuit, { ...parameters, intensity: NaN }));
  const left = simulate(circuit, { ...parameters, direction: -90 });
  const right = simulate(circuit, { ...parameters, direction: 90 });
  assert.notDeepEqual(left.onsets, right.onsets);
});

test("binary exports have valid coordinates, lengths and circuit owners", () => {
  for (const filename of [
    "overview.bin",
    "background.bin",
    "escape.bin",
    "region-0.bin",
    "region-1.bin",
    "region-2.bin",
  ]) {
    const file = gunzipSync(
      readFileSync(new URL(`../public/brain/${filename}.gz`, import.meta.url)),
    );
    const buffer = file.buffer.slice(
      file.byteOffset,
      file.byteOffset + file.byteLength,
    ) as ArrayBuffer;
    const data = decode(buffer);
    assert.ok(data.count > 0);
    if (filename === "escape.bin")
      assert.ok(data.owners.every((owner) => owner < circuit.neurons.length));
    assert.throws(() => decode(buffer.slice(0, -4)));
  }
});
