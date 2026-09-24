import assert from "node:assert/strict";
import test from "node:test";
import { createHash } from "node:crypto";
import { gzipSync } from "node:zlib";
import {
  decodeNetwork,
  loadNetwork,
  loadNeuralAnatomy,
  type DynamicsManifest,
  type DynamicsAsset,
} from "./neural-data.ts";

// Synthetic bytes check transport and identity preservation, not biological validity.
test("neural assets verify gzip/hash/length, preserve CSR identity, and omit only missing display anchors", async () => {
  const network = new ArrayBuffer(56);
  new Uint8Array(network, 0, 4).set([76, 73, 70, 49]);
  new DataView(network).setUint32(4, 4, true);
  new DataView(network).setUint32(8, 3, true);
  new Uint32Array(network, 12, 5).set([0, 2, 2, 3, 3]);
  new Uint32Array(network, 32, 3).set([1, 2, 3]);
  new Float32Array(network, 44, 3).set([12, -7, 23]);
  const coordinates = new ArrayBuffer(52);
  new Float32Array(coordinates, 0, 12).set([
    1, 2, 3, 0, 0, 0, 3, 4, 5, 5, 6, 7,
  ]);
  new Uint8Array(coordinates, 48).set([1, 0, 1, 1]);
  const payloads = new Map<string, Uint8Array>();
  const asset = (file: string, data: ArrayBuffer): DynamicsAsset => {
    const compressed = gzipSync(new Uint8Array(data));
    payloads.set("/dynamics/" + file, compressed);
    return {
      file,
      bytes: data.byteLength,
      compressedBytes: compressed.byteLength,
      sha256: createHash("sha256").update(new Uint8Array(data)).digest("hex"),
    };
  };
  const graphAsset = asset("graph.gz", network);
  const positions = asset("positions.gz", coordinates);
  const manifest: DynamicsManifest = {
    format: "LIF1",
    neuronCount: 4,
    edgeCount: 3,
    graph: {
      bytes: network.byteLength,
      sha256: graphAsset.sha256,
      chunks: [{ ...graphAsset, offset: 0 }],
    },
    positions,
    ids: graphAsset,
    protocol: { file: "unused.json" },
  };
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input) => {
    const bytes = payloads.get(String(input));
    return bytes
      ? new Response(Uint8Array.from(bytes))
      : new Response(null, { status: 404 });
  };
  try {
    const fractions: number[] = [];
    const graph = await loadNetwork(manifest, (value) => fractions.push(value));
    assert.deepEqual([...graph.offsets], [0, 2, 2, 3, 3]);
    assert.deepEqual([...graph.targets], [1, 2, 3]);
    assert.deepEqual([...graph.weights], [12, -7, 23]);
    assert.deepEqual(fractions, [1]);
    const { overview } = await loadNeuralAnatomy(manifest);
    assert.deepEqual([...overview.owners], [0, 2, 3]);
    assert.equal(
      graph.count,
      4,
      "Missing display positions must not delete simulated cells",
    );
    assert.equal(overview.positions[0], -overview.positions[6]);
    assert.equal(overview.positions[3], 0);
    const outlierData = new ArrayBuffer(10001 * 13);
    const outlierPositions = new Float32Array(outlierData, 0, 10001 * 3);
    for (let i = 0; i < 10000; i++) outlierPositions[i * 3] = i;
    outlierPositions[30000] = 1e8;
    new Uint8Array(outlierData, 10001 * 12).fill(1);
    const framed = await loadNeuralAnatomy({
      ...manifest,
      neuronCount: 10001,
      positions: asset("outliers.gz", outlierData),
    });
    assert.equal(
      framed.overview.count,
      10001,
      "Outlier points remain in rendering data",
    );
    assert.ok(
      Math.abs(framed.overview.positions[15000]) < 0.02,
      "A source outlier must not shift the central brain off screen",
    );
    assert.ok(
      framed.overview.positions[30000] > 1000,
      "Outliers retain their actual relative position",
    );
    payloads.set("/dynamics/graph.gz", new Uint8Array(network));
    assert.equal(
      (await loadNetwork(manifest, () => {})).count,
      4,
      "Already decompressed HTTP responses are supported",
    );
    manifest.graph.chunks[0].sha256 = "0".repeat(64);
    await assert.rejects(
      loadNetwork(manifest, () => {}),
      /Integrity check/,
    );
    manifest.graph.chunks[0].offset = 4;
    await assert.rejects(
      loadNetwork(manifest, () => {}),
      /not contiguous/,
    );
    assert.throws(() => decodeNetwork(network.slice(0, -1)), /Unsupported/);
    new Uint8Array(network)[0] = 0;
    assert.throws(() => decodeNetwork(network), /Unsupported/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
