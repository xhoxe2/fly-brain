import type { Network } from "./neural.ts";
import type { GeometryData } from "./data.ts";

export interface DynamicsAsset {
  file: string;
  bytes: number;
  compressedBytes: number;
  sha256: string;
}
export interface DynamicsManifest {
  format: "LIF1";
  neuronCount: number;
  edgeCount: number;
  graph: {
    bytes: number;
    sha256: string;
    chunks: (DynamicsAsset & { offset: number })[];
  };
  ids: DynamicsAsset;
  positions: DynamicsAsset;
  protocol: { file: string };
}
export interface FeedingProtocol {
  inputs: Record<"sugar" | "bitter", { ids: string[]; indices: number[] }>;
  outputs: {
    primary: { id: string; index: number; label: string };
    additional: { id: string; index: number; label: string }[];
  };
}

export interface FingerProtocol {
  inputs: Record<"lc4Left" | "lc4Right" | "lplc2Left" | "lplc2Right", { ids: string[]; indices: number[] }>;
  outputs: Record<"gfLeft" | "gfRight", { id: string; index: number; label: string }>;
}

const base = `${import.meta.env?.BASE_URL ?? "/"}dynamics/`;
export async function dynamicsJSON<T>(file: string): Promise<T> {
  const response = await fetch(base + file);
  if (!response.ok)
    throw new Error(`Cannot load ${file}: HTTP ${response.status}`);
  return response.json();
}

async function checkedAsset(asset: DynamicsAsset): Promise<ArrayBuffer> {
  const response = await fetch(base + asset.file);
  if (!response.ok)
    throw new Error(`Cannot load ${asset.file}: HTTP ${response.status}`);
  const downloaded = await response.arrayBuffer();
  const bytes = new Uint8Array(downloaded);
  const buffer =
    bytes[0] === 0x1f && bytes[1] === 0x8b
      ? await new Response(
          new Blob([downloaded])
            .stream()
            .pipeThrough(new DecompressionStream("gzip")),
        ).arrayBuffer()
      : downloaded;
  if (buffer.byteLength !== asset.bytes)
    throw new Error(`Incomplete ${asset.file}`);
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  const hex = Array.from(new Uint8Array(digest), (n) =>
    n.toString(16).padStart(2, "0"),
  ).join("");
  if (hex !== asset.sha256)
    throw new Error(`Integrity check failed for ${asset.file}`);
  return buffer;
}

export function decodeNetwork(buffer: ArrayBuffer): Network {
  if (buffer.byteLength < 16) throw new Error("Incomplete neural network");
  const header = new DataView(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  const count = header.getUint32(4, true),
    edges = header.getUint32(8, true);
  if (
    magic !== "LIF1" ||
    !count ||
    count > 200_000 ||
    edges > 30_000_000 ||
    buffer.byteLength !== 12 + (count + 1) * 4 + edges * 8
  )
    throw new Error("Unsupported neural network format");
  const offsets = new Uint32Array(buffer, 12, count + 1);
  const targets = new Uint32Array(buffer, 12 + (count + 1) * 4, edges);
  const weights = new Float32Array(buffer, 12 + (count + 1 + edges) * 4, edges);
  return { count, offsets, targets, weights };
}

export async function loadNetwork(
  manifest: DynamicsManifest,
  progress: (fraction: number) => void,
) {
  if (manifest.format !== "LIF1" || manifest.graph.bytes > 260_000_000)
    throw new Error("Unsupported neural network manifest");
  const buffer = new ArrayBuffer(manifest.graph.bytes);
  const destination = new Uint8Array(buffer);
  let offset = 0;
  // One chunk at a time bounds peak memory while preserving every source edge.
  for (const chunk of manifest.graph.chunks) {
    if (chunk.offset !== offset || offset + chunk.bytes > destination.length)
      throw new Error("Neural data chunks are not contiguous");
    destination.set(new Uint8Array(await checkedAsset(chunk)), offset);
    offset += chunk.bytes;
    progress(offset / destination.length);
  }
  if (offset !== destination.length)
    throw new Error("Missing neural data chunk");
  const network = decodeNetwork(buffer);
  if (
    network.count !== manifest.neuronCount ||
    network.targets.length !== manifest.edgeCount
  )
    throw new Error("Neural data counts do not match the manifest");
  return network;
}

export async function loadNeuralAnatomy(manifest: DynamicsManifest) {
  const buffer = await checkedAsset(manifest.positions);
  const count = manifest.neuronCount;
  if (buffer.byteLength !== count * 13)
    throw new Error("Invalid anchor coordinates");
  const raw = new Float32Array(buffer, 0, count * 3);
  const valid = new Uint8Array(buffer, count * 12, count);
  const low = [Infinity, Infinity, Infinity],
    high = [-Infinity, -Infinity, -Infinity];
  let mapped = 0;
  for (let i = 0; i < count; i++) {
    if (valid[i] > 1) throw new Error("Invalid coordinate validity flag");
    if (!valid[i]) continue;
    mapped++;
    for (let axis = 0; axis < 3; axis++) {
      const value = raw[i * 3 + axis];
      if (!Number.isFinite(value)) throw new Error("Invalid anchor position");
      low[axis] = Math.min(low[axis], value);
      high[axis] = Math.max(high[axis], value);
    }
  }
  if (!mapped || high[0] <= low[0])
    throw new Error("Missing anatomical coordinates");
  const positions = new Float32Array(mapped * 3),
    owners = new Uint32Array(mapped);
  // Source annotations contain a few extreme coordinate outliers. Frame the
  // central 99.98% per axis plus padding, retaining every valid point with one rigid scale.
  // This changes the initial framing, never the graph or relative geometry.
  const axes = Array.from({ length: 3 }, () => new Float32Array(mapped));
  for (let i = 0, row = 0; i < count; i++) {
    if (!valid[i]) continue;
    for (let axis = 0; axis < 3; axis++) axes[axis][row] = raw[i * 3 + axis];
    row++;
  }
  for (let axis = 0; axis < 3; axis++) {
    axes[axis].sort();
    low[axis] = axes[axis][Math.floor((mapped - 1) * 0.0001)];
    high[axis] = axes[axis][Math.ceil((mapped - 1) * 0.9999)];
    const padding = (high[axis] - low[axis]) * 0.05;
    low[axis] -= padding;
    high[axis] += padding;
  }
  if (high[0] <= low[0]) throw new Error("Degenerate anatomical coordinates");
  const center = low.map((value, axis) => (value + high[axis]) / 2);
  const scale = 10 / (high[0] - low[0]);
  for (let i = 0, row = 0; i < count; i++) {
    if (!valid[i]) continue;
    owners[row] = i;
    positions[row * 3] = (raw[i * 3] - center[0]) * scale;
    positions[row * 3 + 1] = -(raw[i * 3 + 1] - center[1]) * scale;
    positions[row * 3 + 2] = (raw[i * 3 + 2] - center[2]) * scale;
    row++;
  }
  const overview: GeometryData = {
    kind: 1,
    count: mapped,
    positions,
    owners,
    values: new Uint32Array(mapped),
    bytes: positions.byteLength + owners.byteLength,
  };
  const background: GeometryData = {
    kind: 2,
    count: 0,
    positions: new Float32Array(),
    owners: new Uint32Array(),
    values: new Float32Array(),
    bytes: 0,
  };
  return { overview, background };
}
