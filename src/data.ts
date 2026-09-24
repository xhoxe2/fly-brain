export interface Asset {
  file: string;
  bytes: number;
  gzipBytes: number;
  sha256: string;
}
export interface Manifest {
  dataset: string;
  source: string;
  tracedNeurons: number;
  mappedSomas: number;
  omittedWithoutSoma: number;
  overview: Asset;
  background: Asset;
  circuit: string;
  circuits: Record<"escape" | "motion" | "light", string>;
  circuitNeurons: number;
  circuitEdges: number;
  regions: { id: number; name: string; neurons: number; geometry: Asset }[];
}
export interface Neuron {
  id: string;
  type: string;
  name: string;
  side: string;
  region: number;
  group: string;
  position: [number, number, number];
  inputs: number;
  outputs: number;
  role: string;
  path: [number, number, number][];
  polarity: 1 | -1;
}
export interface Circuit {
  kind: "escape" | "motion" | "light";
  action: "jump" | "turn" | "none";
  inputTypes: string[];
  outputTypes: string[];
  stages: {
    id: "visual" | "descending" | "motor";
    label: string;
    detail: string;
    types: string[];
  }[];
  sources: { title: string; url: string }[];
  scientificNote: string;
  neurons: Neuron[];
  edges: [number, number, number][];
  geometry: Asset;
  modelNotice: string;
}
export interface GeometryData {
  kind: number;
  count: number;
  positions: Float32Array;
  owners: Uint32Array;
  values: Float32Array | Uint32Array;
  bytes: number;
}

export function decode(buffer: ArrayBuffer): GeometryData {
  if (buffer.byteLength < 16) throw new Error("Incomplete brain data header");
  const header = new DataView(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  const version = header.getUint32(4, true);
  const kind = header.getUint32(8, true);
  const count = header.getUint32(12, true);
  if (
    magic !== "FLY1" ||
    version !== 1 ||
    ![1, 2].includes(kind) ||
    buffer.byteLength !== 16 + count * 20 ||
    (kind === 2 && count % 2)
  ) {
    throw new Error("Unsupported or damaged brain geometry");
  }
  const positions = new Float32Array(buffer, 16, count * 3);
  if (positions.some((v) => !Number.isFinite(v)))
    throw new Error("Invalid neuron coordinates");
  return {
    kind,
    count,
    positions,
    owners: new Uint32Array(buffer, 16 + count * 12, count),
    values:
      kind === 1
        ? new Uint32Array(buffer, 16 + count * 16, count)
        : new Float32Array(buffer, 16 + count * 16, count),
    bytes: buffer.byteLength,
  };
}

export async function json<T>(name: string): Promise<T> {
  const response = await fetch(`${import.meta.env.BASE_URL}brain/${name}`);
  if (!response.ok)
    throw new Error(`Could not load ${name} (${response.status})`);
  return response.json();
}

export async function geometry(asset: Asset, signal?: AbortSignal) {
  const response = await fetch(
    `${import.meta.env.BASE_URL}brain/${asset.file}`,
    { signal },
  );
  if (!response.ok)
    throw new Error(`Could not load ${asset.file} (${response.status})`);
  const downloaded = await response.arrayBuffer();
  const compressed = new Uint8Array(
    downloaded,
    0,
    Math.min(2, downloaded.byteLength),
  );
  const buffer =
    compressed[0] === 0x1f && compressed[1] === 0x8b
      ? await new Response(
          new Blob([downloaded])
            .stream()
            .pipeThrough(new DecompressionStream("gzip")),
        ).arrayBuffer()
      : downloaded;
  if (buffer.byteLength !== asset.bytes)
    throw new Error(`Incomplete download: ${asset.file}`);
  // Hashes in the manifest also allow offline integrity verification of exports.
  return decode(buffer);
}
