import { simulate, type Parameters } from "./simulation";
import type { Circuit } from "./data";

self.onmessage = (
  event: MessageEvent<{ id: number; circuit: Circuit; parameters: Parameters }>,
) => {
  const { id, circuit, parameters } = event.data;
  try {
    const result = simulate(circuit, parameters);
    const baseline =
      parameters.lesion === "none"
        ? null
        : simulate(circuit, { ...parameters, lesion: "none" });
    self.postMessage(
      { id, result, baseline },
      {
        transfer: [
          result.onsets.buffer,
          result.parents.buffer,
          ...(baseline
            ? [baseline.onsets.buffer, baseline.parents.buffer]
            : []),
        ],
      },
    );
  } catch (error) {
    self.postMessage({
      id,
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
