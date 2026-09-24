import { SpikingBrain } from "./neural.ts";
import { FeedingWorld, type FeedingEnvironment } from "./embodiment.ts";
import {
  dynamicsJSON,
  loadNetwork,
  type DynamicsManifest,
  type FeedingProtocol,
  type FingerProtocol,
} from "./neural-data.ts";

let world: FeedingWorld | null = null;
let initialized = false;

self.onmessage = async (
  event: MessageEvent<
    | { type: "init"; seed: number }
    | { type: "reset"; seed: number }
    | {
        type: "advance";
        id: number;
        durationMs: number;
        environment: FeedingEnvironment;
      }
  >,
) => {
  try {
    const message = event.data;
    if (message.type === "init") {
      if (initialized) throw new Error("Neural model is already initialized");
      initialized = true;
      const manifest = await dynamicsJSON<DynamicsManifest>("manifest.json");
      const protocol = await dynamicsJSON<FeedingProtocol>(
        manifest.protocol.file,
      );
      const fingerProtocol = await dynamicsJSON<FingerProtocol>("finger.json");
      const network = await loadNetwork(manifest, (fraction) =>
        self.postMessage({ type: "loading", fraction }),
      );
      world = new FeedingWorld(
        new SpikingBrain(network, message.seed),
        protocol,
        fingerProtocol,
      );
      self.postMessage({
        type: "ready",
        neurons: network.count,
        edges: network.targets.length,
      });
    } else {
      if (!world) throw new Error("Neural model is not ready");
      if (message.type === "reset") {
        world.reset(message.seed);
        self.postMessage({ type: "reset" });
      } else {
        const start = performance.now();
        const result = world.advance(message.durationMs, message.environment);
        const active: number[] = [];
        for (let i = 0; i < result.counts.length; i++) {
          if (result.counts[i]) {
            active.push(i);
          }
        }
        const indices = Uint32Array.from(active);
        const { counts: _, ...state } = result;
        self.postMessage(
          {
            type: "frame",
            id: message.id,
            state,
            indices,
            computeMs: performance.now() - start,
          },
          { transfer: [indices.buffer] },
        );
      }
    }
  } catch (error) {
    self.postMessage({
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
