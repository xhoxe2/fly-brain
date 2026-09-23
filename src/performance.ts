import type { WebGPURenderer } from "three/webgpu";

export function monitor(renderer: WebGPURenderer) {
  const frames: number[] = [];
  const submissions: number[] = [];
  const stats = {
    fps: 0,
    frameMs: 0,
    p95Ms: 0,
    cpuMs: 0,
    gpuMs: null as number | null,
    drawCalls: 0,
    points: 0,
    lines: 0,
    triangles: 0,
    bufferMB: 0,
    gpuTrackedMB: null as number | null,
    samples: 0,
  };
  let last = 0;
  let published = 0;
  let resolving = false;
  return {
    stats,
    sample(now: number, cpu: number, bytes: number) {
      if (last && !document.hidden) {
        frames.push(now - last);
        submissions.push(cpu);
      }
      last = now;
      if (now - published < 1000 || !frames.length) return false;
      const sorted = [...frames].sort((a, b) => a - b);
      stats.frameMs = frames.reduce((a, b) => a + b, 0) / frames.length;
      stats.fps = 1000 / stats.frameMs;
      stats.p95Ms = sorted[Math.floor(sorted.length * 0.95)];
      stats.cpuMs = submissions.reduce((a, b) => a + b, 0) / submissions.length;
      stats.samples += frames.length;
      const info = renderer.info.render;
      stats.drawCalls = info.drawCalls;
      stats.points = info.points;
      stats.lines = info.lines;
      stats.triangles = info.triangles;
      stats.bufferMB = bytes / 1048576;
      const memory = renderer.info.memory as { total?: number };
      stats.gpuTrackedMB =
        typeof memory.total === "number" ? memory.total / 1048576 : null;
      if (!resolving && renderer.hasFeature("timestamp-query")) {
        resolving = true;
        renderer
          .resolveTimestampsAsync("render")
          .then((value) => {
            if (typeof value === "number" && Number.isFinite(value))
              stats.gpuMs = value;
          })
          .catch(() => {
            stats.gpuMs = null;
          })
          .finally(() => {
            resolving = false;
          });
      }
      published = now;
      frames.length = submissions.length = 0;
      return true;
    },
    reset() {
      last = 0;
      frames.length = submissions.length = 0;
    },
  };
}
