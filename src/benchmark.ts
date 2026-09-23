import {
  BufferAttribute,
  BufferGeometry,
  PerspectiveCamera,
  Points,
  PointsNodeMaterial,
  Scene,
  WebGPURenderer,
} from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { monitor } from "./performance";

export async function benchmark() {
  document.querySelector("#app")?.remove();
  document.body.style.cssText =
    "margin:0;background:#090e11;color:#c2dadd;font:14px monospace";
  const report = document.createElement("pre");
  report.style.cssText =
    "position:fixed;left:24px;top:20px;pointer-events:none";
  report.textContent = "Synthetic rendering benchmark — initializing";
  document.body.append(report);
  const renderer = new WebGPURenderer({
    antialias: false,
    trackTimestamp: true,
    forceWebGL: new URLSearchParams(location.search).has("webgl"),
  });
  await renderer.init();
  renderer.setSize(innerWidth, innerHeight);
  document.body.append(renderer.domElement);
  const scene = new Scene();
  const camera = new PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 100);
  camera.position.z = 7;
  const controls = new OrbitControls(camera, renderer.domElement);
  const count = 166700;
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const angle = i * 2.399963229728653;
    const y = 1 - (2 * i) / count;
    const r = Math.sqrt(1 - y * y);
    positions.set([r * Math.cos(angle) * 2, y * 2, r * Math.sin(angle)], i * 3);
  }
  const geometry = new BufferGeometry().setAttribute(
    "position",
    new BufferAttribute(positions, 3),
  );
  const cloud = new Points(
    geometry,
    new PointsNodeMaterial({ color: 0xaed8db }),
  );
  scene.add(cloud);
  const perf = monitor(renderer);
  Object.assign(window, { flyBrainBenchmark: perf.stats });
  renderer.setAnimationLoop((now) => {
    const start = performance.now();
    cloud.rotation.y = now / 20000;
    controls.update();
    renderer.render(scene, camera);
    if (perf.sample(now, performance.now() - start, positions.byteLength)) {
      report.textContent = `SYNTHETIC BENCHMARK · ${count.toLocaleString()} points\n${JSON.stringify(perf.stats, null, 2)}`;
    }
  });
  window.addEventListener("resize", () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });
}
