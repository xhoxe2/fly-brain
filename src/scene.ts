import {
  AdditiveBlending,
  AmbientLight,
  BufferAttribute,
  BufferGeometry,
  Color,
  CylinderGeometry,
  DirectionalLight,
  DoubleSide,
  Group,
  InstancedMesh,
  LineBasicNodeMaterial,
  LineSegments,
  Matrix4,
  Mesh,
  MeshBasicNodeMaterial,
  MeshStandardNodeMaterial,
  PerspectiveCamera,
  Points,
  PointLight,
  PointsNodeMaterial,
  Quaternion,
  Raycaster,
  RingGeometry,
  Scene,
  SphereGeometry,
  Vector2,
  Vector3,
  WebGPURenderer,
} from "three/webgpu";
import { abs, attribute, float, mix, uniform, vec3 } from "three/tsl";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  geometry,
  type Circuit,
  type GeometryData,
  type Manifest,
} from "./data";
import { CONDUCTION, type Parameters, type Result } from "./simulation";

export type View = "brain" | "world" | "split";
export type Quality = "ultra" | "high" | "medium" | "low";
const qualitySettings = {
  ultra: [1, 1.75],
  high: [1, 1.25],
  medium: [0.6, 1],
  low: [0.25, 0.8],
} as const;

function buffers(data: GeometryData) {
  return new BufferGeometry()
    .setAttribute("position", new BufferAttribute(data.positions, 3))
    .setAttribute(
      "owner",
      new BufferAttribute(Float32Array.from(data.owners), 1),
    )
    .setAttribute(
      "progress",
      new BufferAttribute(Float32Array.from(data.values), 1),
    );
}

export class LabScene {
  renderer: WebGPURenderer;
  brain = new Scene();
  world = new Scene();
  camera = new PerspectiveCamera(40, 1, 0.02, 150);
  worldCamera = new PerspectiveCamera(38, 1, 0.05, 100);
  controls: OrbitControls;
  time = uniform(-10);
  selected = uniform(-1);
  isolate = uniform(0);
  view: View = "brain";
  quality: Quality = "high";
  following = false;
  bytes = 0;
  circuit: Circuit | null = null;
  circuitData: GeometryData | null = null;
  result: Result | null = null;
  reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  private points: Points;
  private background: LineSegments;
  private circuitLines: LineSegments | null = null;
  private regionLines: LineSegments | null = null;
  private regionBytes = 0;
  private regionAbort: AbortController | null = null;
  private regionGeneration = 0;
  private regionIndex = -1;
  private fly = new Group();
  private wings: Mesh[] = [];
  private threat: Mesh;
  private lightMarker: Mesh;
  private stimulusLight = new PointLight(0xffe4a4, 0, 8, 1.5);
  private selector: Mesh;
  private raycaster = new Raycaster();
  private mouse = new Vector2();
  private target = new Vector3();
  private position = new Vector3();
  private next = new Vector3();
  private up = new Vector3(0, 1, 0);
  private matrix = new Matrix4();
  private rotation = new Quaternion();
  private scale = new Vector3();
  private aspect = 0;
  private height = 0;
  private width = 0;
  private center = new Vector3(0, 0.2, 0);
  private observer: ResizeObserver;
  onSelect: (index: number) => void = () => {};
  onCameraInterrupted: () => void = () => {};

  constructor(
    private host: HTMLElement,
    private orbit: HTMLElement,
    overview: GeometryData,
    background: GeometryData,
  ) {
    this.renderer = new WebGPURenderer({
      antialias: true,
      alpha: true,
      trackTimestamp: true,
      forceWebGL: new URLSearchParams(location.search).has("webgl"),
    });
    this.renderer.setClearColor(0x090e11, 0);
    this.renderer.info.autoReset = false;
    this.host.prepend(this.renderer.domElement);
    this.renderer.domElement.setAttribute(
      "aria-label",
      "Interactive fruit fly connectome and experiment",
    );
    this.camera.position.set(0, 0.2, 17.5);
    this.controls = new OrbitControls(this.camera, orbit);
    this.controls.target.copy(this.center);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 0.4;
    this.controls.maxDistance = 35;
    this.controls.enablePan = true;
    this.controls.addEventListener("start", () => {
      if (this.following) {
        this.following = false;
        this.onCameraInterrupted();
      }
    });
    let downX = 0,
      downY = 0;
    orbit.addEventListener("pointerdown", (e) => {
      downX = e.clientX;
      downY = e.clientY;
    });
    orbit.addEventListener("pointerup", (e) => {
      if (Math.hypot(e.clientX - downX, e.clientY - downY) < 5) this.pick(e);
    });
    const pointGeometry = new BufferGeometry().setAttribute(
      "position",
      new BufferAttribute(overview.positions, 3),
    );
    const pointMaterial = new PointsNodeMaterial({
      color: 0x76999d,
      transparent: true,
      opacity: 0.44,
      depthWrite: false,
    });
    this.points = new Points(pointGeometry, pointMaterial);
    this.brain.add(this.points);
    const backgroundMaterial = new LineBasicNodeMaterial({
      color: 0x60898e,
      transparent: true,
      opacity: 0.14,
      depthWrite: false,
    });
    this.background = new LineSegments(buffers(background), backgroundMaterial);
    this.brain.add(this.background);
    this.bytes = overview.bytes + background.bytes;
    this.selector = new Mesh(
      new SphereGeometry(0.045, 12, 8),
      new MeshBasicNodeMaterial({ color: 0xf3dbaa }),
    );
    this.selector.visible = false;
    this.brain.add(this.selector);
    this.threat = new Mesh(
      new SphereGeometry(0.35, 32, 20),
      new MeshStandardNodeMaterial({
        color: 0xce7756,
        roughness: 0.38,
        metalness: 0.12,
      }),
    );
    this.lightMarker = new Mesh(
      new SphereGeometry(0.08, 12, 8),
      new MeshBasicNodeMaterial({ color: 0xffe5a7 }),
    );
    this.world.add(this.threat, this.lightMarker, this.stimulusLight);
    this.makeWorld();
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(host);
    this.raycaster.params.Line = { threshold: 0.07 };
  }

  async init() {
    await this.renderer.init();
    this.setQuality("high");
    this.resize();
    return this;
  }

  get backend() {
    return "isWebGPUBackend" in this.renderer.backend ? "WebGPU" : "WebGL2";
  }

  setView(view: View) {
    this.view = view;
    this.orbit.style.left = view === "split" ? "46%" : "0";
    this.orbit.style.display = view === "world" ? "none" : "block";
    this.aspect = 0;
  }

  setQuality(quality: Quality) {
    this.quality = quality;
    const [density, pixelRatio] = qualitySettings[quality];
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, pixelRatio));
    this.points.geometry.setDrawRange(
      0,
      Math.floor(this.points.geometry.attributes.position.count * density),
    );
    this.background.geometry.setDrawRange(
      0,
      Math.floor(
        (this.background.geometry.attributes.position.count * density) / 2,
      ) * 2,
    );
    this.resize();
  }

  async loadCircuit(circuit: Circuit) {
    const data = await geometry(circuit.geometry);
    if (data.owners.some((owner) => owner >= circuit.neurons.length))
      throw new Error("Invalid circuit geometry owner");
    this.circuit = circuit;
    this.circuitData = data;
    const shape = buffers(data);
    shape.setAttribute(
      "onset",
      new BufferAttribute(new Float32Array(data.count).fill(1e6), 1),
    );
    shape.setAttribute(
      "emphasis",
      new BufferAttribute(new Float32Array(data.count).fill(1), 1),
    );
    const progress = attribute<"float">("progress", "float");
    const distance = abs(
      this.time
        .sub(attribute<"float">("onset", "float"))
        .div(CONDUCTION)
        .sub(progress),
    );
    const pulse = float(1)
      .sub(distance.div(0.14))
      .clamp(0, 1)
      .mul(float(progress.greaterThanEqual(0)));
    const chosen = float(
      attribute<"float">("owner", "float").equal(this.selected),
    );
    const emphasis = attribute<"float">("emphasis", "float");
    const material = new LineBasicNodeMaterial({
      transparent: true,
      depthWrite: false,
      blending: AdditiveBlending,
    });
    material.colorNode = mix(
      vec3(0.061, 0.14, 0.165),
      vec3(0.905, 0.672, 0.356),
      pulse.max(chosen.mul(0.7)),
    );
    material.opacityNode = float(0.18)
      .add(pulse.mul(0.82))
      .add(chosen.mul(0.35))
      .mul(mix(float(1), emphasis, this.isolate));
    this.circuitLines = new LineSegments(shape, material);
    this.brain.add(this.circuitLines);
    this.bytes += data.bytes + data.count * 12;
  }

  setResult(result: Result | null) {
    this.result = result;
    if (!this.circuitLines || !this.circuitData) return;
    const onset = this.circuitLines.geometry.getAttribute(
      "onset",
    ) as BufferAttribute;
    for (let i = 0; i < this.circuitData.count; i++) {
      const value = result?.onsets[this.circuitData.owners[i]];
      onset.setX(
        i,
        value !== undefined && Number.isFinite(value) ? value : 1e6,
      );
    }
    onset.needsUpdate = true;
  }

  select(index: number, focus = false) {
    this.selected.value = index;
    this.selector.visible = index >= 0;
    if (index < 0 || !this.circuit) return;
    this.selector.position.fromArray(this.circuit.neurons[index].position);
    if (focus) {
      this.controls.target.copy(this.selector.position);
      this.camera.position
        .copy(this.selector.position)
        .add(this.position.set(0, 0.4, 4));
      this.controls.update();
    }
  }

  showConnections(mode: "all" | "inputs" | "outputs" | "isolate") {
    if (!this.circuitData || !this.circuitLines || !this.circuit) return [];
    const selected = this.selected.value;
    const edges = this.circuit.edges
      .filter(([a, b]) => (mode === "inputs" ? b === selected : a === selected))
      .sort((a, b) => b[2] - a[2])
      .slice(0, 12);
    const indices = new Set([selected]);
    if (mode !== "isolate")
      edges.forEach(([a, b]) => {
        indices.add(a);
        indices.add(b);
      });
    const attribute = this.circuitLines.geometry.getAttribute(
      "emphasis",
    ) as BufferAttribute;
    for (let i = 0; i < this.circuitData.count; i++)
      attribute.setX(i, indices.has(this.circuitData.owners[i]) ? 1 : 0.025);
    attribute.needsUpdate = true;
    this.isolate.value = mode === "all" ? 0 : 1;
    this.points.visible = this.background.visible = mode === "all";
    if (this.regionLines) this.regionLines.visible = mode === "all";
    return edges;
  }

  async loadRegion(index: number, manifest: Manifest) {
    this.regionAbort?.abort();
    const generation = ++this.regionGeneration;
    if (index < 0) {
      this.removeRegion();
      this.regionIndex = -1;
      return;
    }
    if (index === this.regionIndex) return;
    this.regionAbort = new AbortController();
    const data = await geometry(
      manifest.regions[index].geometry,
      this.regionAbort.signal,
    );
    if (generation !== this.regionGeneration) return;
    this.removeRegion();
    this.regionLines = new LineSegments(
      buffers(data),
      new LineBasicNodeMaterial({
        color: 0xa8c9cb,
        transparent: true,
        opacity: 0.2,
        depthWrite: false,
      }),
    );
    this.brain.add(this.regionLines);
    this.regionBytes = data.bytes;
    this.bytes += data.bytes;
    this.regionIndex = index;
  }

  private removeRegion() {
    if (!this.regionLines) return;
    this.brain.remove(this.regionLines);
    this.regionLines.geometry.dispose();
    (this.regionLines.material as LineBasicNodeMaterial).dispose();
    this.regionLines = null;
    this.bytes -= this.regionBytes;
    this.regionBytes = 0;
  }

  resetCamera() {
    this.following = false;
    this.controls.target.copy(this.center);
    this.camera.position.set(0, 0.2, 17.5);
    this.controls.update();
  }

  focusRegion(index: number) {
    const targets = [
      [-2.4, 2.4, -0.5],
      [0, 2.4, -0.7],
      [0, -2.5, 0.8],
    ];
    this.controls.target.fromArray(targets[index]);
    this.camera.position
      .copy(this.controls.target)
      .add(this.position.set(0, 0, 7));
    this.controls.update();
  }

  zoom(multiplier: number) {
    this.position
      .copy(this.camera.position)
      .sub(this.controls.target)
      .multiplyScalar(multiplier);
    if (this.position.length() < 0.4 || this.position.length() > 35) return;
    this.camera.position.copy(this.controls.target).add(this.position);
    this.controls.update();
  }

  render(t: number, parameters: Parameters, delta: number) {
    this.time.value = t;
    if (this.following && this.result && this.circuit && !this.reducedMotion)
      this.follow(t, delta);
    this.controls.update();
    this.updateWorld(t, parameters);
    const width = this.width,
      height = this.height;
    const split = this.view === "split" ? Math.round(width * 0.46) : 0;
    this.renderer.info.reset();
    this.renderer.setScissorTest(true);
    if (this.view !== "brain") {
      const w = this.view === "split" ? split : width;
      this.worldCamera.aspect = w / height;
      this.worldCamera.updateProjectionMatrix();
      this.renderer.setViewport(0, 0, w, height);
      this.renderer.setScissor(0, 0, w, height);
      this.renderer.render(this.world, this.worldCamera);
    }
    if (this.view !== "world") {
      const aspect = (width - split) / height;
      if (aspect !== this.aspect) {
        this.camera.aspect = aspect;
        this.camera.updateProjectionMatrix();
        this.aspect = aspect;
      }
      this.renderer.setViewport(split, 0, width - split, height);
      this.renderer.setScissor(split, 0, width - split, height);
      this.renderer.render(this.brain, this.camera);
    }
    this.renderer.setScissorTest(false);
  }

  private follow(t: number, delta: number) {
    const result = this.result!,
      circuit = this.circuit!;
    const neuron = result.path.find(
      (index) =>
        t >= result.onsets[index] && t < result.onsets[index] + CONDUCTION,
    );
    if (neuron === undefined) return;
    const route = circuit.neurons[neuron].path;
    if (route.length < 2) return;
    const u =
      Math.max(0, Math.min(0.9999, (t - result.onsets[neuron]) / CONDUCTION)) *
      (route.length - 1);
    const index = Math.floor(u);
    this.target.fromArray(route[index]);
    this.next.fromArray(route[index + 1]);
    this.target.lerp(this.next, u - index);
    this.position.copy(this.target).add(this.next.set(0.5, 0.35, 2.2));
    const alpha = 1 - Math.exp(-6 * delta);
    this.controls.target.lerp(this.target, alpha);
    this.camera.position.lerp(this.position, alpha);
  }

  private pick(event: PointerEvent) {
    if (!this.circuitLines || !this.circuitData) return;
    const bounds = this.orbit.getBoundingClientRect();
    this.mouse.set(
      ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
      (-(event.clientY - bounds.top) / bounds.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const hit = this.raycaster.intersectObject(this.circuitLines)[0];
    if (hit?.index !== undefined)
      this.onSelect(this.circuitData.owners[hit.index]);
  }

  private resize() {
    this.width = Math.max(1, this.host.clientWidth);
    this.height = Math.max(1, this.host.clientHeight);
    this.renderer.setSize(this.width, this.height);
    this.aspect = 0;
  }

  private makeWorld() {
    this.worldCamera.position.set(4.3, 3.5, -6.5);
    this.worldCamera.lookAt(0, 0.45, 0);
    this.world.add(new AmbientLight(0xb7cdd6, 1.4));
    const light = new DirectionalLight(0xffe6bf, 3);
    light.position.set(3, 5, -2);
    this.world.add(light);
    const ground = new Mesh(
      new CylinderGeometry(3.6, 3.6, 0.035, 96),
      new MeshStandardNodeMaterial({ color: 0x151e22, roughness: 0.85 }),
    );
    ground.position.y = -0.06;
    this.world.add(ground);
    for (const radius of [1.6, 2.6, 3.5]) {
      const ring = new Mesh(
        new RingGeometry(radius, radius + 0.008, 96),
        new MeshBasicNodeMaterial({ color: 0x34464a, side: DoubleSide }),
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = -0.03;
      this.world.add(ring);
    }
    const shell = new MeshStandardNodeMaterial({
      color: 0x614b35,
      roughness: 0.62,
      metalness: 0.12,
    });
    const dark = new MeshStandardNodeMaterial({
      color: 0x24272a,
      roughness: 0.45,
    });
    const ellipsoid = (
      material: MeshStandardNodeMaterial,
      x: number,
      y: number,
      z: number,
      sx: number,
      sy: number,
      sz: number,
    ) => {
      const mesh = new Mesh(new SphereGeometry(1, 24, 16), material);
      mesh.position.set(x, y, z);
      mesh.scale.set(sx, sy, sz);
      this.fly.add(mesh);
      return mesh;
    };
    ellipsoid(shell, 0, 0.47, 0.43, 0.28, 0.27, 0.55);
    ellipsoid(dark, 0, 0.53, -0.04, 0.26, 0.28, 0.31);
    ellipsoid(shell, 0, 0.55, -0.43, 0.23, 0.21, 0.2);
    const eye = new MeshStandardNodeMaterial({
      color: 0x8a352c,
      roughness: 0.45,
    });
    ellipsoid(eye, -0.16, 0.59, -0.49, 0.115, 0.155, 0.13);
    ellipsoid(eye, 0.16, 0.59, -0.49, 0.115, 0.155, 0.13);
    const wingMaterial = new MeshStandardNodeMaterial({
      color: 0xbacdd0,
      transparent: true,
      opacity: 0.3,
      roughness: 0.22,
      metalness: 0.2,
      depthWrite: false,
      side: DoubleSide,
    });
    for (const side of [-1, 1]) {
      const wing = ellipsoid(
        wingMaterial,
        side * 0.42,
        0.74,
        0.31,
        0.26,
        0.012,
        0.72,
      );
      wing.rotation.y = side * -0.48;
      this.wings.push(wing);
    }
    const limbs: [number[], number[]][] = [];
    for (const side of [-1, 1]) {
      for (let leg = 0; leg < 3; leg++) {
        const z = -0.3 + leg * 0.28;
        const a = [side * 0.13, 0.46, z],
          b = [side * 0.46, 0.27, z - 0.12];
        const c = [side * 0.61, 0.11, z + (leg - 1) * 0.23],
          d = [side * 0.8, 0.02, z + (leg - 1) * 0.32];
        limbs.push([a, b], [b, c], [c, d]);
      }
      limbs.push([
        [side * 0.07, 0.66, -0.57],
        [side * 0.17, 0.78, -0.76],
      ]);
    }
    const legs = new InstancedMesh(
      new CylinderGeometry(0.012, 0.009, 1, 6),
      dark,
      limbs.length,
    );
    for (let i = 0; i < limbs.length; i++) {
      this.position.fromArray(limbs[i][0]);
      this.next.fromArray(limbs[i][1]);
      this.target.subVectors(this.next, this.position);
      this.scale.set(1, this.target.length(), 1);
      this.rotation.setFromUnitVectors(this.up, this.target.normalize());
      this.position.lerp(this.next, 0.5);
      this.matrix.compose(this.position, this.rotation, this.scale);
      legs.setMatrixAt(i, this.matrix);
    }
    this.fly.add(legs);
    this.world.add(this.fly);
  }

  private updateWorld(t: number, p: Parameters) {
    const active = t >= 0;
    const at = this.result?.stimulusAt ?? 1.3;
    const approach = active ? Math.min(1, t / Math.max(0.1, at)) : 0;
    const angle = (p.direction * Math.PI) / 180;
    this.threat.visible = p.experiment !== "light";
    this.lightMarker.visible = p.experiment === "light";
    this.threat.scale.setScalar(p.size * (0.6 + approach * 0.7));
    const distance = 3.4 - approach * 1.9;
    this.threat.position.set(
      Math.sin(angle) * distance,
      0.65,
      -Math.cos(angle) * distance,
    );
    if (p.experiment === "motion")
      this.threat.position.set(
        (approach * 2 - 1) * 2.8 * (p.direction < 0 ? -1 : 1),
        0.6,
        -1.4,
      );
    this.lightMarker.position.set(
      Math.sin(angle) * 2,
      1.6,
      -Math.cos(angle) * 2,
    );
    this.stimulusLight.position.copy(this.lightMarker.position);
    this.stimulusLight.intensity =
      p.experiment === "light" ? p.intensity * 8 : 0;
    const motorAt = this.result?.motorAt ?? Infinity;
    const elapsed = Math.max(0, t - motorAt);
    const flight = Math.min(1, elapsed / 1.2);
    this.fly.position.set(
      -Math.sin(angle) * flight * 0.8,
      Math.sin(flight * Math.PI * 0.65) * 1.4,
      flight * 1.7,
    );
    this.fly.rotation.x = -flight * 0.2;
    for (let i = 0; i < this.wings.length; i++) {
      this.wings[i].rotation.z =
        elapsed > 0 ? Math.sin(elapsed * 65) * 0.55 * (i ? 1 : -1) : 0;
    }
  }
}
