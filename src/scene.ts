import {
  ACESFilmicToneMapping,
  CanvasTexture,
  CircleGeometry,
  Color,
  CylinderGeometry,
  HemisphereLight,
  BufferAttribute,
  BufferGeometry,
  DirectionalLight,
  DoubleSide,
  Group,
  InstancedBufferAttribute,
  LineBasicNodeMaterial,
  LineSegments,
  Object3D,
  PCFShadowMap,
  Plane,
  PlaneGeometry,
  PMREMGenerator,
  Mesh,
  MeshBasicNodeMaterial,
  MeshPhysicalNodeMaterial,
  MeshStandardNodeMaterial,
  MeshStandardMaterial,
  PerspectiveCamera,
  Points,
  PointsNodeMaterial,
  Quaternion,
  Raycaster,
  Sprite,
  SpriteNodeMaterial,
  Scene,
  SphereGeometry,
  SpotLight,
  Vector2,
  Vector3,
  WebGPURenderer,
} from "three/webgpu";
import {
  abs,
  attribute,
  float,
  instancedBufferAttribute,
  mix,
  mx_noise_float,
  normalView,
  positionLocal,
  positionViewDirection,
  positionWorld,
  uniform,
  uv,
  vec2,
  vec3,
} from "three/tsl";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { MeshSurfaceSampler } from "three/addons/math/MeshSurfaceSampler.js";
import {
  FEEDING_BODY_POSE,
  FOOD_RADII,
  feedingJointAngles,
  foodPosition,
} from "./feeding-geometry";
import {
  geometry,
  type Circuit,
  type GeometryData,
  type Manifest,
} from "./data";
import {
  CONDUCTION,
  escapePose,
  type Parameters,
  type Result,
} from "./simulation";
import type { BodyState } from "./body";

export type Quality = "ultra" | "high" | "medium" | "low";
type FlyJoint = {
  part: Object3D;
  rest: Quaternion;
  kind: string;
  side: number;
  pitch: Vector3;
  yaw: Vector3;
};
type MiddleLegRig = {
  femur: FlyJoint;
  tibia: FlyJoint;
  reaches: Float64Array;
};
const LEG_ANGLE_STEP = 0.35 / 64;
const TIBIA_ANGLE_RATIO = 0.3;
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
  private somaGlow: Sprite | null = null;
  private circuitBytes = 0;
  private circuitGeneration = 0;
  private circuitAbort: AbortController | null = null;
  private regionLines: LineSegments | null = null;
  private regionBytes = 0;
  private regionAbort: AbortController | null = null;
  private regionGeneration = 0;
  private regionIndex = -1;
  private fly = new Group();
  private finger = new Group();
  private fingerProjection = new Vector3();
  private fingerScreenPosition = { x: 0, y: 0 };
  private fingerDragPlane = new Plane(new Vector3(0, 1, 0), -0.95);
  private fingerDragPoint = new Vector3();
  private fingerDragPosition: [number, number, number] = [0, 0.95, 0];
  private fingerFraming = false;
  private beforeFingerPosition = new Vector3();
  private beforeFingerRotation = new Quaternion();
  private beforeFingerZoom = 0.96;
  private joints: FlyJoint[] = [];
  private middleLegs: MiddleLegRig[] = [];
  private feet: { part: Object3D; point: Vector3 }[] = [];
  private standingPitch = 0;
  private standingHeight = 0;
  private contact: Mesh;
  private worldZoom = 0.8;
  private activeOnsets: InstancedBufferAttribute | null = null;
  private threat: Mesh;
  private threatShadow: Mesh;
  private flowScreens = new Group();
  private flowTime = uniform(0);
  private flowContrast = uniform(0.85);
  private stimulusLight = new SpotLight(0xffe4a4, 0, 10, 0.45, 0.65, 1.5);
  private selector: Mesh;
  private raycaster = new Raycaster();
  private mouse = new Vector2();
  private target = new Vector3();
  private position = new Vector3();
  private next = new Vector3();
  private brainViewport = { x: 0, y: 0, width: 1, height: 1 };
  private height = 0;
  private width = 0;
  private center = new Vector3(0, 0.2, 0);
  private observer: ResizeObserver;
  private spikeTimes: BufferAttribute | null = null;
  private neuronToPoint: Int32Array | null = null;
  private feedingScene = new Group();
  private foodDrop: Mesh | null = null;
  private foodColor = new Color();
  private bitterColor = new Color(0x9974b8);
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
    this.renderer.toneMapping = ACESFilmicToneMapping;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = PCFShadowMap;
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
      color: 0x89b7bc,
      transparent: true,
      opacity: 0.6,
      depthWrite: false,
    });
    this.points = new Points(pointGeometry, pointMaterial);
    this.brain.add(this.points);
    const backgroundMaterial = new LineBasicNodeMaterial({
      color: 0x60898e,
      transparent: true,
      opacity: 0.22,
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
      new CircleGeometry(0.65, 64),
      new MeshBasicNodeMaterial({
        color: 0x050809,
        side: DoubleSide,
        transparent: true,
      }),
    );
    this.threat.rotation.x = -Math.PI / 2;
    this.stimulusLight.target.position.set(0, 0.5, 0);
    this.world.add(this.threat, this.stimulusLight, this.stimulusLight.target);
    this.contact = this.makeWorld();
    this.finger.visible = false;
    this.world.add(this.finger);
    this.threatShadow = new Mesh(
      new PlaneGeometry(3, 3),
      new MeshBasicNodeMaterial({
        map: (this.contact.material as MeshBasicNodeMaterial).map,
        transparent: true,
        opacity: 0.45,
        depthWrite: false,
      }),
    );
    this.threatShadow.rotation.x = -Math.PI / 2;
    this.threatShadow.position.y = -0.012;
    this.world.add(this.threatShadow);
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(host);
    this.observer.observe(orbit);
    this.raycaster.params.Line = { threshold: 0.07 };
  }

  async init() {
    await this.renderer.init();
    const room = new RoomEnvironment();
    const environment = new PMREMGenerator(this.renderer);
    this.world.environment = environment.fromScene(room, 0.06, 0.1, 100, {
      size: 128,
    }).texture;
    this.world.environmentIntensity = 0.7;
    room.dispose();
    environment.dispose();
    await Promise.all([this.loadFly(), this.loadFinger()]);
    this.setQuality("high");
    this.resize();
    return this;
  }

  get backend() {
    return "isWebGPUBackend" in this.renderer.backend ? "WebGPU" : "WebGL2";
  }

  setQuality(quality: Quality) {
    this.quality = quality;
    const [density, pixelRatio] = qualitySettings[quality];
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, pixelRatio));
    this.points.geometry.setDrawRange(
      0,
      Math.floor(
        this.points.geometry.attributes.position.count *
          (this.spikeTimes ? 1 : density),
      ),
    );
    this.background.geometry.setDrawRange(
      0,
      Math.floor(
        (this.background.geometry.attributes.position.count * density) / 2,
      ) * 2,
    );
    this.renderer.shadowMap.enabled = quality !== "low";
    this.resize();
  }

  clearCircuit() {
    this.circuitGeneration++;
    this.circuitAbort?.abort();
    if (this.circuitLines) {
      this.brain.remove(this.circuitLines);
      this.circuitLines.geometry.dispose();
      (this.circuitLines.material as LineBasicNodeMaterial).dispose();
    }
    if (this.somaGlow) {
      this.brain.remove(this.somaGlow);
      this.somaGlow.material.dispose();
    }
    this.circuitLines = this.somaGlow = null;
    this.activeOnsets = null;
    this.circuit = this.circuitData = this.result = null;
    this.bytes -= this.circuitBytes;
    this.circuitBytes = 0;
    this.selected.value = -1;
    this.isolate.value = 0;
    this.selector.visible = false;
    this.following = false;
    this.points.visible = this.background.visible = true;
    (this.points.material as PointsNodeMaterial).opacity = 0.6;
    (this.background.material as LineBasicNodeMaterial).opacity = 0.22;
    this.regionAbort?.abort();
    this.regionGeneration++;
    this.removeRegion();
    this.regionIndex = -1;
  }

  async loadCircuit(circuit: Circuit) {
    this.clearCircuit();
    const generation = this.circuitGeneration;
    this.circuitAbort = new AbortController();
    const data = await geometry(circuit.geometry, this.circuitAbort.signal);
    if (generation !== this.circuitGeneration) return;
    if (data.owners.some((owner) => owner >= circuit.neurons.length))
      throw new Error("Invalid circuit geometry owner");
    this.circuit = circuit;
    this.circuitData = data;
    (this.points.material as PointsNodeMaterial).opacity = 0.007;
    (this.background.material as LineBasicNodeMaterial).opacity = 0.025;
    const shape = buffers(data);
    shape.setAttribute(
      "onset",
      new BufferAttribute(new Float32Array(data.count).fill(1e6), 1),
    );
    shape.setAttribute(
      "emphasis",
      new BufferAttribute(new Float32Array(data.count).fill(1), 1),
    );
    shape.setAttribute(
      "polarity",
      new BufferAttribute(
        Float32Array.from(
          data.owners,
          (owner) => circuit.neurons[owner].polarity,
        ),
        1,
      ),
    );
    const progress = attribute<"float">("progress", "float");
    const distance = abs(
      this.time
        .sub(attribute<"float">("onset", "float"))
        .div(CONDUCTION)
        .sub(progress),
    );
    const pulse = float(1)
      .sub(distance.div(0.24))
      .clamp(0, 1)
      .mul(float(progress.greaterThanEqual(0)))
      .mul(
        float(this.time.greaterThanEqual(attribute<"float">("onset", "float"))),
      );
    const chosen = float(
      attribute<"float">("owner", "float").equal(this.selected),
    );
    const emphasis = attribute<"float">("emphasis", "float");
    const material = new LineBasicNodeMaterial({
      transparent: true,
      depthWrite: false,
      toneMapped: false,
    });
    material.colorNode = mix(
      vec3(0.12, 0.25, 0.28),
      mix(
        vec3(0.12, 0.4, 1),
        vec3(1, 0.73, 0.36),
        float(attribute<"float">("polarity", "float").greaterThan(0)),
      ),
      pulse.max(chosen.mul(0.7)),
    );
    material.opacityNode = float(0.045)
      .add(pulse.mul(0.82))
      .add(chosen.mul(0.35))
      .mul(mix(float(1), emphasis, this.isolate));
    this.circuitLines = new LineSegments(shape, material);
    this.brain.add(this.circuitLines);
    // One instanced draw highlights the actual circuit somas, with the same clock as the axon pulse.
    const positions = new InstancedBufferAttribute(
      Float32Array.from(circuit.neurons.flatMap((neuron) => neuron.position)),
      3,
    );
    this.activeOnsets = new InstancedBufferAttribute(
      new Float32Array(circuit.neurons.length).fill(1e6),
      1,
    );
    const activity = float(1)
      .sub(
        abs(
          this.time
            .sub(instancedBufferAttribute(this.activeOnsets, "float"))
            .sub(CONDUCTION / 2),
        ).div(CONDUCTION / 2),
      )
      .clamp(0, 1);
    const glow = new SpriteNodeMaterial({
      color: 0xffc478,
      transparent: true,
      depthWrite: false,
      toneMapped: false,
    });
    glow.positionNode = instancedBufferAttribute(positions, "vec3");
    const polarities = new InstancedBufferAttribute(
      Float32Array.from(circuit.neurons, (n) => n.polarity),
      1,
    );
    glow.colorNode = mix(
      vec3(0.12, 0.4, 1),
      vec3(1, 0.73, 0.36),
      float(
        instancedBufferAttribute<"float">(polarities, "float").greaterThan(0),
      ),
    );
    glow.scaleNode = vec2(0.85);
    glow.opacityNode = float(1)
      .sub(uv().sub(0.5).length().mul(2))
      .clamp(0, 1)
      .pow(2)
      .mul(activity);
    const somas = new Sprite(glow);
    somas.count = circuit.neurons.length;
    somas.frustumCulled = false;
    this.brain.add(somas);
    this.somaGlow = somas;
    this.circuitBytes =
      data.bytes + data.count * 12 + circuit.neurons.length * 20;
    this.bytes += this.circuitBytes;
  }

  setResult(result: Result | null) {
    this.result = result;
    if (this.activeOnsets) {
      for (let i = 0; i < this.activeOnsets.count; i++)
        this.activeOnsets.setX(
          i,
          Number.isFinite(result?.onsets[i]) ? result!.onsets[i] : 1e6,
        );
      this.activeOnsets.needsUpdate = true;
    }
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
    if (this.somaGlow) this.somaGlow.visible = mode === "all";
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
    this.camera.position.set(0, 0.2, this.spikeTimes ? 11 : 17.5);
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
    if (
      this.position.length() < this.controls.minDistance ||
      this.position.length() > this.controls.maxDistance
    )
      return;
    this.camera.position.copy(this.controls.target).add(this.position);
    this.controls.update();
  }

  render(t: number, parameters: Parameters, delta: number) {
    this.time.value = t;
    if (this.following && this.result && this.circuit && !this.reducedMotion)
      this.follow(t, delta);
    this.controls.update();
    this.updateWorld(t, parameters);
    this.draw();
  }

  private draw() {
    this.renderer.info.reset();
    this.renderer.setScissorTest(true);
    this.renderer.setClearColor(0x090e11, 0);
    this.renderer.setViewport(0, 0, this.width, this.height);
    this.renderer.setScissor(0, 0, this.width, this.height);
    this.renderer.render(this.world, this.worldCamera);
    // One renderer: the inset shares the simulation clock and clears only its rectangle.
    const { x, y, width, height } = this.brainViewport;
    this.renderer.setClearColor(0x0b1215, 1);
    this.renderer.setViewport(x, y, width, height);
    this.renderer.setScissor(x, y, width, height);
    this.renderer.render(this.brain, this.camera);
    this.renderer.setScissorTest(false);
  }

  enableFeeding(owners: Uint32Array, neuronCount: number) {
    this.neuronToPoint = new Int32Array(neuronCount).fill(-1);
    owners.forEach((neuron, point) => {
      this.neuronToPoint![neuron] = point;
    });
    this.spikeTimes = new BufferAttribute(
      new Float32Array(owners.length).fill(-1000),
      1,
    );
    this.points.geometry.setAttribute("spikeTime", this.spikeTimes);
    this.points.geometry.setDrawRange(0, owners.length);
    this.controls.maxDistance = 160;
    this.camera.far = 300;
    this.resetCamera();
    const material = this.points.material as PointsNodeMaterial;
    const pulse = this.time
      .sub(attribute("spikeTime", "float"))
      .max(0)
      .mul(-7)
      .exp();
    material.colorNode = mix(vec3(0.3, 0.52, 0.55), vec3(1, 0.69, 0.28), pulse);
    material.opacityNode = float(0.055).add(pulse.mul(0.94));
    material.needsUpdate = true;
    this.bytes += this.spikeTimes.array.byteLength;
    this.threat.visible =
      this.threatShadow.visible =
      this.flowScreens.visible =
        false;
    this.stimulusLight.intensity = 0;
    this.worldZoom = 0.96;
    this.worldCamera.position.set(4.6, 1.65, -2.4);
    this.worldCamera.lookAt(0, 0.62, -0.35);

    const platform = new Mesh(
      new CylinderGeometry(2.1, 2.15, 0.13, 96),
      new MeshStandardNodeMaterial({ color: 0x263c41, roughness: 0.57 }),
    );
    platform.position.y = -0.065;
    platform.receiveShadow = true;
    this.world.add(platform);
    const capillary = new Mesh(
      new CylinderGeometry(0.047, 0.047, 1.15, 24, 1, true),
      new MeshPhysicalNodeMaterial({
        color: 0xb2d4d8,
        roughness: 0.3,
        transparent: true,
        opacity: 0.18,
        side: DoubleSide,
        depthWrite: false,
        forceSinglePass: true,
      }),
    );
    capillary.rotation.x = Math.PI / 2;
    capillary.position.set(0, foodPosition(0)[1], -0.72);
    const foodMaterial = new MeshPhysicalNodeMaterial({
      color: 0xc88532,
      roughness: 0.28,
      metalness: 0,
      transparent: true,
      depthWrite: false,
      specularIntensity: 0.25,
      clearcoat: 0.1,
      clearcoatRoughness: 0.3,
    });
    // Keep the contact volume legible at its edge while showing the actual
    // articulated mouth through its center, without refractive distortion.
    foodMaterial.opacityNode = normalView
      .dot(positionViewDirection)
      .abs()
      .oneMinus()
      .pow(2)
      .mul(0.36)
      .add(0.1);
    this.foodDrop = new Mesh(
      new SphereGeometry(FOOD_RADII[0], 40, 28),
      foodMaterial,
    );
    this.foodDrop.scale.set(
      1,
      FOOD_RADII[1] / FOOD_RADII[0],
      FOOD_RADII[2] / FOOD_RADII[0],
    );
    this.foodDrop.position.y = foodPosition(0)[1];
    this.foodDrop.castShadow = false;
    this.feedingScene.add(capillary, this.foodDrop);
    this.world.add(this.feedingScene);
    this.resize();
  }

  recordSpikes(indices: Uint32Array, timeMs: number) {
    if (!this.spikeTimes || !this.neuronToPoint) return;
    for (const index of indices) {
      const point = this.neuronToPoint[index];
      if (point >= 0) this.spikeTimes.setX(point, timeMs / 1000);
    }
    this.spikeTimes.needsUpdate = true;
  }

  clearSpikes() {
    this.spikeTimes?.array.fill(-1000);
    if (this.spikeTimes) this.spikeTimes.needsUpdate = true;
  }

  /** Position the visible distal tip; this stimulus does not move the fly. */
  setFinger(position: readonly [number, number, number] | null): void {
    if (position === null) {
      this.finger.visible = false;
      if (this.fingerFraming) {
        this.worldCamera.position.copy(this.beforeFingerPosition);
        this.worldCamera.quaternion.copy(this.beforeFingerRotation);
        this.worldZoom = this.beforeFingerZoom;
        this.fingerFraming = false;
        this.resize();
      }
      return;
    }
    if (!position.every(Number.isFinite))
      throw new RangeError("Finger position must contain finite coordinates");
    this.finger.position.fromArray(position);
    this.finger.visible = true;
    if (!this.fingerFraming) {
      this.beforeFingerPosition.copy(this.worldCamera.position);
      this.beforeFingerRotation.copy(this.worldCamera.quaternion);
      this.beforeFingerZoom = this.worldZoom;
      this.fingerFraming = true;
      this.worldZoom = 0.8;
      this.worldCamera.position.set(5.6, 2.05, -3.6);
      this.worldCamera.lookAt(0, 1.05, -1.1);
      this.resize();
    }
  }

  /** Reused result: stage-relative CSS pixels, independent of render resolution. */
  getFingerScreenPosition(): { x: number; y: number } | null {
    if (!this.finger.visible) return null;
    this.worldCamera.updateMatrixWorld();
    this.fingerProjection.copy(this.finger.position).project(this.worldCamera);
    if (this.fingerProjection.z < -1 || this.fingerProjection.z > 1) return null;
    this.fingerScreenPosition.x = (this.fingerProjection.x + 1) * this.width / 2;
    this.fingerScreenPosition.y = (1 - this.fingerProjection.y) * this.height / 2;
    return this.fingerScreenPosition;
  }

  /** Reused result: intersect stage CSS pixels with the stimulus plane y=0.95. */
  fingerPointFromScreen(x: number, y: number): [number, number, number] | null {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    this.mouse.set(x / this.width * 2 - 1, 1 - y / this.height * 2);
    this.worldCamera.updateMatrixWorld();
    this.raycaster.setFromCamera(this.mouse, this.worldCamera);
    if (!this.raycaster.ray.intersectPlane(this.fingerDragPlane, this.fingerDragPoint))
      return null;
    this.fingerDragPosition[0] = this.fingerDragPoint.x;
    this.fingerDragPosition[1] = 0.95;
    this.fingerDragPosition[2] = this.fingerDragPoint.z;
    return this.fingerDragPosition;
  }

  renderFeeding(
    timeMs: number,
    extension: number,
    foodDistance: number,
    bitterness: number,
    body: BodyState,
  ) {
    this.time.value = timeMs / 1000;
    this.controls.update();
    this.fly.position.set(0, FEEDING_BODY_POSE.height + body.height, 0);
    this.fly.rotation.set(FEEDING_BODY_POSE.pitch, 0, 0);
    const angles = feedingJointAngles(extension);
    // Mouth and leg positions come from the worker's body state. Wings and
    // passive front/hind legs stay at rest; there is no timed escape pose here.
    for (const { part, rest, kind, pitch } of this.joints) {
      part.quaternion.copy(rest);
      if (kind === "mouth_rostrum") part.rotateOnAxis(pitch, angles.rostrum);
      if (kind === "mouth_tip") part.rotateOnAxis(pitch, angles.tip);
    }
    for (let side = 0; side < this.middleLegs.length; side++) {
      const { femur, tibia, reaches } = this.middleLegs[side];
      const reach = Math.max(0, Math.min(0.3, body.legExtension[side]));
      let low = 0, high = reaches.length - 1;
      while (high - low > 1) {
        const middle = (low + high) >>> 1;
        if (reaches[middle] < reach) low = middle;
        else high = middle;
      }
      const fraction = (reach - reaches[low]) / (reaches[high] - reaches[low]);
      const angle = LEG_ANGLE_STEP * (low + fraction);
      femur.part.rotateOnAxis(femur.pitch, angle);
      tibia.part.rotateOnAxis(tibia.pitch, angle * TIBIA_ANGLE_RATIO);
    }
    this.feedingScene.position.z = foodPosition(foodDistance)[2];
    if (this.foodDrop) {
      this.foodColor.setHex(0xc88532).lerp(this.bitterColor, bitterness);
      (this.foodDrop.material as MeshPhysicalNodeMaterial).color.copy(
        this.foodColor,
      );
    }
    this.draw();
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
    const stage = this.host.getBoundingClientRect();
    const orbit = this.orbit.getBoundingClientRect();
    this.brainViewport = {
      x: Math.round(orbit.left - stage.left),
      y: Math.round(orbit.top - stage.top),
      width: Math.max(1, Math.round(orbit.width)),
      height: Math.max(1, Math.round(orbit.height)),
    };
    this.camera.aspect = this.brainViewport.width / this.brainViewport.height;
    this.camera.updateProjectionMatrix();
    this.worldCamera.aspect = this.width / this.height;
    this.worldCamera.fov =
      (2 *
        Math.atan(
          Math.tan(Math.PI / 10) / Math.min(1, this.worldCamera.aspect),
        ) *
        180) /
      Math.PI;
    this.worldCamera.zoom = this.width > 680
      ? this.worldZoom
      : this.fingerFraming ? 0.53 : 0.86;
    // Shift the specimen away from the inset while preserving its perspective.
    this.worldCamera.setViewOffset(
      this.width,
      this.height,
      this.width > 680 ? this.width * 0.12 : this.width * 0.06,
      -this.height * (this.width > 680 ? 0.07 : 0.13),
      this.width,
      this.height,
    );
    this.worldCamera.updateProjectionMatrix();
  }

  private makeWorld() {
    this.worldCamera.position.set(3.8, 2.5, -4.7);
    this.worldCamera.lookAt(0, 0.6, -0.4);
    this.world.add(new HemisphereLight(0xd8e5ec, 0x252723, 0.8));
    const key = new DirectionalLight(0xffefd6, 2.8);
    key.position.set(-3, 6, -4);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    Object.assign(key.shadow.camera, {
      left: -5,
      right: 5,
      top: 5,
      bottom: -5,
      near: 0.1,
      far: 20,
    });
    key.shadow.bias = -0.0003;
    key.shadow.normalBias = 0.015;
    key.shadow.radius = 3;
    const rim = new DirectionalLight(0xd4e6eb, 1.8);
    rim.position.set(3, 3, 2);
    const fill = new DirectionalLight(0xe8e0cf, 1.1);
    fill.position.set(2, 1, -4);
    this.world.add(key, rim, fill);
    const floorMaterial = new MeshStandardNodeMaterial({
      color: 0x263236,
      roughness: 0.93,
      transparent: true,
      depthWrite: false,
    });
    floorMaterial.opacityNode = float(1)
      .sub(positionWorld.xz.length().div(7).clamp(0, 1).pow(2))
      .mul(0.55);
    const floor = new Mesh(new PlaneGeometry(20, 20), floorMaterial);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.025;
    floor.receiveShadow = true;
    this.world.add(floor);
    // A broad contact shadow remains available on the lowest graphics tier.
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 128;
    const context = canvas.getContext("2d")!;
    const gradient = context.createRadialGradient(64, 64, 4, 64, 64, 64);
    gradient.addColorStop(0, "rgba(0,0,0,0.65)");
    gradient.addColorStop(0.45, "rgba(0,0,0,0.3)");
    gradient.addColorStop(1, "rgba(0,0,0,0)");
    context.fillStyle = gradient;
    context.fillRect(0, 0, 128, 128);
    const contact = new Mesh(
      new PlaneGeometry(3, 3),
      new MeshBasicNodeMaterial({
        map: new CanvasTexture(canvas),
        transparent: true,
        depthWrite: false,
      }),
    );
    contact.rotation.x = -Math.PI / 2;
    contact.position.y = -0.015;
    this.world.add(contact, this.fly);
    const flowMaterial = new MeshBasicNodeMaterial({
      transparent: true,
      depthWrite: false,
      side: DoubleSide,
    });
    const stripe = float(
      uv().y.mul(12).add(this.flowTime).fract().greaterThan(0.5),
    );
    flowMaterial.colorNode = mix(
      vec3(0.04, 0.055, 0.06),
      vec3(0.55, 0.7, 0.7),
      stripe,
    );
    flowMaterial.opacityNode = float(1)
      .sub(uv().sub(0.5).length().mul(2))
      .clamp(0, 1)
      .pow(2)
      .mul(this.flowContrast)
      .mul(0.65);
    const projection = new Mesh(new PlaneGeometry(8, 8), flowMaterial);
    projection.rotation.x = -Math.PI / 2;
    projection.position.y = -0.02;
    this.flowScreens.add(projection);
    this.world.add(this.flowScreens);
    this.threat.castShadow = true;
    return contact;
  }

  private async loadFinger() {
    const gltf = await new GLTFLoader().loadAsync(
      `${import.meta.env.BASE_URL}finger/finger.glb`,
    ).catch((cause: unknown) => {
      throw new Error("Cannot load the finger stimulus asset (finger/finger.glb)", { cause });
    });
    const skin = new MeshPhysicalNodeMaterial({
      color: new Color().setRGB(0.54, 0.29, 0.2),
      roughness: 0.6,
      metalness: 0,
      specularIntensity: 0.3,
    });
    const originals = new Set<MeshStandardMaterial>();
    let meshes = 0;
    gltf.scene.traverse((part) => {
      if (!(part instanceof Mesh)) return;
      meshes++;
      const materials = Array.isArray(part.material) ? part.material : [part.material];
      materials.forEach((material) => originals.add(material as MeshStandardMaterial));
      part.material = skin;
      part.castShadow = true;
      part.receiveShadow = true;
      for (const name of Object.keys(part.geometry.attributes))
        this.bytes += part.geometry.getAttribute(name).array.byteLength;
      if (part.geometry.index) this.bytes += part.geometry.index.array.byteLength;
    });
    originals.forEach((material) => material.dispose());
    if (!meshes) {
      skin.dispose();
      throw new Error("Finger stimulus asset contains no mesh");
    }
    // CC0 authored human mesh, not a scan. Its distal surface vertex is the
    // origin, with the shaft along -Z; scale matches the shared stimulus size.
    this.finger.scale.setScalar(3.2);
    this.finger.add(gltf.scene);
  }

  private async loadFly() {
    const gltf = await new GLTFLoader().loadAsync(
      `${import.meta.env.BASE_URL}fly/fly.glb`,
    );
    const cuticle = new MeshPhysicalNodeMaterial({
      color: 0x8b7352,
      vertexColors: true,
      roughness: 0.58,
      metalness: 0,
      clearcoat: 0.12,
      clearcoatRoughness: 0.5,
    });
    cuticle.roughnessNode = mx_noise_float(positionLocal.mul(110))
      .mul(0.08)
      .add(0.6);
    // A constant material lift makes the small mouth parts readable in shadow.
    // Motor activity changes their pose only; this is not an activity indicator.
    const mouthCuticle = cuticle.clone();
    mouthCuticle.color.setHex(0xd5b38b);
    mouthCuticle.emissive.setHex(0x7d532c);
    mouthCuticle.emissiveIntensity = 0.12;
    const eyes = new MeshPhysicalNodeMaterial({
      color: 0x861b0d,
      roughness: 0.48,
      metalness: 0,
      clearcoat: 0.08,
      clearcoatRoughness: 0.45,
      bumpMap: this.eyeFacets(),
      bumpScale: 0.003,
    });
    const wings = new MeshPhysicalNodeMaterial({
      color: 0xb5c4ba,
      roughness: 0.36,
      metalness: 0,
      transparent: true,
      opacity: 0.17,
      depthWrite: false,
      side: DoubleSide,
      forceSinglePass: true,
      iridescence: 0.35,
      specularIntensity: 0.25,
      iridescenceIOR: 1.35,
      iridescenceThicknessRange: [180, 380],
    });
    this.fly.add(gltf.scene);
    const parts: Mesh[] = [];
    const originals = new Set<MeshStandardMaterial>();
    gltf.scene.traverse((part) => {
      if (part instanceof Mesh) {
        part.castShadow = !part.name.startsWith("wing");
        part.receiveShadow = !part.name.startsWith("wing");
        const material = part.material as MeshStandardMaterial;
        originals.add(material);
        if (material.name === "Cuticle") {
          part.material = part.name.startsWith("mouth_")
            ? mouthCuticle
            : cuticle;
          parts.push(part);
        } else if (material.name === "Compound eyes") {
          part.material = eyes;
        } else if (material.name === "Wing membrane") part.material = wings;
        for (const name of Object.keys(part.geometry.attributes))
          this.bytes += part.geometry.getAttribute(name).array.byteLength;
        if (part.geometry.index)
          this.bytes += part.geometry.index.array.byteLength;
        if (part.name.endsWith("tibia")) {
          const positions = part.geometry.getAttribute("position");
          let bottom = 0;
          for (let i = 1; i < positions.count; i++)
            if (positions.getY(i) < positions.getY(bottom)) bottom = i;
          this.feet.push({
            part,
            point: part.userData.footContact
              ? new Vector3().fromArray(part.userData.footContact)
              : new Vector3().fromBufferAttribute(positions, bottom),
          });
        }
      }
      if (/^(wing_|leg_|antenna_|mouth_|head$)/.test(part.name)) {
        this.joints.push({
          part,
          rest: part.quaternion.clone(),
          kind: part.name,
          side: part.name.includes("_L") ? -1 : 1,
          pitch: new Vector3().fromArray(
            part.userData.pitchAxis ?? part.userData.hingeAxis ?? [1, 0, 0],
          ),
          yaw: new Vector3().fromArray(part.userData.yawAxis ?? [0, 1, 0]),
        });
      }
    });
    originals.forEach((material) => material.dispose());
    parts.forEach((part) => this.addBristles(part));
    this.fly.updateMatrixWorld(true);
    const front = this.feet.find(({ part }) => part.name === "leg_LF_tibia");
    const rear = this.feet.find(({ part }) => part.name === "leg_LH_tibia");
    if (front && rear) {
      const a = front.point.clone().applyMatrix4(front.part.matrixWorld);
      const b = rear.point.clone().applyMatrix4(rear.part.matrixWorld);
      this.standingPitch = Math.atan((a.y - b.y) / (a.z - b.z));
      this.fly.rotation.x = this.standingPitch;
      this.fly.updateMatrixWorld(true);
      this.standingHeight = -Math.min(
        ...this.feet.map(
          ({ part, point }) =>
            this.next.copy(point).applyMatrix4(part.matrixWorld).y,
        ),
      );
    }
    this.calibrateMiddleLegs();
  }

  private calibrateMiddleLegs() {
    const position = this.fly.position.clone();
    const rotation = this.fly.quaternion.clone();
    this.fly.position.set(0, FEEDING_BODY_POSE.height, 0);
    this.fly.rotation.set(FEEDING_BODY_POSE.pitch, 0, 0);
    const point = new Vector3();
    // Sample the real mesh hierarchy once. Each table inverts downward foot
    // reach, so body height and foot contact do not require per-frame IK or
    // snapping. The tibia coupling is an illustrative actuator assumption.
    for (const side of ["LM", "RM"]) {
      const femur = this.joints.find(({ kind }) => kind === `leg_${side}_femur`);
      const tibia = this.joints.find(({ kind }) => kind === `leg_${side}_tibia`);
      const foot = this.feet.find(({ part }) => part.name === `leg_${side}_tibia`);
      if (!femur || !tibia || !foot)
        throw new Error(`Missing anatomical middle-leg rig: ${side}`);
      this.fly.updateMatrixWorld(true);
      const neutralY = point.copy(foot.point).applyMatrix4(foot.part.matrixWorld).y;
      const reaches = new Float64Array(65);
      for (let sample = 0; sample < reaches.length; sample++) {
        const angle = sample * LEG_ANGLE_STEP;
        femur.part.quaternion.copy(femur.rest);
        tibia.part.quaternion.copy(tibia.rest);
        femur.part.rotateOnAxis(femur.pitch, angle);
        tibia.part.rotateOnAxis(tibia.pitch, angle * TIBIA_ANGLE_RATIO);
        this.fly.updateMatrixWorld(true);
        reaches[sample] = neutralY - point.copy(foot.point).applyMatrix4(foot.part.matrixWorld).y;
        if (sample && reaches[sample] <= reaches[sample - 1])
          throw new Error(`Non-monotonic middle-leg calibration: ${side}`);
      }
      if (reaches[reaches.length - 1] < 0.3)
        throw new Error(`Insufficient anatomical middle-leg reach: ${side}`);
      femur.part.quaternion.copy(femur.rest);
      tibia.part.quaternion.copy(tibia.rest);
      this.middleLegs.push({ femur, tibia, reaches });
    }
    this.fly.position.copy(position);
    this.fly.quaternion.copy(rotation);
    this.fly.updateMatrixWorld(true);
  }

  private eyeFacets() {
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 256;
    const context = canvas.getContext("2d")!;
    context.fillStyle = "#252525";
    context.fillRect(0, 0, 512, 256);
    for (let row = -1; row < 20; row++) {
      for (let column = -1; column < 34; column++) {
        const x = column * 16 + (row % 2) * 8,
          y = row * 14;
        const facet = context.createRadialGradient(x - 1, y - 1, 1, x, y, 7.8);
        facet.addColorStop(0, "#d2d2d2");
        facet.addColorStop(0.65, "#999999");
        facet.addColorStop(1, "#353535");
        context.fillStyle = facet;
        context.beginPath();
        context.arc(x, y, 7.8, 0, Math.PI * 2);
        context.fill();
      }
    }
    return new CanvasTexture(canvas);
  }

  private addBristles(part: Mesh) {
    // Illustrative surface detail, sampled on the anatomical mesh, not measured setal positions.
    if (/^(antenna|hinge_|mouth_)/.test(part.name)) return;
    const leg = part.name.startsWith("leg_");
    const count = leg ? 40 : part.name === "body" ? 750 : 120;
    let seed =
      314159 + [...part.name].reduce((sum, c) => sum + c.charCodeAt(0), 0);
    const random = () =>
      (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296;
    // The installed renderer exposes this method; its older type package omits it.
    const sampler = new MeshSurfaceSampler(part) as MeshSurfaceSampler & {
      setRandomGenerator(generator: () => number): MeshSurfaceSampler;
    };
    sampler.setRandomGenerator(random).build();
    const p = new Vector3(),
      normal = new Vector3();
    const vertices: number[] = [];
    for (let i = 0; i < count; i++) {
      sampler.sample(p, normal);
      const length =
        (leg ? 0.012 : 0.016) + random() ** 4 * (leg ? 0.025 : 0.07);
      vertices.push(p.x, p.y, p.z);
      p.addScaledVector(normal, length);
      vertices.push(p.x, p.y, p.z);
    }
    const data = new Float32Array(vertices);
    const bristles = new LineSegments(
      new BufferGeometry().setAttribute(
        "position",
        new BufferAttribute(data, 3),
      ),
      new LineBasicNodeMaterial({
        color: 0x201c13,
        transparent: true,
        opacity: 0.8,
      }),
    );
    bristles.name = "illustrative-setae";
    part.add(bristles);
    this.bytes += data.byteLength;
  }

  private updateWorld(t: number, p: Parameters) {
    const at = this.result?.stimulusAt ?? 1.3;
    const approach = Math.max(0, Math.min(1, t / Math.max(0.1, at)));
    const angle = (p.direction * Math.PI) / 180;
    const distance = 2.6 * (1 - approach);
    this.threat.visible =
      (p.experiment === "threat" || p.experiment === "lesion") &&
      p.intensity > 0;
    (this.threat.material as MeshBasicNodeMaterial).opacity = p.intensity;
    this.flowScreens.visible = p.experiment === "motion";
    this.flowScreens.scale.set(p.size, 1, p.size);
    this.flowScreens.position.x = Math.sin(angle) * 2;
    this.flowTime.value = Math.max(0, t - at) * p.speed * 2;
    this.flowContrast.value = p.intensity;
    this.threat.scale.setScalar(p.size * (0.3 + approach));
    this.threat.position.set(
      Math.sin(angle) * distance - 1.8,
      3.5,
      -Math.cos(angle) * distance - 2.4,
    );
    this.threatShadow.visible = this.threat.visible;
    (this.threatShadow.material as MeshBasicNodeMaterial).opacity =
      p.intensity * 0.45;
    this.threatShadow.position.x = Math.sin(angle) * distance;
    this.threatShadow.position.z = -Math.cos(angle) * distance;
    this.threatShadow.scale.setScalar(p.size * (0.2 + approach));
    this.stimulusLight.position.set(
      Math.sin(angle) * 2.4,
      2.8,
      -Math.cos(angle) * 2.4,
    );
    this.stimulusLight.angle = 0.18 + p.size * 0.28;
    const lightOn = Math.max(0, Math.min(1, (t - at) / 0.35));
    this.stimulusLight.intensity =
      p.experiment === "light"
        ? lightOn * lightOn * (3 - 2 * lightOn) * p.intensity * 35
        : 0;
    const pose = escapePose(
      t,
      this.result?.action === "jump" ? this.result.motorAt : Infinity,
    );
    this.fly.position.set(
      -Math.sin(angle) * pose.travel * 0.7,
      this.standingHeight + pose.height,
      Math.cos(angle) * pose.travel * 0.7,
    );
    this.fly.rotation.x = this.standingPitch + pose.pitch;
    const turn =
      this.result?.action === "turn"
        ? Math.max(0, Math.min(1, (t - this.result.motorAt) / 1.3))
        : 0;
    this.fly.rotation.y = -angle * turn * turn * (3 - 2 * turn) * 0.4;
    this.contact.position.x = this.fly.position.x;
    this.contact.position.z = this.fly.position.z;
    this.contact.scale.setScalar(1 + pose.height * 0.4);
    (this.contact.material as MeshBasicNodeMaterial).opacity =
      1 - pose.height * 0.6;
    const idle = this.reducedMotion ? 0 : Math.max(0, t);
    for (const { part, rest, kind, side, pitch, yaw } of this.joints) {
      part.quaternion.copy(rest);
      if (kind.startsWith("wing")) {
        part.rotateOnAxis(pitch, pose.wing * 1.5);
        part.rotateOnAxis(yaw, pose.fold * 0.48);
      } else if (kind === "head") {
        // Resting movement is illustrative and independent of the stimulus response.
        part.rotateOnAxis(pitch, Math.sin(idle * 1.7) * 0.035);
        part.rotateOnAxis(yaw, Math.sin(idle * 1.1) * 0.025);
      } else if (kind.startsWith("antenna")) {
        part.rotateOnAxis(
          pitch,
          Math.sin(idle * 5.1 + side) * 0.1 * Math.min(1, idle),
        );
        part.rotateOnAxis(yaw, Math.sin(idle * 3.2) * 0.065);
      } else {
        const middle = kind.includes("M_");
        const tripod = /leg_(LF|RM|LH)_/.test(kind) ? 0 : Math.PI;
        const step = Math.sin(turn * Math.PI * 4 + tripod);
        const gait = Math.sin(turn * Math.PI) ** 2 * Math.abs(angle) * 0.12;
        if (kind.endsWith("coxa")) {
          part.rotateOnAxis(pitch, pose.fold * 0.18);
          part.rotateOnAxis(yaw, Math.cos(turn * Math.PI * 4 + tripod) * gait);
        } else if (kind.endsWith("femur")) {
          part.rotateOnAxis(
            pitch,
            pose.fold * 0.55 -
              pose.push * (middle ? 0.18 : 0.07) +
              pose.settle * 0.1 +
              step * gait,
          );
        } else {
          part.rotateOnAxis(
            pitch,
            -pose.fold * 0.8 +
              pose.push * (middle ? 0.22 : 0.09) -
              pose.settle * 0.14 -
              Math.max(0, step) * gait * 1.5,
          );
        }
      }
    }
    this.fly.updateMatrixWorld(true);
    if (this.feet.length && pose.airborne < 1) {
      const lowest = Math.min(
        ...this.feet.map(
          ({ part, point }) =>
            this.next.copy(point).applyMatrix4(part.matrixWorld).y,
        ),
      );
      this.fly.position.y -= lowest * (1 - pose.airborne);
    }
  }
}
