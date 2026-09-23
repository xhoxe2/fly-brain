import "./style.css";
import { geometry, json, type Circuit, type Manifest } from "./data";
import { LabScene, type Quality, type View } from "./scene";
import { monitor } from "./performance";
import {
  CONDUCTION,
  type Experiment,
  type Lesion,
  type Parameters,
  type Result,
} from "./simulation";

const $ = <T extends HTMLElement = HTMLElement>(selector: string) =>
  document.querySelector<T>(selector)!;
const icon = (path: string) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true">${path}</svg>`;
const playIcon = icon('<path d="m9 5 10 7-10 7Z"/>');
const experiments: Record<
  Experiment,
  { number: string; title: string; description: string; icon: string }
> = {
  threat: {
    number: "01",
    title: "Escape response",
    description:
      "An approaching object. A signal through the brain. A jump to safety.",
    icon: '<circle cx="12" cy="12" r="7"/><path d="m3 3 4 4m10 10 4 4m-4-14 4-4M3 21l4-4"/>',
  },
  motion: {
    number: "02",
    title: "A world in motion",
    description:
      "Move a visual stimulus across the field of view. Compare left and right activation.",
    icon: '<path d="M3 7h12m-4-4 4 4-4 4M21 17H9m4-4-4 4 4 4"/>',
  },
  light: {
    number: "03",
    title: "Into the light",
    description:
      "Change the light position and strength. Explore a modeled visual input.",
    icon: '<circle cx="12" cy="12" r="4"/><path d="M12 1v3m0 16v3M1 12h3m16 0h3M4 4l2 2m12 12 2 2M4 20l2-2M18 6l2-2"/>',
  },
  lesion: {
    number: "04",
    title: "Break the connection",
    description:
      "Disable a neuron group. Repeat the same threat and compare the motor output.",
    icon: '<path d="m4 4 16 16M5 14l-2 2a4 4 0 0 0 6 6l3-3m7-9 2-2a4 4 0 0 0-6-6l-3 3M9 15l6-6"/>',
  },
};

function shell() {
  $("#app").innerHTML = `
  <header class="topbar">
    <a class="wordmark" href="${import.meta.env.BASE_URL}" aria-label="Fly Brain home">FLY<span>//</span>BRAIN<span class="wordmark-dot">β</span></a>
    <span class="header-description">THE ANATOMY OF A DECISION</span>
    <div class="header-actions"><span class="live-dot"></span><span class="dataset-tag">MALE CNS / V1.0</span><button class="text-button" id="science">The science <span>↗</span></button></div>
  </header>
  <main class="workspace">
    <section class="visualization" aria-label="Interactive connectome">
      <div class="scene-top"><span class="micro-label"><span class="live-dot"></span> DROSOPHILA MELANOGASTER</span>
        <div class="view-switch" role="group" aria-label="View mode">${(["world", "brain", "split"] as const).map((view) => `<button data-view="${view}" aria-pressed="${view === "brain"}">${view}</button>`).join("")}</div>
      </div>
      <div id="stage"><div id="orbit" tabindex="0" aria-label="Rotate brain with arrow keys; plus and minus to zoom. Drag to orbit, scroll to zoom."></div></div>
      <div class="intro" id="intro"><div class="eyebrow">A WINDOW INTO THE NERVOUS SYSTEM</div><h1>Watch a<br><em>decision</em> form.</h1><p>166,000+ neurons.<br>One nervous system.</p><button class="enter-button" id="enter" disabled>ENTER THE BRAIN <span>↗</span></button></div>
      <div class="scene-captions"><span id="world-caption">BEHAVIOR <span class="dim">/ VIRTUAL ARENA</span></span><span id="brain-caption">ANATOMY <span class="dim">/ REAL CONNECTOME</span></span></div>
      <div class="anatomy-label label-optic">OPTIC LOBES <span></span></div><div class="anatomy-label label-vnc">VENTRAL NERVE CORD <span></span></div>
      <div class="scene-bottom"><div class="scene-legend"><span class="legend-dot"></span> STRUCTURE <span class="legend-dot active"></span> ACTIVITY</div><div class="camera-tools"><button id="zoom-out" aria-label="Zoom out">−</button><button id="zoom-in" aria-label="Zoom in">+</button><button id="reset-camera" aria-label="Reset camera">${icon('<path d="M4 9a8 8 0 1 1 0 7M4 3v6h6"/>')}</button></div></div>
      <div id="loading" role="status"><span class="loading-line"></span><span id="loading-text">Preparing real neuron geometry…</span></div>
      <div id="selection" class="selection" hidden><div class="selection-heading"><span class="micro-label">NEURON INSPECTOR</span><button id="close-selection" aria-label="Close neuron inspector">×</button></div><h3 id="neuron-name"></h3><p id="neuron-role"></p><dl id="neuron-data"></dl><div class="connection-buttons"><button data-connections="inputs">Show inputs</button><button data-connections="outputs">Show outputs</button><button data-connections="isolate">Isolate</button><button data-connections="all">Show all</button></div><p id="connection-result"></p><button class="text-button" id="follow-path">Follow path ↗</button></div>
    </section>
    <aside class="lab" aria-label="Experiment controls">
      <div class="lab-heading"><span class="micro-label">THE EXPERIMENT LAB</span><span class="lab-index" id="experiment-number">01 / 04</span></div>
      <nav class="experiment-tabs" aria-label="Experiments">${Object.entries(
        experiments,
      )
        .map(
          ([key, value]) =>
            `<button data-experiment="${key}" aria-pressed="${key === "threat"}">${icon(value.icon)}<span>${key}</span></button>`,
        )
        .join("")}</nav>
      <div class="experiment-copy"><div class="eyebrow" id="experiment-kind">EXPERIENCE 01</div><h2 id="experiment-title">Escape response</h2><p id="experiment-description">An approaching object. A signal through the brain. A jump to safety.</p></div>
      <fieldset class="parameters"><legend>STIMULUS</legend>
        <label class="slider-label" for="direction">Direction <output id="direction-value">0°</output></label><input id="direction" type="range" min="-90" max="90" value="0" step="5"><div class="range-ends"><span>LEFT</span><span>RIGHT</span></div>
        <label class="slider-label" for="speed">Approach speed <output id="speed-value">1.0×</output></label><input id="speed" type="range" min="0.25" max="3" value="1" step="0.25">
        <label class="slider-label" for="size">Object size <output id="size-value">1.0×</output></label><input id="size" type="range" min="0.2" max="2" value="1" step="0.1">
        <label class="slider-label" for="intensity">Intensity <output id="intensity-value">85%</output></label><input id="intensity" type="range" min="0" max="1" value="0.85" step="0.05">
      </fieldset>
      <div id="lesion-options" class="lesion-options" hidden><label for="lesion">DISABLE NEURONS</label><select id="lesion"><option value="descending">Descending · giant fibers</option><option value="visual">Visual · LC4 / LPLC2</option><option value="motor">Motor · TTMn</option><option value="none">None · control condition</option></select><div class="comparison" id="comparison" aria-live="polite">The same stimulus is compared with an intact circuit.</div><div id="comparison-playback" class="comparison-playback" hidden><button id="play-normal">Play normal</button><button id="play-disabled">Play disabled</button></div></div>
      <button class="run-button" id="run" disabled>${playIcon}<span>Run experiment</span><kbd>↵</kbd></button>
      <button class="follow-button" id="follow" aria-pressed="false" disabled>${icon('<path d="M3 8V3h5m8 0h5v5M3 16v5h5m8 0h5v-5"/><circle cx="12" cy="12" r="3"/>')} Follow signal <span>↗</span></button>
      <div class="pathway"><div class="micro-label">SIGNAL PATHWAY <span class="dim">/ MODEL</span></div><ol><li data-phase="visual"><i></i><div>Visual input<span>LC4 · LPLC2</span></div><b>01</b></li><li data-phase="descending"><i></i><div>Descending signal<span>Giant fiber · DNp01</span></div><b>02</b></li><li data-phase="motor"><i></i><div>Motor output<span>Jump motor neuron · TTMn</span></div><b>03</b></li></ol></div>
      <details class="explore"><summary>Explore the anatomy <span>+</span></summary><label for="region">REGION DETAIL</label><select id="region"><option value="-1">Full nervous system</option><option value="0">Optic lobes</option><option value="1">Central brain</option><option value="2">Ventral nerve cord</option></select><label for="neuron-select">SELECT A CIRCUIT NEURON</label><select id="neuron-select"><option value="-1">Load an experiment first</option></select><p id="region-status">Representative skeletons load on demand.</p></details>
      <div class="honesty"><span>ANATOMY <b>MEASURED</b></span><span>DYNAMICS <b>MODELED</b></span></div>
    </aside>
  </main>
  <section class="transport" aria-label="Simulation playback"><button id="play-pause" aria-label="Play simulation" disabled>${playIcon}</button><button id="restart" aria-label="Restart experiment" disabled>${icon('<path d="M5 5v14M19 5 7 12l12 7Z"/>')}</button><div class="time-readout"><strong id="elapsed">00.00</strong><span>MODEL SECONDS</span></div><div class="timeline"><div class="timeline-labels"><span id="run-status" role="status">READY TO EXPLORE</span><span id="duration">—</span></div><input type="range" id="scrub" min="0" max="6" step="0.01" value="0" disabled aria-label="Simulation time"></div><div class="rates" role="group" aria-label="Playback speed">${[1, 0.25, 0.05, 0.01].map((rate) => `<button data-rate="${rate}" aria-pressed="${rate === 1}">${rate}×</button>`).join("")}</div></section>
  <footer class="statusbar"><span class="source-credit">REAL DATA. OPEN SCIENCE. <a href="https://male-cns.janelia.org/" target="_blank" rel="noreferrer">FlyEM / Janelia ↗</a></span><span id="neuron-count">LOADING ANATOMY</span><div class="runtime"><button id="stats-toggle" aria-expanded="false"><span class="live-dot"></span><span id="fps">— FPS</span></button><span id="backend">INITIALIZING</span><label for="quality" class="sr-only">Graphics quality</label><select id="quality"><option value="auto">AUTO QUALITY</option><option value="ultra">ULTRA</option><option value="high">HIGH</option><option value="medium">MEDIUM</option><option value="low">LOW</option></select></div></footer>
  <pre id="stats" hidden></pre>
  <dialog id="science-dialog"><button id="close-science" class="dialog-close" aria-label="Close scientific notes">×</button><span class="eyebrow">WHAT YOU ARE LOOKING AT</span><h2>Real anatomy.<br>Illustrative dynamics.</h2><p>The geometry, neuron identities and chemical-synapse counts come from <a href="https://male-cns.janelia.org/download/" target="_blank" rel="noreferrer">Male CNS v1.0</a>. The overview contains 140,024 traced neurons with published soma coordinates; 25,098 traced neurons without these coordinates are omitted. Detailed skeletons are representative subsets.</p><p>The escape subgraph contains 36 neurons and 250 measured connections. Inputs and outputs in the inspector count partners across the full source graph. Highlighting is limited to 12 partners within this subgraph.</p><p>Activation delays, within-neuron pulse direction, stimulus sensitivity and fly movements are model assumptions. They do not reproduce a living fly. Electrical synapses, inhibition and membrane dynamics are not simulated. Connections with fewer than five chemical synapses are excluded from propagation.</p><p>Motion and light explore hypothetical inputs to this same escape subgraph; they are not independently validated motion or phototaxis circuits. A lesion result means loss of output in this selected model, not a prediction that a real fly cannot escape.</p><p>One scene unit represents 100 µm. Signal travel and camera routes follow prepared skeleton geometry; transitions between neurons are schematic, since synapse locations are not loaded. Playback time is model time, not measured biological latency.</p><p class="credit">Data: FlyEM / HHMI Janelia, University of Cambridge, MRC LMB and Google Research. <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noreferrer">CC BY 4.0</a>. Runs entirely in your browser.</p></dialog>`;
}

async function start() {
  if (new URLSearchParams(location.search).has("benchmark"))
    return (await import("./benchmark")).benchmark();
  shell();
  $("#science").onclick = () =>
    $("#science-dialog") instanceof HTMLDialogElement &&
    ($("#science-dialog") as HTMLDialogElement).showModal();
  $("#close-science").onclick = () =>
    ($("#science-dialog") as HTMLDialogElement).close();
  const p: Parameters = {
    experiment: "threat",
    direction: 0,
    speed: 1,
    size: 1,
    intensity: 0.85,
    lesion: "none",
  };
  let runParameters = { ...p };
  let manifest: Manifest;
  let scene: LabScene;
  try {
    manifest = await json<Manifest>("manifest.json");
    const [overview, background] = await Promise.all([
      geometry(manifest.overview),
      geometry(manifest.background),
    ]);
    scene = await new LabScene(
      $("#stage"),
      $("#orbit"),
      overview,
      background,
    ).init();
  } catch (error) {
    $("#loading-text").textContent =
      `The 3D scene could not start. ${error instanceof Error ? error.message : error}. Reload to retry.`;
    console.error(error);
    return;
  }
  $("#loading").hidden = true;
  $("#run").removeAttribute("disabled");
  $("#enter").removeAttribute("disabled");
  $("#backend").textContent = scene.backend.toUpperCase();
  $("#neuron-count").textContent =
    `${manifest.mappedSomas.toLocaleString("en-US")} MAPPED NEURONS`;
  const perf = monitor(scene.renderer);
  const worker = new Worker(
    new URL("./simulation.worker.ts", import.meta.url),
    { type: "module" },
  );
  let circuit: Circuit | null = null;
  let circuitLoading: Promise<void> | null = null;
  let result: Result | null = null;
  let baselineResult: Result | null = null,
    lesionResult: Result | null = null;
  let time = -10,
    playing = false,
    rate = 1,
    runId = 0,
    busy = false;
  let last = 0,
    lastUI = 0,
    automatic = true,
    calibrated = false,
    warmSamples = 0,
    slowSamples = 0;
  let entered = false,
    followEnabled = false,
    followReturned = false;
  let autoRegion = -1;
  const fpsHistory: number[] = [];
  const debug = {
    stats: perf.stats,
    get time() {
      return time;
    },
    get playing() {
      return playing;
    },
    get result() {
      return result;
    },
    get parameters() {
      return p;
    },
    scene,
  };
  Object.assign(window, { flyBrain: debug, flyBrainStats: perf.stats });

  async function ensureCircuit() {
    if (circuit) return;
    if (!circuitLoading)
      circuitLoading = (async () => {
        const loaded = await json<Circuit>(manifest.circuit);
        await scene.loadCircuit(loaded);
        circuit = loaded;
        const select = $<HTMLSelectElement>("#neuron-select");
        select.replaceChildren(
          new Option("Choose a neuron…", "-1"),
          ...loaded.neurons.map(
            (n, i) => new Option(`${n.name} · ${n.id}`, String(i)),
          ),
        );
      })().catch((error) => {
        circuitLoading = null;
        throw error;
      });
    return circuitLoading;
  }

  function setView(view: View) {
    scene.setView(view);
    $(".visualization").dataset.view = view;
    document
      .querySelectorAll<HTMLButtonElement>("[data-view]")
      .forEach((button) =>
        button.setAttribute(
          "aria-pressed",
          String(button.dataset.view === view),
        ),
      );
  }
  setView("brain");
  function enter() {
    entered = true;
    $("#app").classList.add("experiment-open");
    $("#intro").hidden = true;
    $(".visualization").classList.add("entered");
  }
  function setPlaying(value: boolean) {
    playing = value;
    $("#play-pause").innerHTML = value
      ? icon('<path d="M8 5v14M16 5v14"/>')
      : playIcon;
    $("#play-pause").setAttribute(
      "aria-label",
      value ? "Pause simulation" : "Play simulation",
    );
    $("#run-status").textContent = value
      ? "EXPERIMENT RUNNING"
      : result
        ? time >= result.duration
          ? "EXPERIMENT COMPLETE"
          : "PAUSED"
        : "READY TO EXPLORE";
  }
  function invalidate() {
    runId++;
    busy = false;
    result = baselineResult = lesionResult = null;
    $("#comparison-playback").hidden = true;
    time = -10;
    scene.setResult(null);
    setPlaying(false);
    scene.following = false;
    $("#run").removeAttribute("disabled");
    $("#run span").textContent = "Run experiment";
    for (const id of ["#play-pause", "#restart", "#scrub", "#follow"])
      $(id).setAttribute("disabled", "");
    $("#comparison").textContent =
      "The same stimulus is compared with an intact circuit.";
    $("#elapsed").textContent = "00.00";
    $<HTMLInputElement>("#scrub").value = "0";
    updatePhases();
  }
  async function run() {
    if (busy) return;
    enter();
    busy = true;
    const id = ++runId;
    setPlaying(false);
    $("#run").setAttribute("disabled", "");
    $("#run span").textContent = circuit
      ? "Preparing experiment…"
      : "Loading neural circuit…";
    try {
      await ensureCircuit();
      if (id !== runId) return;
      runParameters = { ...p };
      worker.postMessage({ id, circuit, parameters: runParameters });
    } catch (error) {
      if (id !== runId) return;
      busy = false;
      $("#run").removeAttribute("disabled");
      $("#run span").textContent = "Retry experiment";
      $("#run-status").textContent =
        error instanceof Error ? error.message : "Could not load the circuit";
    }
  }
  worker.onmessage = (
    event: MessageEvent<{
      id: number;
      result: Result;
      baseline: Result | null;
      error?: string;
    }>,
  ) => {
    if (event.data.id !== runId) return;
    busy = false;
    $("#run").removeAttribute("disabled");
    $("#run span").textContent = "Run again";
    if (event.data.error) {
      $("#run-status").textContent = event.data.error;
      return;
    }
    result = event.data.result;
    lesionResult = result;
    baselineResult = event.data.baseline;
    $("#comparison-playback").hidden = !baselineResult;
    scene.setResult(result);
    time = 0;
    followReturned = false;
    scene.following = followEnabled;
    setPlaying(true);
    for (const id of ["#play-pause", "#restart", "#scrub", "#follow"])
      $(id).removeAttribute("disabled");
    $<HTMLInputElement>("#scrub").max = String(result.duration);
    $("#duration").textContent = `${result.duration.toFixed(2)} s`;
    if (event.data.baseline) {
      const normal = event.data.baseline.motorAt;
      $("#comparison").textContent =
        `NORMAL: ${Number.isFinite(normal) ? `motor output at ${normal.toFixed(2)} s` : "no motor output"}. DISABLED: ${Number.isFinite(result.motorAt) ? `motor output at ${result.motorAt.toFixed(2)} s` : "signal blocked; no jump"}.`;
    }
  };
  worker.onerror = () => {
    busy = false;
    $("#run").removeAttribute("disabled");
    $("#run span").textContent = "Retry experiment";
    $("#run-status").textContent = "Experiment worker failed. Reload to retry.";
  };
  function playComparison(normal: boolean) {
    const selected = normal ? baselineResult : lesionResult;
    if (!selected) return;
    result = selected;
    scene.setResult(result);
    time = 0;
    followReturned = false;
    $<HTMLInputElement>("#scrub").max = String(result.duration);
    $("#duration").textContent = `${result.duration.toFixed(2)} s`;
    setView("split");
    setPlaying(true);
    $("#run-status").textContent = normal
      ? "NORMAL CIRCUIT"
      : "NEURONS DISABLED";
    $("#play-normal").setAttribute("aria-pressed", String(normal));
    $("#play-disabled").setAttribute("aria-pressed", String(!normal));
  }
  $("#play-normal").onclick = () => playComparison(true);
  $("#play-disabled").onclick = () => playComparison(false);
  document.addEventListener("keydown", (event) => {
    const tag = (event.target as HTMLElement).tagName;
    if (
      ["INPUT", "SELECT", "BUTTON", "TEXTAREA", "SUMMARY"].includes(tag) ||
      $<HTMLDialogElement>("#science-dialog").open
    )
      return;
    if (event.key === "Enter") {
      event.preventDefault();
      run();
    }
    if (event.code === "Space" && result) {
      event.preventDefault();
      $("#play-pause").click();
    }
  });
  $("#run").onclick = run;
  $("#enter").onclick = () => {
    setView("split");
    run();
  };
  $("#play-pause").onclick = () => {
    if (!result) return;
    if (time >= result.duration) {
      time = 0;
      followReturned = false;
    }
    setPlaying(!playing);
  };
  $("#restart").onclick = () => {
    if (result) {
      time = 0;
      followReturned = false;
      scene.following = followEnabled;
      setPlaying(true);
    }
  };
  $<HTMLInputElement>("#scrub").oninput = (e) => {
    if (result) {
      time = Number((e.target as HTMLInputElement).value);
      setPlaying(false);
    }
  };
  document.querySelectorAll<HTMLButtonElement>("[data-rate]").forEach(
    (button) =>
      (button.onclick = () => {
        rate = Number(button.dataset.rate);
        document
          .querySelectorAll("[data-rate]")
          .forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
      }),
  );
  document.querySelectorAll<HTMLButtonElement>("[data-view]").forEach(
    (button) =>
      (button.onclick = () => {
        enter();
        setView(button.dataset.view as View);
      }),
  );
  document.querySelectorAll<HTMLButtonElement>("[data-experiment]").forEach(
    (button) =>
      (button.onclick = () => {
        const experiment = button.dataset.experiment as Experiment;
        p.experiment = experiment;
        p.lesion =
          experiment === "lesion"
            ? ($<HTMLSelectElement>("#lesion").value as Lesion)
            : "none";
        const info = experiments[experiment];
        $("#experiment-number").textContent = `${info.number} / 04`;
        $("#experiment-kind").textContent =
          `EXPERIENCE ${info.number}${experiment === "motion" || experiment === "light" ? " · EXPLORATORY MODEL" : ""}`;
        $("#experiment-title").textContent = info.title;
        $("#experiment-description").textContent = info.description;
        $("#lesion-options").hidden = experiment !== "lesion";
        document
          .querySelectorAll("[data-experiment]")
          .forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
        invalidate();
      }),
  );
  for (const key of ["direction", "speed", "size", "intensity"] as const) {
    $<HTMLInputElement>(`#${key}`).oninput = (e) => {
      p[key] = Number((e.target as HTMLInputElement).value);
      $(`#${key}-value`).textContent =
        key === "direction"
          ? `${p[key]}°`
          : key === "intensity"
            ? `${Math.round(p[key] * 100)}%`
            : `${p[key].toFixed(2).replace(/0$/, "")}×`;
      invalidate();
    };
  }
  $<HTMLSelectElement>("#lesion").onchange = (e) => {
    p.lesion = (e.target as HTMLSelectElement).value as Lesion;
    invalidate();
  };
  $("#follow").onclick = () => {
    followEnabled = !followEnabled;
    $("#follow").setAttribute("aria-pressed", String(followEnabled));
    scene.following = followEnabled;
    followReturned = false;
    if (followEnabled) {
      setView("brain");
      if (result && time >= result.motorAt) {
        time = 0;
        setPlaying(true);
      }
    }
  };
  scene.onCameraInterrupted = () => {
    followEnabled = false;
    $("#follow").setAttribute("aria-pressed", "false");
  };
  if (scene.reducedMotion)
    $("#follow").title =
      "Reduced motion is enabled. Automatic camera movement is suppressed.";
  $("#zoom-in").onclick = () => scene.zoom(0.8);
  $("#zoom-out").onclick = () => scene.zoom(1.25);
  $("#reset-camera").onclick = () => {
    scene.resetCamera();
    scene.onCameraInterrupted();
  };
  $("#orbit").onkeydown = (event) => {
    if (event.key === "+" || event.key === "=") scene.zoom(0.9);
    else if (event.key === "-") scene.zoom(1.1);
    else if (
      ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)
    ) {
      const offset = scene.camera.position.clone().sub(scene.controls.target);
      if (
        event.key.startsWith("Arrow") &&
        ["ArrowLeft", "ArrowRight"].includes(event.key)
      )
        offset.applyAxisAngle(
          scene.camera.up,
          event.key === "ArrowLeft" ? -0.1 : 0.1,
        );
      else offset.y += event.key === "ArrowUp" ? 0.3 : -0.3;
      scene.camera.position.copy(scene.controls.target).add(offset);
      scene.controls.update();
    } else return;
    event.preventDefault();
  };
  async function region(index: number, focus: boolean) {
    try {
      $("#region-status").textContent =
        index < 0 ? "Overview restored." : "Loading representative skeletons…";
      await scene.loadRegion(index, manifest);
      if (focus && index >= 0) scene.focusRegion(index);
      $("#region-status").textContent =
        index < 0
          ? "Representative skeletons load on demand."
          : `${manifest.regions[index].neurons} representative neuron skeletons loaded.`;
    } catch (error) {
      if (error instanceof Error && error.name !== "AbortError")
        $("#region-status").textContent =
          `Could not load detail. ${error.message}`;
    }
  }
  $<HTMLSelectElement>("#region").onchange = (e) => {
    enter();
    setView("brain");
    autoRegion = Number((e.target as HTMLSelectElement).value);
    region(autoRegion, true);
    if (autoRegion < 0) scene.resetCamera();
  };
  function inspect(index: number) {
    if (!circuit || index < 0) return;
    const n = circuit.neurons[index];
    scene.select(index);
    $("#selection").hidden = false;
    $("#neuron-name").textContent = n.name;
    $("#neuron-role").textContent = n.role;
    $("#neuron-data").replaceChildren();
    for (const [key, value] of [
      ["ID", n.id],
      ["TYPE", n.type],
      ["REGION", manifest.regions[n.region].name],
      ["INPUT PARTNERS", n.inputs.toLocaleString()],
      ["OUTPUT PARTNERS", n.outputs.toLocaleString()],
      ["ACTIVITY", "—"],
    ]) {
      const term = document.createElement("dt"),
        description = document.createElement("dd");
      term.textContent = key;
      description.textContent = value;
      if (key === "ACTIVITY") description.id = "neuron-activity";
      $("#neuron-data").append(term, description);
    }
    $("#connection-result").textContent =
      "Select inputs or outputs to highlight measured partners.";
    $<HTMLSelectElement>("#neuron-select").value = String(index);
  }
  scene.onSelect = inspect;
  $<HTMLSelectElement>("#neuron-select").onchange = (e) =>
    inspect(Number((e.target as HTMLSelectElement).value));
  $("#close-selection").onclick = () => {
    $("#selection").hidden = true;
    scene.select(-1);
    scene.showConnections("all");
  };
  document.querySelectorAll<HTMLButtonElement>("[data-connections]").forEach(
    (button) =>
      (button.onclick = () => {
        const mode = button.dataset.connections as
          | "inputs"
          | "outputs"
          | "isolate"
          | "all";
        const edges = scene.showConnections(mode);
        $("#connection-result").textContent =
          mode === "isolate"
            ? "Selected neuron isolated."
            : mode === "all"
              ? "Full anatomy restored."
              : edges.length
                ? edges
                    .map(
                      ([a, b, w]) =>
                        `${circuit!.neurons[mode === "inputs" ? a : b].name} (${w})`,
                    )
                    .join(" · ")
                : "No partners in this prepared subgraph.";
      }),
  );
  $("#follow-path").onclick = () => {
    if (!result) return;
    const index = scene.selected.value;
    if (!result.path.includes(index)) {
      $("#connection-result").textContent =
        "This neuron is outside the earliest motor-output path. Select a neuron on the active path.";
      return;
    }
    followEnabled = true;
    scene.following = true;
    $("#follow").setAttribute("aria-pressed", "true");
    time = Math.max(0, result.onsets[index]);
    followReturned = false;
    setView("brain");
    setPlaying(true);
  };
  $("#stats-toggle").onclick = () => {
    $("#stats").hidden = !$("#stats").hidden;
    $("#stats-toggle").setAttribute(
      "aria-expanded",
      String(!$("#stats").hidden),
    );
  };
  $<HTMLSelectElement>("#quality").onchange = (e) => {
    const value = (e.target as HTMLSelectElement).value;
    automatic = value === "auto";
    if (!automatic) scene.setQuality(value as Quality);
    else {
      calibrated = false;
      warmSamples = 0;
      fpsHistory.length = 0;
    }
  };
  document.addEventListener("visibilitychange", () => {
    last = 0;
    perf.reset();
  });

  function updatePhases() {
    for (const group of ["visual", "descending", "motor"]) {
      const indices =
        circuit?.neurons.flatMap((n, i) =>
          (
            group === "visual"
              ? ["LC4", "LPLC2"].includes(n.type)
              : group === "descending"
                ? n.type === "DNp01"
                : n.type === "TTMn"
          )
            ? [i]
            : [],
        ) ?? [];
      const active =
        result &&
        indices.some(
          (i) =>
            time >= result!.onsets[i] && time <= result!.onsets[i] + CONDUCTION,
        );
      const reached = result && indices.some((i) => time >= result!.onsets[i]);
      const disabled =
        result && indices.some((i) => result!.blocked.includes(i));
      const row = $(`[data-phase="${group}"]`);
      row.classList.toggle("active", !!active);
      row.classList.toggle("reached", !!reached);
      row.classList.toggle("disabled", !!disabled);
    }
  }
  scene.renderer.setAnimationLoop((now) => {
    if (document.hidden) {
      last = 0;
      return;
    }
    const begin = performance.now();
    const delta = last ? Math.min(0.1, (now - last) / 1000) : 0;
    last = now;
    if (playing && result) {
      time = Math.min(result.duration, time + delta * rate);
      if (time >= result.duration) setPlaying(false);
    }
    if (
      followEnabled &&
      result &&
      Number.isFinite(result.motorAt) &&
      time >= result.motorAt &&
      !followReturned
    ) {
      setView("split");
      scene.following = false;
      followReturned = true;
    }
    scene.render(time, result ? runParameters : p, delta);
    if (now - lastUI > 100) {
      lastUI = now;
      if (result) {
        $("#elapsed").textContent = time.toFixed(2).padStart(5, "0");
        $<HTMLInputElement>("#scrub").value = String(time);
        if (playing && time >= result.motorAt)
          $("#run-status").textContent = "MOTOR OUTPUT → JUMP";
        const selected = scene.selected.value;
        if (selected >= 0 && $("#neuron-activity")) {
          const onset = result.onsets[selected];
          const from = result.parents[selected];
          $("#neuron-activity").textContent = result.blocked.includes(selected)
            ? "Disabled"
            : !Number.isFinite(onset)
              ? "Unreached"
              : time < onset
                ? "Waiting"
                : time < onset + CONDUCTION
                  ? `Active${from >= 0 ? ` ← ${circuit!.neurons[from].type}` : ""}`
                  : "Signal passed";
        }
      }
      updatePhases();
    }
    if (perf.sample(now, performance.now() - begin, scene.bytes)) {
      const s = perf.stats;
      $("#fps").textContent = `${Math.round(s.fps)} FPS`;
      $("#stats").textContent =
        `${scene.backend} · ${scene.quality.toUpperCase()}\n${s.fps.toFixed(1)} FPS · frame ${s.frameMs.toFixed(2)} ms · p95 ${s.p95Ms.toFixed(2)} ms\nCPU submission ${s.cpuMs.toFixed(2)} ms\nGPU ${s.gpuMs === null ? "unavailable" : `${s.gpuMs.toFixed(2)} ms`}\n${s.drawCalls} draw calls · ${s.points.toLocaleString()} points\n${s.lines.toLocaleString()} lines · ${s.triangles.toLocaleString()} triangles\nPrepared buffers ${s.bufferMB.toFixed(2)} MiB\nRenderer-tracked GPU ${s.gpuTrackedMB === null ? "unavailable" : `${s.gpuTrackedMB.toFixed(2)} MiB`} (not total VRAM)`;
      if (automatic) {
        warmSamples++;
        if (!calibrated && warmSamples > 2) {
          fpsHistory.push(s.fps);
          if (fpsHistory.length >= 3) {
            const fps =
              fpsHistory.reduce((a, b) => a + b, 0) / fpsHistory.length;
            scene.setQuality(
              fps > 85 && innerWidth > 900
                ? "ultra"
                : fps > 48
                  ? "high"
                  : fps > 28
                    ? "medium"
                    : "low",
            );
            calibrated = true;
            $<HTMLSelectElement>("#quality").options[0].textContent =
              `AUTO · ${scene.quality.toUpperCase()}`;
          }
        }
        if (calibrated) {
          slowSamples = s.fps < 28 ? slowSamples + 1 : 0;
          if (slowSamples >= 3 && scene.quality !== "low") {
            scene.setQuality(
              scene.quality === "ultra"
                ? "high"
                : scene.quality === "high"
                  ? "medium"
                  : "low",
            );
            $<HTMLSelectElement>("#quality").options[0].textContent =
              `AUTO · ${scene.quality.toUpperCase()}`;
            slowSamples = 0;
          }
        }
      }
      if (
        entered &&
        !scene.following &&
        $<HTMLSelectElement>("#region").value === "-1"
      ) {
        const distance = scene.camera.position.distanceTo(
          scene.controls.target,
        );
        const target =
          distance < 8
            ? scene.controls.target.y < -0.4
              ? 2
              : Math.abs(scene.controls.target.x) > 1.4
                ? 0
                : 1
            : -1;
        if (target !== autoRegion) {
          autoRegion = target;
          region(target, false);
        }
      }
    }
  });
}
start().catch((error) => {
  console.error(error);
  const status = document.querySelector("#loading-text");
  if (status) status.textContent = String(error);
});
