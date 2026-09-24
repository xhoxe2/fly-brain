import "./style.css";
import { LabScene, type Quality } from "./scene";
import { monitor } from "./performance";
import {
  dynamicsJSON,
  loadNeuralAnatomy,
  type DynamicsManifest,
} from "./neural-data";
import type { FeedingEnvironment, FeedingState } from "./embodiment";
import { feedingContact } from "./feeding-geometry";
import { TasteIntroduction, TASTE_STEPS } from "./introduction";
import { EscapeBody } from "./body";

const $ = <T extends HTMLElement = HTMLElement>(selector: string) =>
  document.querySelector<T>(selector)!;
const icon = (path: string) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">${path}</svg>`;
const playIcon = icon('<path d="m9 5 10 7-10 7Z"/>');
const pauseIcon = icon('<path d="M8 5v14M16 5v14"/>');
const resetIcon = icon('<path d="M4 9a8 8 0 1 1 0 7M4 3v6h6"/>');
type VisibleState = Omit<FeedingState, "counts">;
const restingState = (): VisibleState => ({
  timeMs: 0,
  spikes: 0,
  motorRates: [0, 0],
  motorRate: 0,
  extension: 0,
  contact: 0,
  sensoryRates: { sugar: 0, bitter: 0 },
  finger: null,
  body: new EscapeBody().state,
});

function shell() {
  $("#app").innerHTML = `
  <header class="topbar"><a class="wordmark" href="${import.meta.env.BASE_URL}" aria-label="Fly Brain home">FLY<span>//</span>BRAIN</a><span class="header-description">A CONNECTOME IN MOTION</span><div class="header-actions"><span class="live-dot"></span><span class="dataset-tag">FLYWIRE / 630</span><button class="text-button" id="science">The science <span>↗</span></button></div></header>
  <main class="workspace">
    <section class="visualization" aria-label="Fly, food and continuously computed neural activity">
      <div id="stage"></div>
      <button id="finger-handle" hidden aria-label="Move finger: drag, or use left and right arrows for side and up and down for distance">Move finger <span aria-hidden="true">↔</span></button>
      <div class="world-story"><div class="eyebrow"><span id="scene-subject">A MATTER OF TASTE</span><span id="intro-number"></span></div><h1 id="intro-step">Meet the<br><em>feeding circuit.</em></h1>
        <div class="introduction" aria-label="Introduction to the feeding experiment">
          <p id="intro-note" aria-live="polite">Sweet food. A bitter addition. Watch the response change.</p>
          <div class="intro-track" aria-hidden="true">${TASTE_STEPS.map(() => '<span><i></i></span>').join("")}</div>
          <div class="intro-actions"><button id="intro-action" disabled>Loading experiment</button><button id="intro-pause" aria-label="Pause introduction and neural computation" hidden>${pauseIcon}</button></div>
          <span class="intro-disclosure" id="intro-disclosure">Conditions change automatically. Response computed live.</span>
        </div>
      </div>
      <section class="brain-inset" aria-label="FlyWire brain anchor coordinates with computed spikes">
        <div class="brain-inset-heading"><span id="brain-status">The brain</span><div class="camera-tools"><button id="zoom-out" aria-label="Zoom out of brain">−</button><button id="zoom-in" aria-label="Zoom into brain">+</button><button id="reset-camera" aria-label="Reset brain camera">${resetIcon}</button></div></div>
        <div id="orbit" tabindex="0" aria-label="Brain: drag or use arrow keys to rotate; plus and minus to zoom"></div>
        <div class="brain-inset-caption"><span class="legend-dot"></span> Computed spikes <span class="inset-count">127,400 neurons</span><p>Measured positions · FlyWire female 630</p></div>
        <div class="response-note" aria-label="Computed mouth position"><span class="micro-label">PROBOSCIS · MOUTH</span><div><strong id="mouth-response">At rest</strong><span id="mouth-amount">0%</span></div><meter id="mouth-meter" min="0" max="1" value="0" aria-label="Computed proboscis extension"></meter><p><span class="live-dot" id="contact-dot"></span><span id="contact-status">Preparing neural model</span></p></div>
        <div class="finger-response" hidden><span class="micro-label">ESCAPE-RELATED NEURONS · GF</span><div class="gf-pair"><span>Cell L <strong id="gf-left">0 Hz</strong><meter id="gf-left-meter" min="0" max="250" value="0" aria-label="Left GF firing rate"></meter></span><span>Cell R <strong id="gf-right">0 Hz</strong><meter id="gf-right-meter" min="0" max="250" value="0" aria-label="Right GF firing rate"></meter></span></div><svg id="gf-chart" viewBox="0 0 280 35" preserveAspectRatio="none" role="img" aria-label="Computed left and right GF firing rates over the last six model seconds"><path id="gf-left-trace" d="M0 33H280"/><path id="gf-right-trace" d="M0 33H280"/></svg><div class="body-output"><strong id="body-state">On the platform</strong><span id="body-count">0 take-offs</span></div><p>GF spikes → approximate leg mechanics</p></div>
      </section>
      <div class="specimen-note"><span class="specimen-name">Drosophila melanogaster</span><span>Micro-CT body · NeuroMechFly</span></div>
      <div id="loading" role="status"><span class="loading-title">Loading the whole network</span><progress id="load-progress" max="1" value="0" aria-label="Neural network download"></progress><span id="loading-text">About 52 MB · computed locally in your browser</span></div>
    </section>
    <aside class="lab" aria-label="Environmental conditions">
      <div class="lab-heading"><span class="micro-label">THE ENVIRONMENT</span><span class="live-indicator" id="live-state">LOADING</span></div>
      <div class="experiment-copy"><h2>Change its world.</h2><p>Change the food or approach with a finger. Watch the network respond.</p></div>
      <fieldset class="finger-panel" id="finger-environment" disabled><legend>AN APPROACHING FINGER</legend>
        <label class="finger-toggle"><input id="finger-enabled" type="checkbox"> Add a finger to the scene</label>
        <div id="finger-controls" hidden>
          <label class="slider-label" for="finger-side">Target side <output id="finger-side-value">In front</output></label><input id="finger-side" type="range" min="-1" max="1" value="0" step="0.05"><div class="range-ends"><span>FLY'S LEFT</span><span>FLY'S RIGHT</span></div>
          <label class="slider-label" for="finger-distance">Target distance <output id="finger-distance-value">Far away</output></label><input id="finger-distance" type="range" min="0" max="1" value="0" step="0.05"><div class="range-ends"><span>FAR</span><span>CLOSE</span></div>
          <button id="finger-approach">Bring closer <span>→</span></button>
          <p id="finger-input-state" role="status">Move closer to create a looming stimulus.</p>
          <p class="finger-limit">Computed brain output drives a simplified jump mechanism. No touch sensing.</p>
        </div>
      </fieldset>
      <fieldset class="parameters" id="environment" disabled><legend>FOOD CONDITIONS</legend>
        <label class="slider-label" for="sugar">Sweetness <output id="sugar-value">75%</output></label><input id="sugar" type="range" min="0" max="1" value="0.75" step="0.05"><div class="range-ends"><span>NONE</span><span>HIGH</span></div>
        <label class="slider-label" for="bitter">Bitterness <output id="bitter-value">0%</output></label><input id="bitter" type="range" min="0" max="1" value="0" step="0.05"><div class="range-ends"><span>NONE</span><span>HIGH</span></div>
        <label class="slider-label" for="distance">Food position <output id="distance-value">In contact</output></label><input id="distance" type="range" min="0" max="0.7" value="0" step="0.01"><div class="range-ends"><span>AT THE MOUTH</span><span>OUT OF REACH</span></div>
      </fieldset>
      <section class="motor-panel" aria-label="Computed motor output"><div class="motor-heading"><span class="micro-label">MOUTH MOTOR OUTPUT</span><span><strong id="motor-rate">0</strong> Hz</span></div><svg id="motor-chart" viewBox="0 0 280 65" preserveAspectRatio="none" role="img" aria-label="MN9 firing rate over the last twelve model seconds"><path class="chart-grid" d="M0 8H280M0 32H280M0 57H280"/><path id="motor-trace" d="M0 57H280"/></svg><div class="motor-detail"><span>MN9 left <b id="motor-left">0 Hz</b></span><span>MN9 right <b id="motor-right">0 Hz</b></span></div><p id="motor-state">Waiting for computation</p></section>
      <details class="model-notes"><summary>What is actually simulated <span>+</span></summary><p>The full published spiking network: 127,400 neurons and 14.7 million directed connections. Sweet and bitter taste enter through identified sensory neurons. Computed MN9 spikes drive the proboscis.</p><p>The finger's apparent expansion supplies approximate visual input to identified LC4 and LPLC2 neurons. Computed GF spikes excite a simplified middle-leg actuator. Ground force, inertia and gravity determine the body movement; the finger never supplies a jump command.</p><p>Sensory encoding and body mechanics are approximations. Thoracic motor circuits are not simulated. The body can jump vertically; steering, flight, touch, free walking and learning are not implemented.</p><button class="text-button" id="model-details">Model, data & limitations ↗</button></details>
      <p class="scope-note">Published neural model.<br>Approximate body and contact mechanics.</p>
    </aside>
  </main>
  <section class="transport" aria-label="Neural computation controls"><button id="play-pause" aria-label="Pause neural computation" disabled>${pauseIcon}</button><button id="restart" aria-label="Reset neural state with the same random seed" disabled>${resetIcon}</button><div class="time-readout"><strong id="elapsed">00.00</strong><span>MODEL SECONDS</span></div><div class="continuous-state"><span id="run-status" role="status">LOADING NEURAL NETWORK</span><span id="clock-status">Continuous dynamics · no recorded responses</span></div><div class="rates" role="group" aria-label="Requested simulation speed">${[0.25, 0.5, 1].map((rate) => `<button data-rate="${rate}" aria-pressed="${rate === 0.5}">${rate}×</button>`).join("")}</div></section>
  <footer class="statusbar"><span class="source-credit">OPEN SCIENCE <button id="source-notes">Shiu et al. / FlyWire ↗</button></span><span id="neuron-count">127,400 COMPUTED NEURONS</span><div class="runtime"><button id="stats-toggle" aria-expanded="false"><span class="live-dot"></span><span id="fps">— FPS</span></button><span id="backend">INITIALIZING</span><label for="quality" class="sr-only">Graphics quality</label><select id="quality"><option value="auto">AUTO QUALITY</option><option value="high">HIGH</option><option value="medium">MEDIUM</option><option value="low">LOW</option></select></div></footer>
  <pre id="stats" hidden></pre>
  <dialog id="science-dialog"><button id="close-science" class="dialog-close" aria-label="Close scientific notes">×</button><span class="eyebrow">DATA, DYNAMICS, BODY</span><h2>A brain model.<br>With a defined scope.</h2><p>The network comes from the <a href="https://www.nature.com/articles/s41586-024-07763-9" target="_blank" rel="noreferrer">Shiu et al. computational brain model (2024)</a>: 127,400 FlyWire v630 neurons and all 14,687,178 directed connections in its published graph. Synapse counts set connection strength. Transmitter assignments provide the model's signs. No connections are invented or filtered out for this experiment.</p><p>The browser integrates leaky integrate-and-fire dynamics every 0.1 ms, including synaptic current, inhibition, conduction delay and refractory periods. The implementation is checked against the original Brian2 equations on deterministic fixtures. Numerical agreement is not validation of every behavior of a real fly.</p><p>Sweet and bitter input uses the authors' 21 + 21 identified gustatory neurons and Poisson stimulation. The two MN9 motor neurons provide the only command to the mouth. The published work studied feeding-related motor output and bitter suppression, not an autonomous animal choosing arbitrary actions.</p><p><strong>The environment and body are approximate.</strong> Contact scales taste input; 120 ms motor-rate smoothing and a 60 ms actuator lag control extension. The point-contact boundary, concentration-to-rate encoding and joint rotations are authored assumptions, not measured biomechanics. The fly does not walk to food, learn choices or run a hidden language model. Environmental changes preserve neural state; only Reset starts over.</p><p>The inset shows 127,322 measured anchor coordinates matched to this exact graph. These are reference points, not detailed neuron skeletons or uniformly soma positions. The 78 neurons without coordinates still participate in computation. Initial framing uses padded robust coordinate bounds; extreme source points remain in the data and can lie outside the initial view. Gold marks actual computed spikes, visually persisted for readability.</p><p>The body is the <a href="https://github.com/NeLy-EPFL/flygym" target="_blank" rel="noreferrer">NeuroMechFly micro-CT morphology</a> (Apache 2.0), a different female specimen. Pigmentation, bristles, eye facets and the food drop are illustrative. No recorded behavior animation controls the response.</p><p>Full neural data downloads once per cache cycle: approximately 52 MB. All computation stays in this browser. Rendering quality changes only the drawing, not the network. Requested speed may fall below the chosen rate on slower devices.</p><p class="credit"><a href="https://github.com/philshiu/Drosophila_brain_model" target="_blank" rel="noreferrer">Published model & code (MIT) ↗</a><br><a href="https://github.com/flyconnectome/flywire_annotations/releases/tag/v1.1.0" target="_blank" rel="noreferrer">FlyWire annotations ↗</a><br>Model graph and sources: Shiu et al., FlyWire / Dorkenwald et al. See bundled data notices for attribution and licenses.</p></dialog>`;
}

async function start() {
  shell();
  const dialog = $<HTMLDialogElement>("#science-dialog");
  dialog.querySelector(".credit")!.insertAdjacentHTML("beforebegin", `<p><strong>The finger supplies an approximate looming feature, not a retinal image.</strong> A spherical fingertip approximation and broad left/right viewing weights convert positive angular expansion into Poisson input to 104 LC4 and 210 LPLC2 neurons, identified by exact FlyWire 630 IDs. The distinction between LC4 velocity and LPLC2 size contributions follows <a href="https://doi.org/10.1016/j.cub.2019.01.079" target="_blank" rel="noreferrer">Ache et al. (2019)</a>; the geometry, uniform population input and 0–100 Hz gains are authored assumptions. Receptive fields, retinal processing and touch are not modeled. Unchanging relative geometry supplies no new looming drive. Body movement changes the eye positions, so a stationary finger can still expand in the fly's modeled view.</p><p>The displayed bilateral giant-fiber (GF/DNp01) rates come only from computed spikes, smoothed over 120 ms. These escape-related neurons are never directly stimulated. Their activity does not establish intent or an escape direction. Raw GF spike counts, not these smoothed rates, now excite a phenomenological middle-leg muscle. A bounded extensor stroke pushes only against the platform; gravity and damping govern rise and landing. The body has one vertical degree of freedom and cannot steer or fly. Eye and mouth coordinates move with it, changing the next sensory input. The finger follows your target at a bounded speed in model time; the renderer shows that same computed position.</p><p><strong>The jump actuator is not a reconstructed motor connectome.</strong> GF-to-TTM muscle coupling is supported by <a href="https://www.janelia.org/sites/default/files/Library/nn.3741.pdf" target="_blank" rel="noreferrer">von Reyn et al. (2014)</a>, but its thoracic TTMn/PSI/DLM pathways are absent here. We bypass them with an authored muscle/spring model. Excitation gain and decay, stiffness, damping, gravity and the slower scene-unit mechanics are not calibrated to fly biomechanics. GF spikes are grouped into 10 ms intervals and mechanics use 1 ms steps; natural submillisecond motor latency is not reproduced. Natural looming can evoke only one or two GF spikes, unlike the longer trains our approximate visual encoder can produce. This is not a validated natural escape or a fear measurement. Wings remain folded; no timed takeoff clip is played.</p><p>The finger is a derivative of the <a href="https://static.makehumancommunity.org/about/license.html" target="_blank" rel="noreferrer">MakeHuman CC0 base mesh</a>, an authored anatomical illustration, not a scan. See the bundled finger notice for the pinned source and processing.</p>`);
  for (const id of ["science", "model-details", "source-notes"])
    $("#" + id).onclick = () => dialog.showModal();
  $("#close-science").onclick = () => dialog.close();
  const environment: FeedingEnvironment = {
    sugar: 0.75,
    bitter: 0,
    foodDistance: 0,
    finger: { enabled: false, lateral: 0, proximity: 0 },
  };
  let introduction: TasteIntroduction | null = null;
  const refreshFinger = () => {
    const finger = environment.finger!;
    $<HTMLInputElement>("#finger-enabled").checked = finger.enabled;
    $("#finger-controls").hidden = !finger.enabled;
    $(".finger-response").hidden = !finger.enabled;
    $(".response-note").hidden = finger.enabled;
    $("#scene-subject").textContent = finger.enabled ? "AN APPROACHING OBJECT" : "A MATTER OF TASTE";
    $<HTMLInputElement>("#finger-side").value = String(finger.lateral);
    $<HTMLInputElement>("#finger-distance").value = String(finger.proximity);
    $("#finger-side-value").textContent = Math.abs(finger.lateral) < 0.1 ? "In front" : `${Math.round(Math.abs(finger.lateral) * 100)}% ${finger.lateral < 0 ? "left" : "right"}`;
    $("#finger-distance-value").textContent = finger.proximity === 0 ? "Far away" : finger.proximity === 1 ? "Close · no contact" : `${Math.round(finger.proximity * 100)}% closer`;
    $("#finger-approach").innerHTML = finger.proximity < 0.5 ? "Bring closer <span>→</span>" : "Move away <span>←</span>";
  };
  const refreshEnvironment = () => {
    for (const [id, key] of [["sugar", "sugar"], ["bitter", "bitter"], ["distance", "foodDistance"]] as const) {
      $<HTMLInputElement>("#" + id).value = String(environment[key]);
      $("#" + id + "-value").textContent = key === "foodDistance"
        ? environment[key] === 0 ? "At the mouth" : environment[key] === 0.7
          ? "Out of reach" : `${Math.round(environment[key] / 0.7 * 100)}% away`
        : `${Math.round(environment[key] * 100)}%`;
    }
  };
  const finishIntroduction = (completed = false) => {
    introduction = null;
    $("#intro-step").textContent = "Your turn.";
    $("#intro-number").textContent = "";
    $("#intro-note").textContent = completed
      ? "Sweet, bitter, sweet again. Now change the conditions yourself."
      : "Change the food or move the finger. Watch the computed response.";
    $("#intro-action").textContent = "Replay · about 25 s";
    $("#intro-pause").hidden = true;
    $("#intro-disclosure").textContent = "The brain keeps its state as you change conditions.";
    $(".intro-track").hidden = true;
  };
  for (const [id, key] of [
    ["sugar", "sugar"],
    ["bitter", "bitter"],
    ["distance", "foodDistance"],
  ] as const) {
    const input = $<HTMLInputElement>("#" + id);
    input.oninput = () => {
      if (introduction) finishIntroduction();
      environment[key] = Number(input.value);
      refreshEnvironment();
    };
  }
  const changeFinger = () => {
    if (introduction) finishIntroduction();
    refreshFinger();
    if (environment.finger!.enabled) {
      $("#intro-step").textContent = "A closer look.";
      $("#intro-note").textContent = "Move the finger closer. Watch neural activity drive the legs and lift the body.";
      $("#intro-disclosure").textContent = "Computed neural output. Approximate sensory and body mechanics.";
    } else finishIntroduction();
  };
  $<HTMLInputElement>("#finger-enabled").onchange = event => {
    environment.finger!.enabled = (event.target as HTMLInputElement).checked;
    changeFinger();
  };
  for (const [id, key] of [["finger-side", "lateral"], ["finger-distance", "proximity"]] as const) {
    $<HTMLInputElement>("#" + id).oninput = event => {
      environment.finger![key] = Number((event.target as HTMLInputElement).value);
      changeFinger();
    };
  }
  $("#finger-approach").onclick = () => {
    environment.finger!.proximity = environment.finger!.proximity < 0.5 ? 1 : 0;
    changeFinger();
  };
  let scene: LabScene;
  try {
    const manifest = await dynamicsJSON<DynamicsManifest>("manifest.json");
    const { overview, background } = await loadNeuralAnatomy(manifest);
    scene = await new LabScene(
      $("#stage"),
      $("#orbit"),
      overview,
      background,
    ).init();
    scene.enableFeeding(overview.owners, manifest.neuronCount);
  } catch (error) {
    $("#loading-text").textContent =
      `The scene could not start. ${String(error)}. Reload to retry.`;
    $(".loading-title").textContent = "Scene unavailable";
    $("#live-state").textContent = "UNAVAILABLE";
    $("#run-status").textContent = "SCENE UNAVAILABLE";
    console.error(error);
    return;
  }
  $("#backend").textContent = scene.backend.toUpperCase();
  const fingerHandle = $("#finger-handle");
  let draggingFinger = false;
  let dragOffset = [0, 0];
  fingerHandle.onpointerdown = event => {
    const screen = scene.getFingerScreenPosition();
    if (!screen) return;
    const bounds = $("#stage").getBoundingClientRect();
    dragOffset = [event.clientX - bounds.left - screen.x, event.clientY - bounds.top - screen.y];
    draggingFinger = true;
    fingerHandle.setPointerCapture(event.pointerId);
  };
  fingerHandle.onpointermove = event => {
    if (!draggingFinger) return;
    const bounds = $("#stage").getBoundingClientRect();
    const position = scene.fingerPointFromScreen(event.clientX - bounds.left - dragOffset[0], event.clientY - bounds.top - dragOffset[1]);
    if (!position) return;
    environment.finger!.lateral = Math.round(Math.max(-1, Math.min(1, position[0] / 1.4)) * 20) / 20;
    environment.finger!.proximity = Math.round(Math.max(0, Math.min(1, (position[2] + 4) / 2.35)) * 20) / 20;
    changeFinger();
  };
  fingerHandle.onpointerup = fingerHandle.onpointercancel = fingerHandle.onlostpointercapture = () => { draggingFinger = false; };
  fingerHandle.onkeydown = event => {
    const finger = environment.finger!;
    if (!event.key.startsWith("Arrow")) return;
    event.preventDefault();
    if (event.key === "ArrowLeft" || event.key === "ArrowRight")
      finger.lateral = Math.max(-1, Math.min(1, finger.lateral + (event.key === "ArrowLeft" ? -0.1 : 0.1)));
    else finger.proximity = Math.max(0, Math.min(1, finger.proximity + (event.key === "ArrowUp" ? 0.1 : -0.1)));
    changeFinger();
  };
  $("#zoom-out").onclick = () => scene.zoom(1.2);
  $("#zoom-in").onclick = () => scene.zoom(1 / 1.2);
  $("#reset-camera").onclick = () => scene.resetCamera();
  $("#orbit").onkeydown = (event) => {
    if (["+", "=", "-"].includes(event.key)) {
      scene.zoom(event.key === "-" ? 1.2 : 1 / 1.2);
      event.preventDefault();
    }
    if (event.key.startsWith("Arrow")) {
      const vector = scene.camera.position.clone().sub(scene.controls.target);
      const step = 0.1;
      if (event.key === "ArrowLeft" || event.key === "ArrowRight")
        vector.applyAxisAngle(
          scene.camera.up,
          event.key === "ArrowLeft" ? -step : step,
        );
      else
        vector.y +=
          event.key === "ArrowUp"
            ? step * vector.length()
            : -step * vector.length();
      scene.camera.position.copy(scene.controls.target).add(vector);
      scene.controls.update();
      event.preventDefault();
    }
  };
  const perf = monitor(scene.renderer);
  const worker = new Worker(new URL("./brain.worker.ts", import.meta.url), {
    type: "module",
  });
  let state = restingState(),
    ready = false,
    playing = false,
    busy = false,
    resetting = false;
  let requestId = 0,
    rate = 0.5,
    last = 0,
    budget = 0,
    lastUI = 0,
    computeMs = 0;
  let lastClock = 0,
    lastModelTime = 0,
    effectiveRate = 0,
    slowFrames = 0;
  let automatic = true;
  const history: { time: number; rate: number; gf: [number, number] }[] = [];
  const refreshPlay = () => {
    $("#brain-status").textContent = ready
      ? playing
        ? "The brain, computing"
        : "The brain, paused"
      : "The brain";
    $("#play-pause").innerHTML = playing ? pauseIcon : playIcon;
    $("#intro-pause").innerHTML = playing ? pauseIcon : playIcon;
    $("#intro-pause").setAttribute("aria-label", playing
      ? "Pause introduction and neural computation" : "Resume introduction and neural computation");
    $("#play-pause").setAttribute(
      "aria-label",
      playing ? "Pause neural computation" : "Resume neural computation",
    );
    $("#live-state").textContent = ready
      ? playing
        ? "COMPUTING"
        : "PAUSED"
      : "LOADING";
    $("#run-status").textContent = ready
      ? playing
        ? "CONTINUOUS NEURAL COMPUTATION"
        : "NEURAL STATE PAUSED"
      : "LOADING NEURAL NETWORK";
  };
  const togglePlaying = () => {
    playing = !playing;
    budget = 0;
    refreshPlay();
  };
  $("#play-pause").onclick = $("#intro-pause").onclick = togglePlaying;
  const applyIntroductionStep = () => {
    if (!introduction) return;
    const step = TASTE_STEPS[introduction.step];
    environment.sugar = 0.75;
    environment.bitter = step.bitter;
    environment.foodDistance = 0;
    refreshEnvironment();
    $("#intro-step").textContent = step.title;
    $("#intro-number").textContent = `${introduction.step + 1} / ${TASTE_STEPS.length}`;
    $("#intro-note").textContent = step.note;
  };
  const startIntroduction = () => {
    if (!ready || resetting) return;
    introduction = new TasteIntroduction(state.timeMs);
    environment.finger!.enabled = false;
    refreshFinger();
    playing = true;
    budget = 0;
    applyIntroductionStep();
    $("#intro-action").textContent = "Take control";
    $("#intro-pause").hidden = false;
    $(".intro-track").hidden = false;
    $("#intro-disclosure").textContent = "Conditions change automatically. Response computed live.";
    refreshPlay();
  };
  $("#intro-action").onclick = () => introduction ? finishIntroduction() : startIntroduction();
  $("#restart").onclick = () => {
    if (!ready || resetting) return;
    if (introduction) finishIntroduction();
    resetting = true;
    busy = true;
    requestId++;
    budget = 0;
    worker.postMessage({ type: "reset", seed: 1 });
  };
  document.querySelectorAll<HTMLButtonElement>("[data-rate]").forEach(
    (button) =>
      (button.onclick = () => {
        rate = Number(button.dataset.rate);
        budget = 0;
        document
          .querySelectorAll("[data-rate]")
          .forEach((item) =>
            item.setAttribute("aria-pressed", String(item === button)),
          );
      }),
  );
  $<HTMLSelectElement>("#quality").onchange = (event) => {
    const value = (event.target as HTMLSelectElement).value;
    automatic = value === "auto";
    scene.setQuality(automatic ? "high" : (value as Quality));
    slowFrames = 0;
  };
  $("#stats-toggle").onclick = () => {
    $("#stats").hidden = !$("#stats").hidden;
    $("#stats-toggle").setAttribute(
      "aria-expanded",
      String(!$("#stats").hidden),
    );
  };
  const fail = (message: string) => {
    if (introduction) finishIntroduction();
    ready = playing = false;
    busy = false;
    $("#loading").hidden = false;
    $(".loading-title").textContent = "Neural model unavailable";
    $("#brain-status").textContent = "Computation stopped";
    $("#loading-text").textContent =
      `Computation stopped: ${message}. Reload to retry.`;
    $("#live-state").textContent = "UNAVAILABLE";
    $("#run-status").textContent = "NEURAL MODEL UNAVAILABLE";
    $("#environment").setAttribute("disabled", "");
    $("#finger-environment").setAttribute("disabled", "");
    fingerHandle.setAttribute("disabled", "");
    $("#play-pause").setAttribute("disabled", "");
    $("#restart").setAttribute("disabled", "");
    $("#intro-action").setAttribute("disabled", "");
    console.error(message);
  };
  worker.onerror = (event) => fail(event.message || "Neural worker failed");
  worker.onmessage = (event) => {
    const message = event.data;
    if (message.type === "loading") {
      $<HTMLProgressElement>("#load-progress").value = message.fraction;
      $("#loading-text").textContent =
        `${Math.round(message.fraction * 100)}% · full graph, no pruned connections`;
    } else if (message.type === "ready") {
      ready = playing = true;
      $("#loading").hidden = true;
      for (const id of ["environment", "finger-environment", "play-pause", "restart", "intro-action"])
        $("#" + id).removeAttribute("disabled");
      startIntroduction();
    } else if (message.type === "reset") {
      state = restingState();
      history.length = 0;
      scene.clearSpikes();
      busy = resetting = false;
      lastModelTime = 0;
      lastClock = performance.now();
    } else if (
      message.type === "frame" &&
      message.id === requestId &&
      !resetting
    ) {
      state = message.state;
      computeMs = message.computeMs;
      busy = false;
      scene.recordSpikes(message.indices, state.timeMs);
      history.push({ time: state.timeMs, rate: state.motorRate, gf: state.finger?.gfRates ?? [0, 0] });
      while (history.length && history[0].time < state.timeMs - 12_000)
        history.shift();
    } else if (message.type === "error") fail(message.message);
  };
  document.addEventListener("visibilitychange", () => {
    last = 0;
    budget = 0;
    perf.reset();
  });
  worker.postMessage({ type: "init", seed: 1 });
  function frame(now: number) {
    requestAnimationFrame(frame);
    if (document.hidden) return;
    const elapsed = last ? Math.min(100, now - last) : 0;
    last = now;
    if (ready && playing && !resetting) {
      if (introduction?.advance(elapsed, state.timeMs)) {
        if (introduction.done) finishIntroduction(true);
        else applyIntroductionStep();
      }
      budget = Math.min(40, budget + elapsed * rate);
      if (!busy && budget >= 20) {
        budget -= 20;
        busy = true;
        requestId++;
        worker.postMessage({
          type: "advance",
          id: requestId,
          durationMs: 20,
          environment: { ...environment },
        });
      }
    }
    const before = performance.now();
    scene.setFinger(environment.finger!.enabled ? state.finger?.position ?? null : null);
    scene.renderFeeding(
      state.timeMs,
      state.extension,
      environment.foodDistance,
      environment.bitter,
      state.body,
    );
    const fingerScreen = scene.getFingerScreenPosition();
    fingerHandle.hidden = !fingerScreen;
    if (fingerScreen) {
      fingerHandle.style.left = `${Math.max(66, Math.min($("#stage").clientWidth - 66, fingerScreen.x + 70))}px`;
      fingerHandle.style.top = `${fingerScreen.y}px`;
    }
    if (perf.sample(now, performance.now() - before, scene.bytes)) {
      $("#fps").textContent = `${Math.round(perf.stats.fps)} FPS`;
      if (automatic && perf.stats.samples > 120) {
        slowFrames = perf.stats.fps < 42 ? slowFrames + 1 : 0;
        if (slowFrames >= 3 && scene.quality !== "low") {
          scene.setQuality(scene.quality === "high" ? "medium" : "low");
          slowFrames = 0;
        }
      }
    }
    if (now - lastUI < 100) return;
    lastUI = now;
    if (now - lastClock > 1000) {
      effectiveRate = (state.timeMs - lastModelTime) / (now - lastClock);
      lastClock = now;
      lastModelTime = state.timeMs;
    }
    $("#elapsed").textContent = (state.timeMs / 1000)
      .toFixed(2)
      .padStart(5, "0");
    $("#motor-rate").textContent = state.motorRate.toFixed(0);
    $("#motor-left").textContent = state.motorRates[0].toFixed(0) + " Hz";
    $("#motor-right").textContent = state.motorRates[1].toFixed(0) + " Hz";
    $("#motor-state").textContent =
      `Proboscis extension · ${Math.round(state.extension * 100)}%`;
    $("#mouth-response").textContent = state.extension > 0.55
      ? "Mouth extended" : state.extension > 0.12 ? "Partly extended" : "Mouth relaxed";
    $("#mouth-amount").textContent = `${Math.round(state.extension * 100)}%`;
    $("#body-state").textContent = !state.body.grounded
      ? state.body.velocity >= 0 ? "Rising" : "Falling"
      : state.body.height > 0.02 ? "Legs extended" : "On the platform";
    $("#body-count").textContent = `${state.body.takeoffs} ${state.body.takeoffs === 1 ? "take-off" : "take-offs"}`;
    for (const [side, index] of [["left", 0], ["right", 1]] as const) {
      const activity = state.finger?.gfRates[index] ?? 0;
      $("#gf-" + side).textContent = `${activity.toFixed(0)} Hz`;
      $<HTMLMeterElement>("#gf-" + side + "-meter").value = activity;
    }
    if (environment.finger!.enabled) {
      const input = state.finger?.sensoryRates ?? [0, 0, 0, 0];
      $("#finger-input-state").textContent = !playing ? "Paused · resume to move and compute" : Math.max(...input) > 0.1
        ? `Looming input · L ${Math.max(input[0], input[2]).toFixed(0)} / R ${Math.max(input[1], input[3]).toFixed(0)} Hz`
        : "No new visual expansion · neural state continues";
    }
    $<HTMLMeterElement>("#mouth-meter").value = state.extension;
    if (introduction) {
      document.querySelectorAll<HTMLElement>(".intro-track i").forEach((segment, index) => {
        const amount = index < introduction!.step ? 1 : index === introduction!.step
          ? introduction!.progress(state.timeMs) : 0;
        segment.style.transform = `scaleX(${amount})`;
      });
    }
    const contact = feedingContact(state.extension, environment.foodDistance, state.body.height);
    $("#contact-status").textContent = !ready
      ? "Preparing neural model"
      : contact > 0.01
        ? "Food in sensory contact"
        : "No food contact";
    $("#contact-dot").classList.toggle("inactive", contact <= 0.01);
    $("#clock-status").textContent = ready
      ? `${effectiveRate.toFixed(2)}× actual speed · state preserved as conditions change`
      : "Continuous dynamics · no recorded responses";
    const points = history.filter(
      (_, i) => i % 3 === 0 || i === history.length - 1,
    );
    const path = points
      .map(
        (p, i) =>
          `${i ? "L" : "M"}${((280 * (p.time - state.timeMs + 12_000)) / 12_000).toFixed(1)},${(57 - (Math.min(150, p.rate) / 150) * 49).toFixed(1)}`,
      )
      .join(" ");
    $("#motor-trace").setAttribute("d", path || "M0 57H280");
    for (const [side, index] of [["left", 0], ["right", 1]] as const) {
      const trace = points.filter(p => p.time >= state.timeMs - 6000).map((p, i) =>
        `${i ? "L" : "M"}${(280 * (p.time - state.timeMs + 6000) / 6000).toFixed(1)},${(33 - Math.min(250, p.gf[index]) / 250 * 30).toFixed(1)}`).join(" ");
      $("#gf-" + side + "-trace").setAttribute("d", trace || "M0 33H280");
    }
    if (!$("#stats").hidden)
      $("#stats").textContent =
        `Rendering: ${perf.stats.fps.toFixed(1)} FPS / ${scene.quality}\np95: ${perf.stats.p95Ms.toFixed(1)} ms\nDraw calls: ${perf.stats.drawCalls}\nNeural compute: ${computeMs.toFixed(1)} ms / 20 ms simulated\nNetwork: 127,400 neurons / 14,687,178 pairs\nNext sweet input: ${(environment.sugar * contact * 200).toFixed(1)} Hz × 21\nNext bitter input: ${(environment.bitter * contact * 200).toFixed(1)} Hz × 21\nBody height: ${state.body.height.toFixed(3)} / velocity: ${state.body.velocity.toFixed(3)}\nLeg activation: ${state.body.activation.map(value => value.toFixed(3)).join(" / ")}\nTake-offs: ${state.body.takeoffs} / peak height: ${state.body.peakHeight.toFixed(3)}\nCurrent contact: ${contact.toFixed(2)}\nSeed: 1 (reset is reproducible)`;
  }
  requestAnimationFrame(frame);
}
start().catch(console.error);
