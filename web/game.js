// URL: ?model=ladder or ?model=rf, ?selftest=1
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.min.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";

const params = new URLSearchParams(location.search);
const MODEL_DIR = { ladder: "model/", rf: "rf/" }[params.get("model")] ?? "model/";
const TOUCH = document.documentElement.classList.contains("touch");
const WIDTH = 128, HEIGHT = 96;
const ACTION_SIZE = 8;
const OUTPUTS = ["x0", "rgba", "k_new", "v_new"];

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const median = (values) => [...values].sort((a, b) => a - b)[values.length >> 1];
const product = (shape) => shape.reduce((a, b) => a * b);
const megabytes = (bytes) => (bytes / 1e6).toFixed(0);
const asFloats = (bytes) => new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);

function randn(out) {
  for (let i = 0; i < out.length; i += 2) {
    const radius = Math.sqrt(-2 * Math.log(1 - Math.random())), angle = 2 * Math.PI * Math.random();
    out[i] = radius * Math.cos(angle);
    if (i + 1 < out.length) out[i + 1] = radius * Math.sin(angle);
  }
  return out;
}

function contentLength(response) {
  return response.headers.get("content-encoding") ? 0 : Number(response.headers.get("content-length")) || 0;
}

async function readBody(response, onChunk) {
  const size = contentLength(response);
  if (!size) return new Uint8Array(await response.arrayBuffer());
  const body = new Uint8Array(size); // one buffer of the final size: iOS Safari closes tabs that take too much memory
  let offset = 0;
  for (const reader = response.body.getReader(); ;) {
    const { done, value } = await reader.read();
    if (done) break;
    body.set(value, offset);
    offset += value.length;
    onChunk(value.length);
  }
  return body.subarray(0, offset);
}

async function download(urls, onProgress = () => {}) {
  const responses = await Promise.all(urls.map((url) => fetch(url)));
  responses.forEach((response, i) => { if (!response.ok) throw new Error(`${urls[i]}: HTTP ${response.status}`); });
  const total = responses.reduce((sum, response) => sum + contentLength(response), 0);
  let loaded = 0;
  return Promise.all(responses.map((response) => readBody(response, (n) => onProgress((loaded += n), total))));
}

function readStarts(bytes, meta) {
  const [count, frames, tokens, channels] = meta.starts_bin.latents;
  const latentsSize = frames * tokens * channels, actionsSize = frames * ACTION_SIZE;
  const floats = asFloats(bytes), actionsAt = count * latentsSize;
  return meta.starts.map(({ level }, i) => ({
    level,
    latents: floats.subarray(i * latentsSize, (i + 1) * latentsSize),
    actions: floats.subarray(actionsAt + i * actionsSize, actionsAt + (i + 1) * actionsSize),
  }));
}

function tensorShapes(meta) {
  const latent = [1, meta.tokens, meta.channels];
  const cache = [meta.layers, meta.heads, meta.slots * meta.tokens, meta.head_dim];
  const frameKV = [meta.layers, meta.heads, meta.tokens, meta.head_dim];
  return { x: latent, z: latent, ctx: latent, use_ctx: [1], a: [1, ACTION_SIZE], pos: [1], n_valid: [1],
           cache_k: cache, cache_v: cache, x0: latent, rgba: [HEIGHT, WIDTH, 4], k_new: frameKV, v_new: frameKV };
}

class World {
  constructor(session, meta) {
    this.session = session;
    this.meta = meta;
    this.shapes = tensorShapes(meta);
    this.latentSize = product(this.shapes.x);
    this.kvFrameSize = meta.tokens * meta.head_dim;
    this.kvRows = meta.layers * meta.heads;
    this.zeros = new Float32Array(this.latentSize);
    this.noiseX = new Float32Array(this.latentSize);
    this.noiseZ = new Float32Array(this.latentSize);
    this.pixels = new Uint8ClampedArray(WIDTH * HEIGHT * 4);
    this.image = new ImageData(this.pixels, WIDTH, HEIGHT);
  }

  get frameNumber() {
    return this.time - this.meta.context;
  }

  async reset(start, noise = null) {
    this.clearCache();
    this.cached = 0;
    const n = this.latentSize;
    for (let i = 0; i < this.meta.context; i++) {
      await this.step({
        x: this.zeros,
        z: noise ? noise.subarray(i * n, (i + 1) * n) : randn(this.noiseZ),
        ctx: start.latents.subarray(i * n, (i + 1) * n),
        useCtx: 1,
        action: start.actions.subarray(i * ACTION_SIZE, (i + 1) * ACTION_SIZE),
        pos: i,
      });
    }
    this.time = this.meta.context;
  }

  async frame(action, noise = null) {
    const x = noise?.x ?? randn(this.noiseX), z = noise?.z ?? randn(this.noiseZ);
    await this.step({ x, z, ctx: this.zeros, useCtx: 0, action, pos: this.time });
    this.time += 1;
  }

  async step({ x, z, ctx, useCtx, action, pos }) {
    const slots = this.meta.slots, scalar = (v) => Float32Array.of(v);
    await this.run({ x, z, ctx, use_ctx: scalar(useCtx), a: action, pos: scalar(pos),
                     n_valid: scalar(Math.min(this.cached, slots)) });
    this.writeCacheSlot(pos % slots);
    this.cached += 1;
  }

  async draw(canvas) {
    await this.readPixels();
    canvas.putImageData(this.image, 0, 0);
  }
}

class CapturedWorld extends World {
  constructor(session, meta, device) {
    super(session, meta);
    this.device = device;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.buffers = {};
    this.inputs = {};
    for (const [name, shape] of Object.entries(this.shapes)) {
      if (OUTPUTS.includes(name)) continue;
      const bytes = product(shape) * 4;
      this.buffers[name] = device.createBuffer({ size: Math.ceil(bytes / 16) * 16, usage });
      this.inputs[name] = ort.Tensor.fromGpuBuffer(this.buffers[name], { dataType: "float32", dims: shape });
    }
    const readable = (name) => device.createBuffer({ size: product(this.shapes[name]) * 4,
                                                     usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.readBuffers = { x0: readable("x0"), rgba: readable("rgba") };
  }

  clearCache() {
    const encoder = this.device.createCommandEncoder();
    encoder.clearBuffer(this.buffers.cache_k);
    encoder.clearBuffer(this.buffers.cache_v);
    this.device.queue.submit([encoder.finish()]);
  }

  async run(feeds) {
    for (const [name, data] of Object.entries(feeds)) this.device.queue.writeBuffer(this.buffers[name], 0, data);
    // outputs in ORT's own buffers: ORT 1.31 re-registers our buffers on every run, and the captured graph keeps
    // the first run's handles, so outputs written to our buffers fail on a later run (onnxruntime PR 33985)
    const outputs = await this.session.run(this.inputs);
    for (const name of OUTPUTS) {
      this.buffers[name] = outputs[name].gpuBuffer;
      outputs[name].dispose();
    }
  }

  writeCacheSlot(slot) {
    const { buffers, meta } = this, bytes = this.kvFrameSize * 4;
    const encoder = this.device.createCommandEncoder();
    for (let row = 0; row < this.kvRows; row++) {
      const from = row * bytes, to = (row * meta.slots + slot) * bytes;
      encoder.copyBufferToBuffer(buffers.k_new, from, buffers.cache_k, to, bytes);
      encoder.copyBufferToBuffer(buffers.v_new, from, buffers.cache_v, to, bytes);
    }
    this.device.queue.submit([encoder.finish()]);
  }

  async readOutput(name, into) {
    const target = this.readBuffers[name], encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.buffers[name], 0, target, 0, target.size);
    this.device.queue.submit([encoder.finish()]);
    await target.mapAsync(GPUMapMode.READ);
    into.set(new Float32Array(target.getMappedRange()));
    target.unmap();
    return into;
  }

  readLatent() {
    return this.readOutput("x0", new Float32Array(this.latentSize));
  }

  readPixels() {
    return this.readOutput("rgba", this.pixels);
  }
}

class ArrayWorld extends World {
  tensor(name, data) {
    return new ort.Tensor("float32", data, this.shapes[name]);
  }

  clearCache() {
    this.cache = {
      cache_k: this.tensor("cache_k", new Float32Array(product(this.shapes.cache_k))),
      cache_v: this.tensor("cache_v", new Float32Array(product(this.shapes.cache_v))),
    };
  }

  async run(feeds) {
    const inputs = { ...this.cache };
    for (const [name, data] of Object.entries(feeds)) inputs[name] = this.tensor(name, data);
    this.results = await this.session.run(inputs);
  }

  writeCacheSlot(slot) {
    const size = this.kvFrameSize, slots = this.meta.slots;
    for (const [cache, fresh] of [["cache_k", "k_new"], ["cache_v", "v_new"]]) {
      const target = this.cache[cache].data, source = this.results[fresh].data;
      for (let row = 0; row < this.kvRows; row++) {
        target.set(source.subarray(row * size, (row + 1) * size), (row * slots + slot) * size);
      }
    }
  }

  async readLatent() {
    return this.results.x0.data.slice();
  }

  async readPixels() {
    this.pixels.set(this.results.rgba.data);
  }
}

async function createWorld(model, meta, webgpu, firstStart) {
  const options = { graphOptimizationLevel: "all", enableCpuMemArena: false, enableMemPattern: false };
  let fallbackReason;
  if (webgpu) {
    let session;
    try {
      session = await ort.InferenceSession.create(model, {
        ...options,
        executionProviders: [{ name: "webgpu" }], // object form: otherwise ORT ignores enableGraphCapture
        enableGraphCapture: true,
      });
      const world = new CapturedWorld(session, meta, ort.env.webgpu.device);
      await world.reset(firstStart); // a graph that cannot be captured fails on its first run
      return world;
    } catch (error) {
      console.warn("graph capture failed, using plain arrays:", error);
      await session?.release().catch(() => {});
      fallbackReason = String(error.message || error);
    }
  }
  const session = await ort.InferenceSession.create(model, { ...options, executionProviders: [webgpu ? "webgpu" : "wasm"] });
  return Object.assign(new ArrayWorld(session, meta), { fallbackReason });
}

const held = new Set();
const GAME_KEYS = new Set(["KeyW", "KeyA", "KeyS", "KeyD", "KeyR", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
                           "Space", "ShiftLeft", "ShiftRight", "ControlLeft", "ControlRight", "Escape"]);

function readAction(meta) {
  const pressed = (...codes) => (codes.some((code) => held.has(code)) ? 1 : 0);
  const index = meta.actions, action = new Float32Array(ACTION_SIZE);
  action[index.fwd] = pressed("KeyW", "ArrowUp");
  action[index.back] = pressed("KeyS", "ArrowDown");
  action[index.left] = pressed("KeyA");
  action[index.right] = pressed("KeyD");
  action[index.run] = pressed("ShiftLeft", "ShiftRight");
  action[index.attack] = pressed("Space", "ControlLeft", "ControlRight");
  action[index.yaw] = (meta.yaw_deg_per_frame * (pressed("ArrowLeft") - pressed("ArrowRight"))) / meta.yaw_scale; // + = left
  return action;
}

function setKeys(keys) {
  for (const [code, down] of Object.entries(keys)) down ? held.add(code) : held.delete(code);
}

function trackDrag(pad, onDrag) {
  const knob = pad.querySelector(".knob");
  let finger = null, startX = 0, startY = 0;
  const moveKnob = (event) => {
    const box = pad.getBoundingClientRect();
    knob.style.left = `${event.clientX - box.left}px`;
    knob.style.top = `${event.clientY - box.top}px`;
  };
  pad.addEventListener("pointerdown", (event) => {
    if (finger !== null || event.target !== pad) return; // the Fire button inside the pad has its own finger
    event.preventDefault();
    [finger, startX, startY] = [event.pointerId, event.clientX, event.clientY];
    pad.setPointerCapture(finger);
    moveKnob(event);
    knob.style.display = "block";
  });
  pad.addEventListener("pointermove", (event) => {
    if (event.pointerId !== finger) return;
    moveKnob(event);
    onDrag(event.clientX - startX, event.clientY - startY);
  });
  const release = (event) => {
    if (event.pointerId !== finger) return;
    finger = null;
    knob.style.display = "none";
    onDrag(0, 0);
  };
  pad.addEventListener("pointerup", release);
  pad.addEventListener("pointercancel", release);
}

function setupTouchControls() {
  const DEAD_ZONE = 14, RUN_DISTANCE = 70, DIAGONAL = Math.sin(Math.PI / 8);
  trackDrag($("pad-move"), (dx, dy) => {
    const distance = Math.hypot(dx, dy);
    const toward = (component) => distance > DEAD_ZONE && component > DIAGONAL * distance;
    setKeys({ KeyW: toward(-dy), KeyS: toward(dy), KeyA: toward(-dx), KeyD: toward(dx), ShiftLeft: distance > RUN_DISTANCE });
  });
  trackDrag($("pad-turn"), (dx) => setKeys({ ArrowLeft: dx < -DEAD_ZONE, ArrowRight: dx > DEAD_ZONE }));

  const fire = $("fire");
  const setFire = (down) => { fire.classList.toggle("on", down); setKeys({ Space: down }); };
  fire.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    fire.setPointerCapture(event.pointerId);
    setFire(true);
  });
  fire.addEventListener("pointerup", () => setFire(false));
  fire.addEventListener("pointercancel", () => setFire(false));
}

function gridStep(dpr) {
  for (let step = 1; step <= 16; step++) if (Math.abs(step * dpr - Math.round(step * dpr)) < 1e-6) return step;
  return 1;
}

function fitPicture() { // whole device pixels per game pixel, on the device-pixel grid: no uneven rows or columns
  const area = $("screen"), picture = $("frame");
  const fit = () => {
    const dpr = devicePixelRatio, box = area.getBoundingClientRect();
    const scale = Math.max(1, Math.floor(Math.min((box.width * dpr) / WIDTH, (box.height * dpr) / HEIGHT)));
    const width = (WIDTH * scale) / dpr, height = (HEIGHT * scale) / dpr, step = gridStep(dpr);
    const snap = (offset, origin) => Math.round((origin + offset) / step) * step - origin;
    Object.assign(picture.style, {
      width: `${width}px`,
      height: `${height}px`,
      marginLeft: `${snap((box.width - width) / 2, box.left + scrollX)}px`,
      marginTop: `${snap((box.height - height) / 2, box.top + scrollY)}px`,
    });
  };
  new ResizeObserver(fit).observe(area);
  addEventListener("resize", fit);
  const refitOnZoom = () => matchMedia(`(resolution: ${devicePixelRatio}dppx)`)
    .addEventListener("change", () => { fit(); refitOnZoom(); }, { once: true });
  refitOnZoom();
}

function showMessage(ui, title, text) {
  ui.title.textContent = title;
  ui.text.textContent = text;
  ui.text.hidden = !text;
  ui.overlay.hidden = false;
}

async function main() {
  const ui = { overlay: $("overlay"), title: $("ov-title"), text: $("ov-text"), bar: $("ov-bar"), stats: $("stats") };
  const canvas = $("view").getContext("2d");
  const meta = await (await fetch(MODEL_DIR + "meta.json")).json();
  const webgpu = Boolean(await navigator.gpu?.requestAdapter().catch(() => null));

  let [model, startsBytes] = await download([MODEL_DIR + "frame.onnx", MODEL_DIR + "starts.bin"], (loaded, total) => {
    ui.bar.style.width = total ? `${(100 * loaded) / total}%` : "0";
    ui.text.textContent = `model weights: ${megabytes(loaded)} of ${megabytes(total)} MB`;
  });
  const starts = readStarts(startsBytes, meta);

  showMessage(ui, "Starting…", `compiling the network for ${webgpu ? "your GPU" : "the CPU: no WebGPU, slow"}`);
  const world = await createWorld(model, meta, webgpu, starts[0]);
  model = null; // the session keeps its own copy; iOS Safari closes tabs that take too much memory

  const app = { world, meta, starts, canvas, ui, webgpu };
  return params.has("selftest") ? selfTest(app) : play(app);
}

async function play({ world, meta, starts, canvas, ui, webgpu }) {
  const screen = $("screen"), levelSelect = $("level");
  for (const level of new Set(starts.map((s) => s.level))) levelSelect.add(new Option(level, level));
  const backend = webgpu ? "WebGPU" : "CPU", frameTimes = [];
  let start = null, playing = false, running = false, restarting = false, loop = Promise.resolve();

  async function newStart() {
    const pool = starts.filter((s) => !levelSelect.value || s.level === levelSelect.value);
    start = pool[Math.floor(Math.random() * pool.length)];
    await world.reset(start);
    canvas.clearRect(0, 0, WIDTH, HEIGHT);
  }

  async function run() {
    running = true;
    const frameTime = 1000 / meta.fps;
    while (playing && !restarting) {
      const t0 = performance.now();
      await world.frame(readAction(meta));
      await world.draw(canvas);
      const elapsed = performance.now() - t0;
      frameTimes.push(elapsed);
      if (frameTimes.length > 60) frameTimes.shift();
      const ms = median(frameTimes);
      ui.stats.textContent = `${backend} · ${ms.toFixed(0)} ms · ${(1000 / ms).toFixed(0)} fps · ` +
                             `${start.level} · frame ${world.frameNumber}`;
      await sleep(Math.max(0, frameTime - elapsed)); // always yield: on the CPU a frame takes longer than frameTime
    }
    running = false;
  }

  function resume() {
    if (playing) return;
    playing = true;
    ui.overlay.hidden = true;
    screen.focus();
    if (!running && !restarting) loop = run();
  }

  function pause(reason) {
    if (!playing) return;
    playing = false;
    held.clear();
    showMessage(ui, reason);
  }

  async function restart() {
    if (restarting) return;
    restarting = true;
    await loop;
    await newStart();
    restarting = false;
    if (playing) loop = run();
  }

  await newStart();
  ui.bar.parentElement.hidden = true;
  showMessage(ui, TOUCH ? "Tap to play" : "Click to play", webgpu ? "" : "No WebGPU: running on the CPU, slow");

  screen.addEventListener("click", resume);
  $("restart").addEventListener("click", restart);
  levelSelect.addEventListener("change", restart);
  addEventListener("blur", () => pause("Paused"));
  addEventListener("keyup", (event) => held.delete(event.code));
  addEventListener("keydown", (event) => {
    if (!playing || !GAME_KEYS.has(event.code)) return;
    event.preventDefault();
    if (event.code === "Escape") pause("Paused");
    else if (event.code === "KeyR") { if (!event.repeat) restart(); }
    else held.add(event.code);
  });
  if (TOUCH) setupTouchControls();
}

async function selfTest({ world, meta, starts, canvas, ui, webgpu }) {
  const dir = MODEL_DIR + "selftest/";
  const info = await (await fetch(dir + "info.json")).json();
  const files = await download(["ctx_noise", "noise", "acts", "ref"].map((name) => `${dir}${name}.bin`));
  const [contextNoise, noise, actions, reference] = files.map(asFloats);
  const n = world.latentSize, diffs = [], times = [];

  ui.title.textContent = "Self test";
  await world.reset(starts[info.start], contextNoise);
  for (let i = 0; i < info.frames; i++) {
    const t0 = performance.now();
    await world.frame(actions.subarray(i * ACTION_SIZE, (i + 1) * ACTION_SIZE),
                      { x: noise.subarray(2 * i * n, (2 * i + 1) * n), z: noise.subarray((2 * i + 1) * n, (2 * i + 2) * n) });
    const latent = await world.readLatent();
    await world.draw(canvas);
    times.push(performance.now() - t0);
    const expected = reference.subarray(i * n, (i + 1) * n);
    diffs.push(latent.reduce((max, v, j) => Math.max(max, Math.abs(v - expected[j])), 0));
    ui.text.textContent = `frame ${i + 1} of ${info.frames}: max difference to PyTorch ${diffs[i].toExponential(1)}`;
  }

  const ms = median(times.slice(5)), last = info.frames - 1;
  const path = world instanceof CapturedWorld ? "captured" : "arrays";
  ui.title.textContent = `Self test: ${webgpu ? "WebGPU" : "CPU"} (${path}), ${(1000 / ms).toFixed(0)} fps possible ` +
                         `(${ms.toFixed(0)} ms/frame)`;
  ui.text.textContent = `difference to PyTorch: frame 1 ${diffs[0].toExponential(1)}, ` +
                        `frame ${info.frames} ${diffs[last].toExponential(1)}`;
  globalThis.selftest = { webgpu, path, fallback: world.fallbackReason, model: meta.model, frames: info.frames,
                          ms_median: ms, diff: diffs, agent: navigator.userAgent };
}

fitPicture();
main().catch((error) => {
  showMessage({ overlay: $("overlay"), title: $("ov-title"), text: $("ov-text") }, "Could not start", String(error.message || error));
  console.error(error);
});
