// Image-to-3D frontend: upload -> poll job -> show mesh (Three.js), splat (GaussianSplats3D) and depth.
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { GLTFLoader } from './vendor/GLTFLoader.js';

const $ = (id) => document.getElementById(id);
const drop = $('drop'), fileInput = $('file'), thumb = $('thumb'), form = $('options'), goBtn = $('go');
const progress = $('progress'), barFill = $('bar-fill'), stageEl = $('stage'), errorEl = $('error');
const resultEl = $('result'), placeholder = $('placeholder');

let selectedFile = null;
let activeView = 'mesh';
let currentJob = null;
let splatViewer = null;
let splatModule = null;

// ----------------------------------------------------------------- upload UI
function setFile(file) {
  if (!file || !file.type.startsWith('image/')) { showError('Please choose an image file.'); return; }
  selectedFile = file;
  thumb.src = URL.createObjectURL(file);
  thumb.hidden = false;
  drop.classList.add('has-image');
  goBtn.disabled = false;
  hideError();
}
drop.addEventListener('click', () => fileInput.click());
drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
fileInput.addEventListener('change', () => setFile(fileInput.files[0]));
['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => setFile(e.dataTransfer.files[0]));
window.addEventListener('paste', (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
  if (item) setFile(item.getAsFile());
});
form.relief.addEventListener('input', () => { $('relief-out').value = Number(form.relief.value).toFixed(2); });

function showError(msg) { errorEl.textContent = msg; errorEl.hidden = false; }
function hideError() { errorEl.hidden = true; }

// ----------------------------------------------------------------- server capabilities
fetch('./api/health').then((r) => r.json()).then((h) => {
  const sel = $('depth-backend');
  sel.innerHTML = '';
  const names = { midas_small: 'MiDaS small (default)', depth_anything: 'Depth Anything V2', inflate: 'none (inflate silhouette)' };
  for (const b of h.depth_backends) {
    const o = document.createElement('option');
    o.value = b; o.textContent = names[b] || b;
    if (b === h.default_depth_backend) o.selected = true;
    sel.appendChild(o);
  }
}).catch(() => {});

// ----------------------------------------------------------------- job flow
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!selectedFile) return;
  hideError();
  resultEl.hidden = true;
  progress.hidden = false;
  setProgress(0.02, 'uploading');
  goBtn.disabled = true;

  const fd = new FormData();
  fd.append('image', selectedFile, selectedFile.name);
  fd.append('remove_background', form.remove_background.checked);
  fd.append('mirror_back', form.mirror_back.checked);
  fd.append('relief', form.relief.value);
  fd.append('depth_backend', form.depth_backend.value || 'auto');
  try {
    const res = await fetch('./api/jobs', { method: 'POST', body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    currentJob = await res.json();
    poll(currentJob.id);
  } catch (err) {
    showError(`Upload failed: ${err.message}`);
    progress.hidden = true;
    goBtn.disabled = false;
  }
});

function setProgress(frac, stage) {
  barFill.style.width = `${Math.round(frac * 100)}%`;
  stageEl.textContent = stage;
}

async function poll(id) {
  try {
    const res = await fetch(`./api/jobs/${id}`);
    if (!res.ok) throw new Error(res.statusText);
    const job = await res.json();
    currentJob = job;
    if (job.status === 'done') { onDone(job); return; }
    if (job.status === 'error') { showError(job.error || 'reconstruction failed'); progress.hidden = true; goBtn.disabled = false; return; }
    setProgress(Math.max(0.03, job.progress), job.stage || job.status);
    setTimeout(() => poll(id), 700);
  } catch (err) {
    showError(`Lost contact with the server: ${err.message}`);
    progress.hidden = true;
    goBtn.disabled = false;
  }
}

function fileUrl(job, key) { return `./api/jobs/${job.id}/files/${job.files[key]}`; }

function onDone(job) {
  currentJob = job;
  setProgress(1, 'done');
  setTimeout(() => { progress.hidden = true; }, 600);
  goBtn.disabled = false;
  const m = job.meta || {};
  $('st-verts').textContent = fmt(m.vertices);
  $('st-faces').textContent = fmt(m.faces);
  $('st-gauss').textContent = fmt(m.gaussians);
  $('st-time').textContent = `${(m.timings?.total ?? job.elapsed ?? 0).toFixed(1)} s`;
  $('dl-glb').href = fileUrl(job, 'glb');
  $('dl-ply').href = fileUrl(job, 'splat');
  $('dl-depth').href = fileUrl(job, 'depth');
  resultEl.hidden = false;
  placeholder.style.display = 'none';
  $('depth-img').src = fileUrl(job, 'depth');
  $('depth-img').hidden = false;
  loadMesh(fileUrl(job, 'glb'));
  if (activeView === 'splat') loadSplat(fileUrl(job, 'splat'));
  else splatLoadedFor = null;
}
const fmt = (n) => (n == null ? '–' : Number(n).toLocaleString());

// ----------------------------------------------------------------- mesh viewer (Three.js)
const meshView = $('mesh-view');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
meshView.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
camera.position.set(0, 0.15, 2.4);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 1.6;
scene.add(new THREE.HemisphereLight(0xffffff, 0x334466, 1.1));
const key = new THREE.DirectionalLight(0xffffff, 1.4); key.position.set(2, 3, 4); scene.add(key);
const fill = new THREE.DirectionalLight(0xbfd4ff, 0.5); fill.position.set(-3, -1, -2); scene.add(fill);
let model = null;

function resize() {
  const w = meshView.clientWidth || 1, h = meshView.clientHeight || 1;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(meshView);
resize();
(function animate() {
  requestAnimationFrame(animate);
  if (activeView !== 'mesh') return;
  controls.update();
  renderer.render(scene, camera);
})();

function loadMesh(url) {
  new GLTFLoader().load(url, (gltf) => {
    if (model) scene.remove(model);
    model = gltf.scene;
    model.traverse((o) => {
      if (o.isMesh) {
        o.material.side = THREE.DoubleSide;
        o.material.wireframe = $('wire').checked;
        if (o.material.map) o.material.map.colorSpace = THREE.SRGBColorSpace;
      }
    });
    // centre and frame
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3()).length();
    const centre = box.getCenter(new THREE.Vector3());
    model.position.sub(centre);
    scene.add(model);
    camera.position.set(0, size * 0.15, size * 1.9);
    controls.target.set(0, 0, 0);
    controls.update();
  }, undefined, (err) => showError(`Could not load the mesh: ${err.message || err}`));
}
$('wire').addEventListener('change', () => model?.traverse((o) => { if (o.isMesh) o.material.wireframe = $('wire').checked; }));
$('autorotate').addEventListener('change', () => { controls.autoRotate = $('autorotate').checked; if (splatViewer) splatViewer.controls && (splatViewer.controls.autoRotate = $('autorotate').checked); });

// ----------------------------------------------------------------- splat viewer (GaussianSplats3D)
let splatLoadedFor = null;
async function loadSplat(url) {
  if (splatLoadedFor === url) return;
  splatLoadedFor = url;
  try {
    splatModule = splatModule || await import('./vendor/gaussian-splats-3d.module.js');
    if (splatViewer) { await splatViewer.dispose(); splatViewer = null; $('splat-view').innerHTML = ''; }
    splatViewer = new splatModule.Viewer({
      rootElement: $('splat-view'),
      cameraUp: [0, 1, 0],
      initialCameraPosition: [0, 0.15, 2.4],
      initialCameraLookAt: [0, 0, 0],
      sharedMemoryForWorkers: false,   // no cross-origin isolation headers needed
      gpuAcceleratedSort: false,
      dynamicScene: false,
      antialiased: true,
      sphericalHarmonicsDegree: 0,
      useBuiltInControls: true,
    });
    await splatViewer.addSplatScene(url, { format: splatModule.SceneFormat.Ply, splatAlphaRemovalThreshold: 3, showLoadingUI: true, progressiveLoad: false });
    splatViewer.start();
    if (splatViewer.controls) { splatViewer.controls.autoRotate = $('autorotate').checked; splatViewer.controls.autoRotateSpeed = 1.6; }
  } catch (err) {
    console.error(err);
    showError(`Splat viewer unavailable in this browser: ${err.message || err}. Download the PLY to view it elsewhere.`);
  }
}

// ----------------------------------------------------------------- tabs
document.querySelectorAll('.tab').forEach((tab) => tab.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t === tab));
  activeView = tab.dataset.view;
  document.querySelectorAll('.view').forEach((v) => v.classList.toggle('active', v.id === `${activeView}-view`));
  if (activeView === 'mesh') resize();
  if (activeView === 'splat' && currentJob?.status === 'done') loadSplat(fileUrl(currentJob, 'splat'));
}));

// debugging hook (used by the headless browser test)
window.imageTo3D = { get splatViewer() { return splatViewer; }, get job() { return currentJob; }, get activeView() { return activeView; } };
