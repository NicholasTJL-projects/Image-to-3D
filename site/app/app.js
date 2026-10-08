// Image-to-3D frontend: upload -> poll job -> show mesh (Three.js), splat (GaussianSplats3D) and depth.
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { GLTFLoader } from './vendor/GLTFLoader.js';

const $ = (id) => document.getElementById(id);
// API base: same origin by default; a static deployment (e.g. Vercel) can point at a hosted backend
// by defining window.IMAGE_TO_3D_API before this script runs (see config.js).
const API = (window.IMAGE_TO_3D_API || '.').replace(/\/$/, '');
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
fetch(`${API}/api/health`).then((r) => r.json()).then((h) => {
  const sel = $('depth-backend');
  sel.innerHTML = '';
  const names = { depth_anything: 'Depth Anything V2 small', midas_small: 'MiDaS v2.1 small', inflate: 'none (inflate silhouette)' };
  for (const b of h.depth_backends) {
    const o = document.createElement('option');
    o.value = b; o.textContent = names[b] || b;
    if (b === h.default_depth_backend) o.selected = true;
    sel.appendChild(o);
  }
  if (h.multiview) $('multiview').hidden = false;
}).catch(() => {
  const n = $('no-backend');
  if (n) n.hidden = false;
  goBtn.disabled = true;
  drop.classList.add('disabled');
});

// ----------------------------------------------------------------- multi-view: camera / video / photos
const mvFiles = $('mv-files'), mvVideo = $('mv-video-file'), mvGo = $('mv-go');
let mvMode = 'camera';
let camShots = [];          // Blobs captured from the device camera
let camStream = null, camTimer = null;

function mvSource() {
  if (mvMode === 'camera') return { photos: camShots };
  if (mvMode === 'video') return { video: mvVideo.files[0] || null };
  return { photos: [...mvFiles.files] };
}
function mvRefresh() {
  const src = mvSource();
  mvGo.disabled = src.video ? false : (src.photos || []).length < 3;
}
document.querySelectorAll('.seg-btn').forEach((b) => b.addEventListener('click', () => {
  mvMode = b.dataset.mode;
  document.querySelectorAll('.seg-btn').forEach((x) => x.classList.toggle('active', x === b));
  document.querySelectorAll('.mv-mode').forEach((el) => { el.hidden = el.id !== `mv-${mvMode}`; });
  if (mvMode !== 'camera') stopCamera();
  mvRefresh();
}));
mvFiles.addEventListener('change', () => {
  const n = mvFiles.files.length;
  $('mv-label').textContent = n ? `${n} photo${n === 1 ? '' : 's'} selected` : 'Choose photos';
  mvRefresh();
});
mvVideo.addEventListener('change', () => {
  const f = mvVideo.files[0];
  $('mv-video-label').textContent = f ? `${f.name} (${(f.size / 1048576).toFixed(1)} MB)` : 'Choose a video';
  mvRefresh();
});

// guided capture: full-resolution stills from the camera stream, one every 600 ms
const camPreview = $('cam-preview'), camCanvas = $('cam-canvas');
async function startCamera() {
  try {
    camStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 2560 }, height: { ideal: 2560 } }, audio: false,
    });
  } catch (err) {
    showError(`Camera unavailable: ${err.message || err}. Use the Video or Photos option instead.`);
    return;
  }
  camPreview.srcObject = camStream;
  camPreview.hidden = false;
  await camPreview.play().catch(() => {});
  $('cam-start').textContent = 'Stop camera';
  $('cam-shoot').disabled = false;
  $('cam-hint').textContent = 'Press Start capturing, then walk slowly around the object.';
}
function stopCamera() {
  stopShooting();
  if (camStream) { camStream.getTracks().forEach((t) => t.stop()); camStream = null; }
  camPreview.srcObject = null; camPreview.hidden = true;
  $('cam-start').textContent = 'Start camera';
  $('cam-shoot').disabled = true;
}
function grabShot() {
  const w = camPreview.videoWidth, h = camPreview.videoHeight;
  if (!w || !h) return;
  camCanvas.width = w; camCanvas.height = h;
  camCanvas.getContext('2d').drawImage(camPreview, 0, 0, w, h);
  camCanvas.toBlob((blob) => {
    if (!blob) return;
    camShots.push(blob);
    $('cam-count').textContent = `${camShots.length} shot${camShots.length === 1 ? '' : 's'}`;
    $('cam-hint').textContent = camShots.length < 40 ? 'Keep going: aim for 40 to 80 shots covering every side.'
      : camShots.length < 80 ? 'Good coverage. A second loop higher or lower helps the top and sides.' : 'Plenty. Press Stop capturing.';
    const wrap = document.querySelector('.cam-wrap'); wrap.classList.remove('flash'); void wrap.offsetWidth; wrap.classList.add('flash');
    $('cam-reset').disabled = false;
    mvRefresh();
  }, 'image/jpeg', 0.92);
}
function startShooting() { if (camTimer) return; camTimer = setInterval(grabShot, 600); $('cam-shoot').textContent = 'Stop capturing'; }
function stopShooting() { if (camTimer) { clearInterval(camTimer); camTimer = null; } $('cam-shoot').textContent = 'Start capturing'; }
$('cam-start').addEventListener('click', () => (camStream ? stopCamera() : startCamera()));
$('cam-shoot').addEventListener('click', () => (camTimer ? stopShooting() : startShooting()));
$('cam-reset').addEventListener('click', () => { camShots = []; $('cam-count').textContent = '0 shots'; $('cam-reset').disabled = true; mvRefresh(); });

mvGo.addEventListener('click', async () => {
  hideError();
  stopShooting();
  resultEl.hidden = true;
  progress.hidden = false;
  setProgress(0.02, 'uploading');
  mvGo.disabled = true;
  const src = mvSource();
  const fd = new FormData();
  if (src.video) fd.append('video', src.video, src.video.name);
  else src.photos.forEach((f, i) => fd.append('images', f, f.name || `shot_${String(i).padStart(4, '0')}.jpg`));
  fd.append('object_only', $('mv-object-only').checked);
  try {
    const res = await fetch(`${API}/api/jobs/multiview`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    currentJob = await res.json();
    poll(currentJob.id);
  } catch (err) {
    showError(`Upload failed: ${err.message}`);
    progress.hidden = true;
    mvRefresh();
  }
});

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
    const res = await fetch(`${API}/api/jobs`, { method: 'POST', body: fd });
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
    const res = await fetch(`${API}/api/jobs/${id}`);
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

function fileUrl(job, key) { return `${API}/api/jobs/${job.id}/files/${job.files[key]}`; }

function onDone(job) {
  currentJob = job;
  setProgress(1, 'done');
  setTimeout(() => { progress.hidden = true; }, 600);
  goBtn.disabled = !selectedFile;
  mvRefresh();
  const m = job.meta || {};
  const cov = m.coverage;
  $('coverage').hidden = !(job.kind === 'multiview' && cov);
  if (job.kind === 'multiview' && cov) {
    $('coverage').className = `coverage ${cov.verdict}`;
    $('cov-title').textContent = `Coverage: ${cov.verdict} · ${cov.cameras_posed} of ${cov.photos_submitted} views posed · ${Math.round(cov.azimuth_coverage_deg)}° around the object`;
    $('cov-advice').innerHTML = (cov.advice || []).map((a) => `<li>${a}</li>`).join('');
  }
  if (job.kind === 'multiview') {
    $('st-verts').textContent = fmt(m.cameras);
    $('st-faces').textContent = fmt(m.points);
    $('st-gauss').textContent = fmt(m.gaussians);
    $('st-time').textContent = `${Math.round(job.elapsed || 0)} s`;
    document.querySelector('.stats div:nth-child(1) dt').textContent = 'Cameras';
    document.querySelector('.stats div:nth-child(2) dt').textContent = 'SfM points';
    $('dl-glb').hidden = true;
    $('dl-depth').hidden = true;
    $('dl-ply').href = fileUrl(job, 'splat');
    resultEl.hidden = false;
    placeholder.style.display = 'none';
    document.querySelector('.tab[data-view=splat]').click();
    return;
  }
  document.querySelector('.stats div:nth-child(1) dt').textContent = 'Vertices';
  document.querySelector('.stats div:nth-child(2) dt').textContent = 'Triangles';
  $('dl-glb').hidden = false;
  $('dl-depth').hidden = false;
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
    if (splatViewer) {
      const v = splatViewer; splatViewer = null;
      try { v.stop?.(); } catch (e) { /* not started */ }
      try { await v.dispose(); } catch (e) { /* the library throws if its canvas was already detached */ }
      $('splat-view').replaceChildren();
    }
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
window.imageTo3D = { get splatViewer() { return splatViewer; }, get job() { return currentJob; }, get activeView() { return activeView; }, get camShots() { return camShots.length; } };
