"""FastAPI backend for the Image-to-3D web app.

    uvicorn image_to_3d.web.server:app --host 0.0.0.0 --port 8000

Endpoints
---------
GET  /                         the single-page frontend
GET  /api/health               which backends are available on this server
POST /api/jobs                 multipart upload: ``image`` (+ options) -> job
POST /api/jobs/multiview       multipart upload: many ``images`` of one object -> splat (needs COLMAP)
GET  /api/jobs/{id}            job status / progress / result file list
GET  /api/jobs/{id}/files/{n}  result files (model.glb, splat.ply, depth.png, ...)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import depth as depth_mod
from ..colmap import colmap_available
from ..single_image import SingleImageConfig, reconstruct, segmenter_available
from .jobs import Job, JobStore

STATIC_DIR = Path(__file__).resolve().parent / "static"
JOBS_DIR = Path(os.environ.get("IMAGE_TO_3D_JOBS", "jobs"))
MAX_UPLOAD_MB = float(os.environ.get("IMAGE_TO_3D_MAX_UPLOAD_MB", "25"))
MAX_SIDE = int(os.environ.get("IMAGE_TO_3D_MAX_SIDE", "512"))
DEVICE = os.environ.get("IMAGE_TO_3D_DEVICE", "cpu")
ENABLE_MULTIVIEW = os.environ.get("IMAGE_TO_3D_MULTIVIEW", "auto")  # auto | 0 | 1
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff", "image/heic", "application/octet-stream"}

app = FastAPI(title="Image-to-3D", version="0.2.0", docs_url="/api/docs", openapi_url="/api/openapi.json")
store = JobStore(JOBS_DIR, workers=int(os.environ.get("IMAGE_TO_3D_WORKERS", "1")))


def multiview_enabled() -> bool:
    if ENABLE_MULTIVIEW == "0":
        return False
    if ENABLE_MULTIVIEW == "1":
        return True
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return colmap_available()


@app.get("/api/health")
def health() -> dict:
    try:
        import torch
        torch_ok, cuda = True, torch.cuda.is_available()
    except ImportError:
        torch_ok, cuda = False, False
    return {
        "ok": True,
        "depth_backends": [b for b in depth_mod.BACKENDS if depth_mod.backend_available(b)],
        "default_depth_backend": depth_mod.default_backend(),
        "segmenters": [s for s in ("rembg", "grabcut") if segmenter_available(s)],
        "multiview": multiview_enabled(),
        "torch": torch_ok,
        "cuda": cuda,
        "max_side": MAX_SIDE,
        "max_upload_mb": MAX_UPLOAD_MB,
    }


async def _read_upload(upload: UploadFile) -> bytes:
    data = await upload.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file larger than {MAX_UPLOAD_MB:g} MB")
    if not data:
        raise HTTPException(400, "empty upload")
    return data


@app.post("/api/jobs")
async def create_job(
    image: UploadFile = File(...),
    remove_background: bool = Form(True),
    depth_backend: str = Form("auto"),
    relief: float = Form(0.35),
    mirror_back: bool = Form(True),
    fov_deg: float = Form(50.0),
    max_side: int = Form(0),
) -> JSONResponse:
    data = await _read_upload(image)
    if depth_backend == "auto":
        depth_backend = depth_mod.default_backend()
    if depth_backend not in depth_mod.BACKENDS:
        raise HTTPException(400, f"unknown depth backend {depth_backend}")
    cfg = SingleImageConfig(
        depth_backend=depth_backend, remove_background=remove_background,
        relief=float(min(max(relief, 0.02), 2.0)), mirror_back=mirror_back,
        fov_deg=float(min(max(fov_deg, 20.0), 120.0)),
        max_side=int(min(max_side or MAX_SIDE, MAX_SIDE)), device=DEVICE,
    )
    job = store.create("single", options=cfg.__dict__.copy())
    src = store.dir(job.id) / ("upload" + Path(image.filename or "image").suffix.lower()[:8])
    src.write_bytes(data)

    def run(job: Job, report) -> dict:
        result = reconstruct(src, cfg, progress=report)
        report("writing files", 0.9)
        paths = result.save(store.dir(job.id), cfg=cfg)
        meta = {"vertices": len(result.vertices), "faces": len(result.faces), "gaussians": len(result.cloud),
                "timings": result.timings, "resolution": [int(result.rgb.shape[1]), int(result.rgb.shape[0])]}
        return {"files": {k: p.name for k, p in paths.items()}, "meta": meta}

    store.submit(job, run)
    return JSONResponse(job.to_dict(), status_code=202)


@app.post("/api/jobs/multiview")
async def create_multiview_job(
    images: list[UploadFile] = File(...),
    iterations: int = Form(3000),
    downscale: int = Form(4),
    object_only: bool = Form(True),
) -> JSONResponse:
    """Photos of one object from many angles -> camera poses (COLMAP) -> Gaussian Splatting scene."""
    if not multiview_enabled():
        raise HTTPException(501, "multi-view reconstruction needs COLMAP and PyTorch on the server")
    if len(images) < 3:
        raise HTTPException(400, "upload at least 3 photos taken from different angles (20+ recommended)")
    if len(images) > 300:
        raise HTTPException(400, "at most 300 photos per job")
    job = store.create("multiview", options={"iterations": iterations, "downscale": downscale, "photos": len(images),
                                             "object_only": object_only})
    ws = store.dir(job.id)
    src_dir = ws / "upload"
    src_dir.mkdir(exist_ok=True)
    for i, up in enumerate(images):
        suffix = Path(up.filename or ".jpg").suffix.lower()[:8] or ".jpg"
        (src_dir / f"img_{i:04d}{suffix}").write_bytes(await _read_upload(up))

    def run(job: Job, report) -> dict:
        from ..capture import CaptureConfig, capture
        from ..colmap import run_sfm
        from ..pipeline import init_gaussians
        from ..train import TrainConfig, train

        report("preparing photos", 0.05)
        capture(str(src_dir), ws / "images", CaptureConfig(max_side=1280, min_blur=0.0))
        report("recovering camera poses (COLMAP)", 0.15)
        scene = run_sfm(ws, matcher="exhaustive")  # unordered photos: match every pair
        masks = None
        if object_only:
            from ..masks import compute_masks

            masks = compute_masks(scene, ws / "masks", progress=lambda msg, f: report(msg, 0.3 + 0.08 * f))
        report("initialising gaussians", 0.4)
        cloud = init_gaussians(scene, ws, masks=masks)
        total = int(iterations)

        def log(msg: str) -> None:
            if msg.startswith("[") and "/" in msg.split("]")[0]:
                try:
                    step = int(msg[1:].split("/")[0])
                    report(f"training gaussians ({step}/{total})", 0.45 + 0.5 * step / total)
                    return
                except ValueError:
                    pass
            report(msg[:80], job.progress)

        cfg = TrainConfig(iterations=total, downscale=int(downscale), log_every=100, eval_every=0,
                          object_only=bool(masks), densify_from=100, densify_until=max(200, int(total * 0.65)),
                          densify_interval=50, densify_grad_threshold=1e-4, opacity_reset_interval=0)
        out = train(scene, cloud, cfg, out_dir=ws / "output", log=log, masks=masks)
        shutil.copy(ws / "output" / "point_cloud.ply", ws / "splat.ply")
        return {"files": {"splat": "splat.ply"},
                "meta": {"gaussians": len(out), "cameras": len(scene), "points": len(scene.points_xyz),
                         "photos": len(images), "object_only": bool(masks)}}

    store.submit(job, run)
    return JSONResponse(job.to_dict(), status_code=202)


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return store.list()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job.to_dict()


MEDIA_TYPES = {".glb": "model/gltf-binary", ".ply": "application/octet-stream", ".png": "image/png",
               ".json": "application/json", ".jpg": "image/jpeg"}


@app.get("/api/jobs/{job_id}/files/{name}")
def get_file(job_id: str, name: str) -> FileResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if "/" in name or name.startswith(".") or name not in set(job.files.values()):
        raise HTTPException(404, "no such file")
    path = store.dir(job_id) / name
    if not path.exists():
        raise HTTPException(404, "file missing")
    return FileResponse(path, media_type=MEDIA_TYPES.get(path.suffix, "application/octet-stream"),
                        filename=f"image-to-3d-{job_id}-{name}")


# frontend (mounted last so /api/* wins)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def main() -> None:  # `image-to-3d-web` entry point
    import uvicorn

    uvicorn.run("image_to_3d.web.server:app", host=os.environ.get("HOST", "0.0.0.0"),
                port=int(os.environ.get("PORT", "8000")), reload=False)
