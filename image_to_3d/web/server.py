"""FastAPI backend for the Image-to-3D web app.

    uvicorn image_to_3d.web.server:app --host 0.0.0.0 --port 8000

Endpoints
---------
GET  /                         the single-page frontend
GET  /api/health               which backends are available on this server
POST /api/jobs                 multipart upload: ``image`` (+ options) -> job
POST /api/jobs/multiview       multipart upload: ``video`` or several ``images`` -> job (needs COLMAP)
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
        "default_depth_backend": "midas_small" if depth_mod.backend_available("midas_small") else "inflate",
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
        depth_backend = "midas_small" if depth_mod.backend_available("midas_small") else "inflate"
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
    video: UploadFile | None = File(None),
    images: list[UploadFile] = File([]),
    iterations: int = Form(3000),
    downscale: int = Form(4),
) -> JSONResponse:
    if not multiview_enabled():
        raise HTTPException(501, "multi-view reconstruction needs COLMAP and PyTorch on the server")
    if video is None and not images:
        raise HTTPException(400, "upload a video or several images")
    job = store.create("multiview", options={"iterations": iterations, "downscale": downscale})
    ws = store.dir(job.id)
    src_dir = ws / "upload"
    src_dir.mkdir(exist_ok=True)
    source: Path
    if video is not None:
        source = src_dir / ("video" + Path(video.filename or "v.mp4").suffix.lower()[:8])
        source.write_bytes(await _read_upload(video))
    else:
        for i, up in enumerate(images):
            (src_dir / f"img_{i:04d}{Path(up.filename or '.jpg').suffix.lower()[:8]}").write_bytes(await _read_upload(up))
        source = src_dir

    def run(job: Job, report) -> dict:
        from ..capture import CaptureConfig, capture
        from ..colmap import run_sfm
        from ..pipeline import init_gaussians
        from ..train import TrainConfig, train

        report("extracting frames", 0.05)
        capture(str(source), ws / "images", CaptureConfig(every_nth=3, max_frames=200, max_side=1280, min_blur=40))
        report("structure from motion (COLMAP)", 0.15)
        scene = run_sfm(ws)
        report("initialising gaussians", 0.4)
        cloud = init_gaussians(scene, ws)
        report(f"training ({iterations} iterations)", 0.45)
        cfg = TrainConfig(iterations=int(iterations), downscale=int(downscale), log_every=500, eval_every=0)
        out = train(scene, cloud, cfg, out_dir=ws / "output", log=lambda msg: report(msg, 0.5))
        shutil.copy(ws / "output" / "point_cloud.ply", ws / "splat.ply")
        return {"files": {"splat": "splat.ply"},
                "meta": {"gaussians": len(out), "cameras": len(scene), "points": len(scene.points_xyz)}}

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
