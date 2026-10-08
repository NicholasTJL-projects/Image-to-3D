import io
import time

import numpy as np
import pytest
from PIL import Image

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_TO_3D_JOBS", str(tmp_path / "jobs"))
    import importlib

    from image_to_3d.web import server

    importlib.reload(server)
    from fastapi.testclient import TestClient

    with TestClient(server.app) as c:
        yield c


def png_bytes(w=96, h=72):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (20, 40, 80)
    yy, xx = np.mgrid[:h, :w]
    img[((yy - h / 2) ** 2 + (xx - w / 2) ** 2) < 25**2] = (230, 120, 50)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def wait_done(client, job_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.1)
    raise TimeoutError


def test_health_and_frontend(client):
    h = client.get("/api/health").json()
    assert h["ok"] and "inflate" in h["depth_backends"]
    assert client.get("/").status_code == 200 and "Kunami Labs" in client.get("/").text
    assert client.get("/vendor/three.module.js").status_code == 200


def test_single_image_job_roundtrip(client):
    r = client.post("/api/jobs", files={"image": ("disc.png", png_bytes(), "image/png")},
                    data={"depth_backend": "inflate", "relief": "0.3", "remove_background": "true"})
    assert r.status_code == 202, r.text
    job = wait_done(client, r.json()["id"])
    assert job["status"] == "done", job.get("error")
    assert job["meta"]["faces"] > 0 and set(job["files"]) >= {"glb", "splat", "depth", "mask", "meta"}
    glb = client.get(f"/api/jobs/{job['id']}/files/{job['files']['glb']}")
    assert glb.status_code == 200 and glb.content[:4] == b"glTF"
    assert client.get(f"/api/jobs/{job['id']}/files/job.json").status_code == 404  # not an advertised file
    assert client.get(f"/api/jobs/{job['id']}/files/..%2Fjob.json").status_code == 404
    assert any(j["id"] == job["id"] for j in client.get("/api/jobs").json())


def test_bad_requests(client):
    assert client.get("/api/jobs/nope").status_code == 404
    r = client.post("/api/jobs", files={"image": ("x.png", b"", "image/png")})
    assert r.status_code == 400
    r = client.post("/api/jobs", files={"image": ("x.png", png_bytes(), "image/png")}, data={"depth_backend": "bogus"})
    assert r.status_code == 400
    r = client.post("/api/jobs", files={"image": ("x.png", b"not an image", "image/png")}, data={"depth_backend": "inflate"})
    job = wait_done(client, r.json()["id"])
    assert job["status"] == "error" and job["error"]


def test_multiview_endpoint_without_colmap(client, monkeypatch):
    from image_to_3d.web import server

    monkeypatch.setattr(server, "ENABLE_MULTIVIEW", "0")
    files = [("images", (f"p{i}.png", png_bytes(), "image/png")) for i in range(3)]
    assert client.post("/api/jobs/multiview", files=files).status_code == 501
    assert client.get("/api/health").json()["multiview"] is False
    monkeypatch.setattr(server, "ENABLE_MULTIVIEW", "1")
    assert client.post("/api/jobs/multiview", files=files[:2]).status_code == 400
