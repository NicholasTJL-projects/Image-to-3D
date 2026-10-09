"""Tiny on-disk job queue for the web app: one worker thread, jobs persisted as JSON so a
restart does not lose results. Good enough for a single-node deployment; swap for Celery/RQ
if you need more."""

from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class Job:
    id: str
    kind: str                      # "single" | "multiview"
    status: str = "queued"         # queued | running | done | error
    stage: str = ""
    progress: float = 0.0
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    error: str | None = None
    files: dict = field(default_factory=dict)   # name -> relative filename
    meta: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["elapsed"] = round(((self.finished or time.time()) - (self.started or self.created)), 2)
        return d


class JobStore:
    def __init__(self, root: str | Path, workers: int = 1, max_age_hours: float = 24.0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job")
        self.max_age = max_age_hours * 3600
        self._load()

    # -- persistence -------------------------------------------------------
    def _path(self, job_id: str) -> Path:
        return self.root / job_id

    def _save(self, job: Job) -> None:
        (self._path(job.id) / "job.json").write_text(json.dumps(job.to_dict(), indent=1))

    def _load(self) -> None:
        for p in self.root.glob("*/job.json"):
            try:
                d = json.loads(p.read_text())
                d.pop("elapsed", None)
                job = Job(**d)
                if job.status in ("queued", "running"):  # interrupted by a restart
                    job.status, job.error = "error", "server restarted while the job was running"
                self.jobs[job.id] = job
            except Exception:
                continue

    # -- API ----------------------------------------------------------------
    def create(self, kind: str, options: dict | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, options=options or {})
        self._path(job.id).mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.jobs[job.id] = job
        self._save(job)
        return job

    def dir(self, job_id: str) -> Path:
        return self._path(job_id)

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def submit(self, job: Job, fn: Callable[[Job, Callable[[str, float], None]], dict]) -> None:
        """Run ``fn(job, report)`` in the worker pool; ``fn`` returns ``{"files": {...}, "meta": {...}}``."""

        def report(stage: str, progress: float) -> None:
            job.stage, job.progress = stage, float(progress)
            self._save(job)

        def run() -> None:
            job.status, job.started = "running", time.time()
            self._save(job)
            try:
                out = fn(job, report)
                job.files = out.get("files", {})
                job.meta = out.get("meta", {})
                job.status, job.progress, job.stage = "done", 1.0, "done"
            except Exception as e:  # surfaced to the UI
                job.status, job.error = "error", f"{type(e).__name__}: {e}"
                job.stage = "failed"
                traceback.print_exc()
            finally:
                job.finished = time.time()
                self._save(job)
                self.cleanup()

        self.pool.submit(run)

    def cleanup(self) -> None:
        """Delete jobs older than ``max_age``."""
        now = time.time()
        with self.lock:
            old = [j for j in self.jobs.values() if j.finished and now - j.finished > self.max_age]
            for j in old:
                self.jobs.pop(j.id, None)
                shutil.rmtree(self._path(j.id), ignore_errors=True)

    def list(self, limit: int = 20) -> list[dict]:
        return [j.to_dict() for j in sorted(self.jobs.values(), key=lambda j: -j.created)[:limit]]
