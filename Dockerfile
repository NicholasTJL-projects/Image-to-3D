# Image-to-3D web app (CPU). Build:  docker build -t image-to-3d .
#                              Run:    docker run -p 8000:8000 image-to-3d
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    IMAGE_TO_3D_CACHE=/models \
    IMAGE_TO_3D_JOBS=/data/jobs \
    U2NET_HOME=/models/u2net \
    OMP_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY image_to_3d ./image_to_3d
COPY configs ./configs

# CPU-only torch keeps the image small; the default PyPI wheel pulls in CUDA.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install ".[web]"

# Fetch model weights at build time so the first request is fast and the container runs offline.
RUN python -c "from image_to_3d.depth import midas_weights_path; midas_weights_path()" \
    && python -c "from rembg import new_session; new_session('u2net')"

VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -sf http://localhost:8000/api/health || exit 1
CMD ["uvicorn", "image_to_3d.web.server:app", "--host", "0.0.0.0", "--port", "8000"]
