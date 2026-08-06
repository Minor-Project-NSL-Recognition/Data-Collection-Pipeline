# Build from the REPO ROOT, not from server/:
#     docker build -f server/Dockerfile -t nsl-server .
#     docker run --rm -p 8000:8000 nsl-server
#
# results/model.tflite must exist before building (scripts/export_tflite.py).

FROM python:3.11-slim

# mediapipe and opencv need these even in headless form
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

# The package layout the server imports: nslr/ is the shared preprocessing core,
# and it must be the SAME code the model was trained with -- copy it, don't
# vendor a second copy.
COPY nslr/ nslr/
COPY server/ server/
# model.tflite + model_meta.json + ood.json. results/*.tflite is gitignored, so
# this fails loudly at build time if the export step was skipped.
COPY results/model.tflite results/model_meta.json results/ood.json results/

ENV PYTHONUNBUFFERED=1 \
    NSL_MAX_SESSIONS=4

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"

# One worker: each WebSocket session pins its own MediaPipe graph, so scale with
# NSL_MAX_SESSIONS and container replicas rather than in-process workers.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
