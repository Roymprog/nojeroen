ARG BUILD_FROM=python:3.11-slim-bookworm
FROM ${BUILD_FROM}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for a cacheable layer.
COPY addon_nojeroen/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy the source tree (both whospeaks core and whospeaks_addon).
COPY src/ /app/src/
ENV PYTHONPATH=/app/src

COPY addon_nojeroen/run.sh /run.sh
RUN chmod +x /run.sh

CMD ["/run.sh"]
