# ---------------------------------------------------------
#  IKRAE-EdNet  |  Reproducible Research Environment
# ---------------------------------------------------------
FROM python:3.10-slim

LABEL maintainer="Aziz Abdelkarim <a.karim.linux@gmail.com>"
LABEL description="Docker image for IKRAE-EdNet: Semantic + Graph-based Adaptive Learning Framework"

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# --- System setup ---
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        openjdk-11-jre-headless \
        git \
        wget \
        unzip \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# --- Environment variables ---
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

# --- Copy repo ---
WORKDIR /app
COPY . /app

# --- Python dependencies ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Default command: run pipeline ---
CMD ["bash", "run_pipeline.sh"]
