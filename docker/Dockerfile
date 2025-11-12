# ---------------------------------------------------------
#  IKRAE |  Reproducible Research Environment
# ---------------------------------------------------------
FROM python:3.10-slim

LABEL maintainer="Aziz Abdelkarim <a.karim.linux@gmail.com>"
LABEL description="Docker image for IKRAE-EdNet: Semantic + Graph-based Adaptive Learning Framework"

# --- System setup ---
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-11-jdk git wget unzip \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Environment variables ---
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

# --- Copy repo ---
WORKDIR /app
COPY . /app

# --- Python deps ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Ports & volumes (optional) ---
VOLUME ["/app/experiments/results"]

# --- Default command: run pipeline ---
CMD ["bash", "run_pipeline.sh"]
