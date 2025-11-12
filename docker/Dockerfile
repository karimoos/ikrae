# ---------------------------------------------------------
#  IKRAE-EdNet | Stable Build (OpenJDK 17 + Python 3.10)
# ---------------------------------------------------------
FROM python:3.10-bullseye

LABEL maintainer="Aziz Abdelkarim <a.karim.linux@gmail.com>"
LABEL description="Docker image for IKRAE-EdNet: Semantic + Graph-based Adaptive Learning Framework"

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# --- Install dependencies ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-17-jdk \
        git \
        wget \
        unzip \
        ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Environment variables ---
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

# --- Copy repo ---
WORKDIR /app
COPY . /app

# --- Python dependencies ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Quick runtime sanity check ---
RUN java -version && python --version

# --- Default command ---
CMD ["bash", "run_pipeline.sh"]
