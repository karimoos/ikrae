# =========================================================
#  IKRAE-EdNet  |  Robust GitHub-Actions-safe Dockerfile
# =========================================================
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-11-jre-headless \
        git wget unzip ca-certificates && \
    rm -rf /var/lib/apt/lists/*

LABEL maintainer="Aziz Abdelkarim <a.karim.linux@gmail.com>"

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# --- Reliable system setup ---
# Use ipv4 & retry mirrors (avoids GitHub mirror timeouts)
RUN echo "Acquire::ForceIPv4 \"true\";" > /etc/apt/apt.conf.d/99force-ipv4 && \
    apt-get update -o Acquire::Retries=5 && \
    apt-get install -y --no-install-recommends \
        openjdk-11-jre-headless \
        curl \
        git \
        wget \
        unzip \
        ca-certificates \
    || (echo 'Initial apt failed, retrying with alternate mirror...' && \
        sed -i 's@deb.debian.org@archive.debian.org@g' /etc/apt/sources.list && \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            openjdk-11-jre-headless curl git wget unzip ca-certificates) && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Environment variables ---
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Quick sanity check (ensures both runtimes exist)
RUN python --version && java --version

CMD ["bash", "run_pipeline.sh"]
