# ---------------------------------------------------------
#  IKRAE-EdNet | Stable Build (Python 3.10 + OpenJDK 17)
# ---------------------------------------------------------
FROM python:3.10-bullseye

LABEL maintainer="Aziz Abdelkarim <a.karim.linux@gmail.com>"
LABEL description="Docker image for IKRAE-EdNet: Semantic + Graph-based Adaptive Learning Framework"

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# ---------------------------------------------------------
#  System Dependencies
# ---------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-17-jdk \
        git \
        wget \
        unzip \
        ca-certificates \
        build-essential \
        pkg-config \
        libfreetype6-dev \
        libpng-dev \
        libjpeg-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

# ---------------------------------------------------------
#  Application Setup
# ---------------------------------------------------------
WORKDIR /app
COPY . /app

# Ensure results folder exists
RUN mkdir -p experiments/results

# Make the pipeline script executable
RUN chmod +x run_pipeline.sh

# ---------------------------------------------------------
#  Python Dependencies
# ---------------------------------------------------------
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Debug info
RUN java -version && python3 --version

# ---------------------------------------------------------
#  Default Command
# ---------------------------------------------------------
CMD ["bash", "run_pipeline.sh"]
