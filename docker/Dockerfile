FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y openjdk-11-jdk
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

COPY lib/hermit/HermiT.jar /app/lib/
COPY src/ /app/src/
COPY ontology/ /app/ontology/

CMD ["python", "src/ednet_loader.py"]
