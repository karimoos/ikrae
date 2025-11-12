# =====================================================
#  IKRAE Makefile
#  shortcuts for development and testing
# =====================================================

.PHONY: setup run docker-build docker-run clean

# Create / update the conda environment
setup:
	conda env create -f environment.yml || true
	conda activate ikrae-ednet

# Run full pipeline locally (requires EdNet data)
run:
	bash run_pipeline.sh

# Build the Docker image
docker-build:
	docker build -t ikrae-ednet .

# Run the Docker container (mount experiments folder)
docker-run:
	docker run --rm -v $$(pwd)/experiments:/app/experiments ikrae-ednet

# Clean result files
clean:
	rm -rf experiments/results/* || true
