#!/bin/bash

# Build the Docker image
docker build -t realtime-runner -f m1.Dockerfile .

# Stop and remove any existing container
docker rm -f rr-instance || true

# Run the Docker container
docker run --name rr-instance -it realtime-runner /bin/bash