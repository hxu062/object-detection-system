# Docker Usage Guide for Object Detection System

This guide explains how to build and run the Object Detection System using Docker.

## Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose installed ([Get Docker Compose](https://docs.docker.com/compose/install/))

## Quick Start

### Build the Docker Image

```bash
docker build -t object-detection-system .
```

### Run Using Docker Command

#### Show Help

```bash
docker run --rm object-detection-system
```

#### Run Webcam Tracking (Linux)

```bash
docker run --rm -it \
  --device=/dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  object-detection-system webcam --model yolov8
```

Note: For webcam access on macOS, you'll need to use a solution like [Docker-Webcam-macOS](https://github.com/dymat/Docker-Webcam-macOS).

#### Process a Video File

```bash
# Create input and output directories
mkdir -p input output

# Place your video in the input directory
# Then run:
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  object-detection-system video --model yolov8 --video /app/input/your_video.mp4 --output /app/output/processed.mp4
```

### Run Using Docker Compose

We've provided a `docker-compose.yml` with several predefined services:

#### Webcam Tracking

```bash
docker-compose up webcam
```

#### Process a Single Video

1. Place your video file at `./input/video.mp4`
2. Run:
```bash
docker-compose up video
```

#### Batch Process Videos

1. Place your videos in the `./input` directory
2. Run:
```bash
docker-compose up batch
```

## Customizing the Container

### Change the Detection Model

You can specify which model to use:

```bash
docker run --rm object-detection-system webcam --model mobilenet
# or
docker run --rm object-detection-system webcam --model yolo
# or
docker run --rm object-detection-system webcam --model yolov8
```

### Other Configuration Options

All command-line options described in the main README.md work with Docker as well:

```bash
docker run --rm object-detection-system webcam --model yolov8 --confidence 0.7 --classes person,car
```

## Troubleshooting

### GUI Display Issues

If you're having trouble with webcam display:

1. On Linux, ensure you've allowed X server connections:
```bash
xhost +local:docker
```

2. On macOS, you may need to install XQuartz and set it up for remote connections.

3. On Windows, you may need to use an X server like VcXsrv or Xming.

### Missing Video Device

If your webcam isn't detected:

```bash
# List available video devices
ls -l /dev/video*

# Then update the device mapping in your docker command
docker run --rm -it --device=/dev/video1:/dev/video0 ...
```

## Building a Custom Image

If you need to modify the Dockerfile, here are the key sections:

1. Base image: Currently uses Python 3.9 slim
2. System dependencies: OpenCV requirements and ffmpeg 
3. Python dependencies: Installed from requirements.txt
4. Model download: Occurs during build

After making changes, rebuild the image:

```bash
docker build -t object-detection-system:custom .
``` 