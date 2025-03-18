# Object Detection System

A versatile object detection system with support for real-time webcam tracking and video file processing.

## Features

- Support for multiple detection models:
  - MobileNet SSD (fastest, but less accurate)
  - YOLOv4-tiny (good balance of speed and accuracy) 
  - YOLOv8 (most accurate, with multiple implementations)
- Real-time webcam tracking
- Video file processing (single or batch)
- Docker containerization for easy deployment

## Installation

### Option 1: Native Installation

1. Clone the repository
   ```bash
   git clone https://github.com/YOUR_USERNAME/object-detection-system.git
   cd object-detection-system
   ```

2. Create a virtual environment (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Download model files
   ```bash
   python core/download_models.py
   ```

### Option 2: Docker Installation (Recommended)

1. Clone the repository
   ```bash
   git clone https://github.com/YOUR_USERNAME/object-detection-system.git
   cd object-detection-system
   ```

2. Run the setup script
   ```bash
   ./setup_docker.sh
   ```

3. Build the Docker image
   ```bash
   docker build -t object-detection-system .
   ```

## Usage

### Native Usage

#### List Available Models
```bash
python main.py --list-models
```

#### Real-time Webcam Tracking
```bash
python main.py webcam --model yolov8
```

#### Process a Video File
```bash
python main.py video --video path/to/video.mp4 --output path/to/output.mp4 --model yolov8
```

### Docker Usage

#### List Available Models
```bash
docker run --rm object-detection-system --list-models
```

#### Real-time Webcam Tracking (Linux)
```bash
docker run --rm -it \
  --device=/dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  object-detection-system webcam --model yolov8
```

#### Process a Video File
```bash
# Place your video in the input directory, then run:
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  object-detection-system video --model yolov8 --video /app/input/your_video.mp4 --output /app/output/processed.mp4
```

### Using Docker Compose

#### Webcam Tracking
```bash
docker-compose up webcam
```

#### Process a Single Video
```bash
docker-compose up video
```

#### Batch Process Videos
```bash
docker-compose up batch
```

## Configuration Options

- `--model {mobilenet,yolo,yolov8}`: Choose detection model (default: mobilenet)
- `--camera CAMERA`: Camera index (default: 0)
- `--confidence CONFIDENCE`: Confidence threshold (default: 0.5)
- `--nms NMS`: Non-maximum suppression threshold for YOLO (default: 0.4)
- `--classes CLASSES`: Comma-separated list of classes to detect (default: all)

YOLOv8 specific options:
- `--yolo-size {n,s,m,l,x}`: Model size (default: m)
- `--force-yolo {ultralytics,onnx,yolov4-tiny,yolov8core}`: YOLOv8 implementation (default: yolov8core)

## Docker-Specific Information

For more detailed information on using Docker with this project, please see the [Docker README](DOCKER_README.md).

## Project Structure

```
.
├── core/                    # Core object detection components
│   ├── models/              # All model files are stored here
│   ├── yolov8_wrapper.py    # YOLOv8 implementation with multiple backends
│   ├── yolov8_core.py       # Lightweight YOLOv8 implementation
│   ├── download_models.py   # Script to download required model files
│   └── coco_classes.txt     # COCO dataset class names
├── webcam_tracking/         # Real-time webcam tracking
│   ├── human_tracker.py     # MobileNet SSD tracker
│   ├── human_tracker_yolo.py # YOLOv4-tiny tracker
│   └── run_tracker.py       # Entry point for webcam tracking
├── video_processing/        # Video file processing
│   ├── video_processor.py   # Single video processing
│   ├── batch_process_videos.py # Batch video processing
│   └── generate_summary.py  # Generate summary reports
├── docker-compose.yml       # Docker Compose configuration
├── Dockerfile               # Docker build instructions
├── DOCKER_README.md         # Docker-specific documentation
├── setup_docker.sh          # Docker setup script
├── requirements.txt         # Python dependencies
└── main.py                  # Main entry point
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 