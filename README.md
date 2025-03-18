# Object Detection System

A versatile object detection system with support for real-time webcam tracking and video file processing.

## Features

- Support for multiple detection models:
  - MobileNet SSD (fastest, but less accurate)
  - YOLOv4-tiny (good balance of speed and accuracy) 
  - YOLOv8 (most accurate, with multiple implementations)
- Real-time webcam tracking
- Video file processing (single or batch)
- Remote processing server for client-server setup

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- Requests
- PyTorch (for YOLOv8)

Install dependencies:
```bash
pip install -r requirements.txt
```

## System Components

### 1. Remote Processing Server

Runs on a server (Linux recommended) to process webcam feeds from clients:

```bash
./remote_processing_server.py --port 8082 --model yolov8
```

Options:
- `--port PORT`: Server port (default: 8082)
- `--model {mobilenet,yolo,yolov8}`: Detection model (default: mobilenet)
- `--confidence CONFIDENCE`: Detection threshold (default: 0.5)
- `--client-url URL`: URL to fetch webcam stream from (default: constructed from client IP)

### 2. Client-Side Tools

#### Webcam Streaming and Viewing

Use the following scripts to interact with the remote server:

```bash
# Test connection to server
./connection_test.py http://server-ip:8082

# Stream webcam to server
./webcam_streamer.py http://server-ip:8082/webcam

# View processed stream in browser
./fix_stream.py http://server-ip:8082/processed.mjpeg

# All-in-one solution (stream and view)
./all_in_one.py http://server-ip:8082
```

### 3. Video Processing

For processing saved video files:

```bash
# Process a single video
./main.py video --video path/to/video.mp4 --output path/to/output.mp4 --model yolov8

# Batch process multiple videos
./main.py video --batch --input-dir path/to/videos --output-dir path/to/outputs --model yolov8
```

Options:
- `--model {mobilenet,yolo,yolov8}`: Detection model (default: mobilenet)
- `--confidence CONFIDENCE`: Detection threshold (default: 0.5)
- `--generate-report`: Generate analysis report (batch mode only)

## Running the Complete System

### Server Side (Remote Machine)

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download models: `python core/download_models.py`
4. Start the server: `./remote_processing_server.py --port 8082 --model yolov8`

### Client Side (Local Machine)

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Use the all-in-one script: `./all_in_one.py http://server-ip:8082`
4. Open your browser: `http://localhost:9099/`

## Troubleshooting

- Use `./connection_test.py` to verify server connectivity
- Try different models if processing is too slow
- Check webcam access on the client
- Ensure proper paths for video processing
- Run with `--debug` flag for more verbose output 