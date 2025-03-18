#!/bin/bash

# Make sure output directory exists
mkdir -p output

# Check if opencv-python is installed
if ! python3 -c "import cv2" &> /dev/null; then
    echo "Installing opencv-python..."
    pip3 install opencv-python
fi

# Start webcam streaming server in the background
echo "Starting webcam streaming server..."
python3 stream_webcam.py &
STREAM_PID=$!

# Give streaming server time to start
sleep 2

# Build Docker image if it doesn't exist
if [[ "$(docker images -q object-detection-system 2> /dev/null)" == "" ]]; then
    echo "Building Docker image..."
    docker build -t object-detection-system .
fi

# Run Docker container
echo "Running object detection on webcam stream..."
docker run --rm \
    -v "$(pwd)/output:/app/output" \
    --add-host=host.docker.internal:host-gateway \
    object-detection-system webcam --model yolov8 --video http://host.docker.internal:8081/video.mjpeg --output /app/output/webcam_output.mp4

# Clean up - kill streaming server
echo "Stopping webcam streaming server..."
kill $STREAM_PID

echo "Done! Output video saved to output/webcam_output.mp4" 