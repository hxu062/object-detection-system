#!/bin/bash

# Create necessary directories for Docker volumes
mkdir -p input output

# Print instructions
echo "====================================================="
echo "Docker setup complete for Object Detection System!"
echo "====================================================="
echo ""
echo "The following directories have been created:"
echo "- input/  : Place your video files here"
echo "- output/ : Processed videos and results will appear here"
echo ""
echo "To build and run the Docker container, use:"
echo "  docker build -t object-detection-system ."
echo ""
echo "For more details, please refer to DOCKER_README.md"
echo "====================================================="

# Make the script executable
chmod +x "$0" 