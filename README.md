# Webcam Processing System

This project provides scripts for streaming webcam footage to a remote processing server and viewing the processed results.

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- Requests

Install dependencies:
```bash
pip install opencv-python numpy requests
```

## Scripts

### 1. Connection Test

Tests basic connectivity to the remote server:

```bash
./connection_test.py http://server-ip:port
```

### 2. Fix Stream (Viewer Only)

Views processed frames from the server in a browser:

```bash
./fix_stream.py http://server-ip:port/processed.mjpeg
```

Options:
- `--port PORT`: Specify a different port (default: 9099)
- `--debug`: Enable debug output
- `--test`: Run in test pattern mode without attempting to connect

### 3. Webcam Streamer (Streamer Only)

Streams webcam footage to the server:

```bash
./webcam_streamer.py http://server-ip:port/webcam
```

Options:
- `--fps FPS`: Target frames per second (default: 15)
- `--width WIDTH`: Width of frames (default: 640)
- `--height HEIGHT`: Height of frames (default: 480)
- `--debug`: Enable debug output

### 4. All-in-One (Streamer + Viewer)

Combines the webcam streaming and frame viewing in a single script:

```bash
./all_in_one.py http://server-ip:port
```

Options:
- `--viewer-port PORT`: Specify a different port for the viewer (default: 9099)
- `--fps FPS`: Target frames per second for webcam (default: 15)
- `--width WIDTH`: Width of webcam frames (default: 640)
- `--height HEIGHT`: Height of webcam frames (default: 480)
- `--debug`: Enable debug output
- `--test`: Run in test pattern mode without webcam

## Troubleshooting

### Connectivity Issues
- Use `connection_test.py` to check basic connectivity to the server
- Ensure the server is running and accepting connections
- Check firewall settings on both client and server
- Verify correct IP addresses and port numbers

### Webcam Issues
- Make sure webcam is connected and not in use by another application
- Try different resolutions if camera fails to initialize

### Display Issues
- Always use the browser at http://localhost:9099/ (or your specified port)
- If no frames appear, check server connectivity
- The test pattern should appear if server connection fails

## Server Endpoints

The system expects the remote server to have these endpoints:
- `/webcam`: Receives webcam frames via POST requests
- `/processed.mjpeg`: Streams processed frames as MJPEG stream 