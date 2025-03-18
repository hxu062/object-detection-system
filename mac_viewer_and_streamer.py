#!/usr/bin/env python3
"""
Combined webcam streamer and browser viewer for macOS.
This script:
1. Captures your MacBook webcam
2. Streams it to the remote processing server
3. Receives the processed frames from the server
4. Displays them in a web browser
"""

import cv2
import os
import time
import threading
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket
import json
import argparse
import sys
import traceback

# Global variables for the processed stream
processed_frame = None
processed_frame_lock = threading.Lock()
frames_received = 0
last_frame_time = 0
connection_active = True

class WebcamStreamHandler(BaseHTTPRequestHandler):
    """Handle HTTP requests for the webcam stream"""
    
    def do_GET(self):
        """Handle GET requests"""
        global connection_active
        
        if self.path == '/video.mjpeg':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            
            try:
                # Open webcam on index 0 (default)
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    print("Error: Could not open webcam")
                    return
                
                print("Webcam stream started. Streaming to remote server.")
                
                while connection_active:
                    ret, frame = cap.read()
                    if not ret:
                        print("Warning: Failed to grab frame")
                        time.sleep(0.1)  # Add delay to avoid busy loop
                        continue
                    
                    # Show local preview if requested
                    try:
                        cv2.imshow('Local Webcam Preview', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            connection_active = False
                            break
                    except Exception as e:
                        # If we can't show the preview, just continue without it
                        if not hasattr(self, 'preview_error_shown'):
                            print(f"Warning: Cannot show preview window: {e}")
                            self.preview_error_shown = True
                    
                    # Add some error handling around JPEG encoding
                    try:
                        # Encode frame as JPEG
                        ret, jpg = cv2.imencode('.jpg', frame)
                        if not ret:
                            print("Warning: Failed to encode frame")
                            continue
                        
                        # Write the JPEG to the stream
                        self.wfile.write(b"--jpgboundary\r\n")
                        self.send_header('Content-type', 'image/jpeg')
                        self.send_header('Content-length', str(len(jpg)))
                        self.end_headers()
                        self.wfile.write(jpg.tobytes())
                        self.wfile.write(b"\r\n")
                    except Exception as e:
                        print(f"Error in webcam streaming: {e}")
                        time.sleep(0.1)  # Add delay to avoid busy loop
                    
                    # Small delay to prevent overwhelming the network
                    time.sleep(0.05)
                    
            except BrokenPipeError:
                print("Client disconnected")
            except Exception as e:
                print(f"Streaming error: {e}")
                print(traceback.format_exc())
            finally:
                if 'cap' in locals() and cap.isOpened():
                    cap.release()
                try:
                    cv2.destroyAllWindows()
                except:
                    pass
                print("Webcam streaming stopped.")
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><head></head><body><h1>404 Not Found</h1></body></html>')
    
    # Disable request logging to console
    def log_message(self, format, *args):
        return

class ViewerHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the browser viewer"""
    
    def do_GET(self):
        """Handle GET requests"""
        global processed_frame, processed_frame_lock, frames_received, last_frame_time
        
        # Serve the main HTML page
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Object Detection Stream</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: #f0f0f0;
                        text-align: center;
                    }
                    h1 {
                        color: #333;
                    }
                    #videoContainer {
                        margin: 20px auto;
                        max-width: 1000px;
                        position: relative;
                    }
                    #streamImage {
                        width: 100%;
                        border: 2px solid #333;
                        box-shadow: 0 0 10px rgba(0,0,0,0.2);
                    }
                    #stats {
                        margin-top: 10px;
                        font-size: 16px;
                        color: #555;
                    }
                    #fpsInfo {
                        position: absolute;
                        top: 10px;
                        left: 10px;
                        background-color: rgba(0,0,0,0.5);
                        color: white;
                        padding: 5px 10px;
                        border-radius: 5px;
                    }
                    .button {
                        background-color: #4CAF50;
                        border: none;
                        color: white;
                        padding: 10px 24px;
                        text-align: center;
                        text-decoration: none;
                        display: inline-block;
                        font-size: 16px;
                        margin: 4px 2px;
                        cursor: pointer;
                        border-radius: 4px;
                    }
                </style>
            </head>
            <body>
                <h1>Object Detection Stream</h1>
                <div id="videoContainer">
                    <img id="streamImage" src="/no-image.jpg" alt="Video Stream">
                    <div id="fpsInfo">FPS: 0</div>
                </div>
                <div id="stats">
                    Connecting...
                </div>
                <button class="button" onclick="toggleStream()">Pause/Resume</button>
                
                <script>
                    let streamPaused = false;
                    let lastFrameTime = Date.now();
                    let frameCount = 0;
                    let fps = 0;
                    
                    function toggleStream() {
                        streamPaused = !streamPaused;
                    }
                    
                    function updateImage() {
                        if (!streamPaused) {
                            // Add cache-busting parameter to prevent browser caching
                            const img = document.getElementById('streamImage');
                            img.src = '/stream.jpg?t=' + new Date().getTime();
                            
                            // Update FPS
                            frameCount++;
                            const now = Date.now();
                            const elapsed = now - lastFrameTime;
                            
                            if (elapsed >= 1000) {
                                fps = (frameCount / elapsed) * 1000;
                                document.getElementById('fpsInfo').innerText = `FPS: ${fps.toFixed(1)}`;
                                frameCount = 0;
                                lastFrameTime = now;
                            }
                        }
                    }
                    
                    function updateStats() {
                        fetch('/stats.json')
                            .then(response => response.json())
                            .then(data => {
                                document.getElementById('stats').innerHTML = 
                                    `Frames received: ${data.frames_received} | ` +
                                    `Last frame: ${data.last_frame_ago.toFixed(1)}s ago | ` +
                                    `Client FPS: ${fps.toFixed(1)}`;
                            })
                            .catch(error => {
                                console.error('Error fetching stats:', error);
                            });
                    }
                    
                    // Update image at 30 FPS (33ms intervals)
                    setInterval(updateImage, 33);
                    
                    // Update stats every 2 seconds
                    setInterval(updateStats, 2000);
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        
        # Serve the current frame as JPEG
        elif self.path.startswith('/stream.jpg'):
            with processed_frame_lock:
                if processed_frame is not None and processed_frame.size > 0:
                    # Log successful frame access on first few requests
                    if frames_received < 10:
                        print(f"Serving frame to browser: shape={processed_frame.shape}")
                    frame_to_send = processed_frame.copy()
                else:
                    # Create a blank frame if no processed frame is available
                    print("Warning: No valid frame to display, using placeholder")
                    frame_to_send = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame_to_send, "Waiting for video...", (50, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Draw stats on the frame
            cv2.putText(frame_to_send, f"Frames: {frames_received}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Send the frame as JPEG
            self.send_response(200)
            self.send_header('Content-type', 'image/jpeg')
            self.end_headers()
            
            try:
                _, jpg_data = cv2.imencode('.jpg', frame_to_send)
                jpg_bytes = jpg_data.tobytes()
                # Log size of first few JPEGs
                if frames_received < 10:
                    print(f"Sending JPEG: {len(jpg_bytes)} bytes")
                self.wfile.write(jpg_bytes)
            except Exception as e:
                print(f"Error encoding frame: {e}")
                print(traceback.format_exc())
                
        # Serve a placeholder image for the initial load
        elif self.path == '/no-image.jpg':
            self.send_response(200)
            self.send_header('Content-type', 'image/jpeg')
            self.end_headers()
            
            # Create a black image with a message
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "Loading stream...", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            try:
                _, jpg_data = cv2.imencode('.jpg', img)
                self.wfile.write(jpg_data.tobytes())
            except Exception as e:
                print(f"Error creating placeholder image: {e}")
        
        # Serve statistics as JSON
        elif self.path == '/stats.json':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            current_time = time.time()
            last_frame_ago = current_time - last_frame_time if last_frame_time > 0 else 0
            
            stats = {
                "frames_received": frames_received,
                "last_frame_ago": last_frame_ago
            }
            
            self.wfile.write(json.dumps(stats).encode())
        
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><body><h1>404 Not Found</h1></body></html>')
    
    # Disable request logging to console
    def log_message(self, format, *args):
        return

def run_webcam_server(port=8081):
    """Run the webcam streaming server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, WebcamStreamHandler)
    print(f"Starting webcam stream server on port {port}")
    
    # Run the server in a separate thread
    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()
    return httpd

def run_viewer_server(port=8088):
    """Run the HTTP server for the viewer"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, ViewerHandler)
    
    # Get the local IP address
    ip = "unknown"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except:
        pass
    
    print(f"\n=== Browser-based Viewer ===")
    print(f"Open in your browser:")
    print(f"  http://localhost:{port}/")
    print(f"  http://{ip}:{port}/")
    print(f"Press Ctrl+C to stop both servers")
    print(f"===========================\n")
    
    # Run the server in a separate thread
    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()
    return httpd

def update_frame(frame):
    """Update the global processed frame"""
    global processed_frame, processed_frame_lock, frames_received, last_frame_time
    
    with processed_frame_lock:
        processed_frame = frame.copy()
    
    frames_received += 1
    last_frame_time = time.time()
    
    # Print a message every 10 frames to show progress
    if frames_received % 10 == 0:
        print(f"✓ Received {frames_received} frames, latest shape: {frame.shape}")

def get_processed_frame_from_stream(remote_host, remote_port=8082):
    """Get processed frames from the remote MJPEG stream and update the global frame"""
    global connection_active
    
    # Import requests here to allow the script to run initially even if the module is missing
    try:
        import requests
    except ImportError:
        print("Error: The 'requests' module is required.")
        print("Please install it using: pip install requests")
        sys.exit(1)
    
    url = f"http://{remote_host}:{remote_port}/processed.mjpeg"
    print(f"Connecting to processed stream at {url}")
    
    # Try to connect to the server
    retry_count = 0
    max_retries = 10
    connected = False
    
    while not connected and retry_count < max_retries and connection_active:
        try:
            print(f"Connecting to {url} (attempt {retry_count+1}/{max_retries})...")
            response = requests.head(url, timeout=2)
            if response.status_code == 200:
                connected = True
                print("Connected to remote server!")
            else:
                print(f"Server responded with status code: {response.status_code}")
                retry_count += 1
                time.sleep(2)
        except Exception as e:
            print(f"Connection error: {e}")
            retry_count += 1
            time.sleep(2)
    
    if not connected:
        print(f"Failed to connect to remote server at {url} after {max_retries} attempts")
        return
    
    try:
        # Start the streaming request
        response = requests.get(url, stream=True, timeout=30)
        
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            return
        
        print("Stream connected, receiving frames...")
        
        # Process the stream
        buffer = bytes()
        content_length = None
        content_type = None
        frame_count = 0
        
        for chunk in response.iter_content(chunk_size=10240):  # Larger chunk size
            if not chunk or not connection_active:
                break
                
            buffer += chunk
            
            # Look for JPEG image boundary
            if b'Content-Type: image/jpeg' in buffer:
                # Find header
                header_start = buffer.find(b'--jpgboundary')
                if header_start >= 0:
                    # Find content type
                    ct_pos = buffer.find(b'Content-Type: ', header_start)
                    if ct_pos > 0:
                        # Find content length
                        cl_pos = buffer.find(b'Content-Length: ', header_start)
                        if cl_pos > 0:
                            cl_end = buffer.find(b'\r\n', cl_pos)
                            if cl_end > 0:
                                try:
                                    content_length = int(buffer[cl_pos + 16:cl_end])
                                except:
                                    content_length = None
                        
                        # Find image data - double newline indicates start of binary data
                        img_start = buffer.find(b'\r\n\r\n', ct_pos)
                        if img_start > 0:
                            img_start += 4  # Skip the \r\n\r\n
                            
                            # If we have content length, use it to find end of image
                            if content_length and len(buffer) >= img_start + content_length:
                                img_data = buffer[img_start:img_start + content_length]
                                
                                # Try to decode the image
                                try:
                                    nparr = np.frombuffer(img_data, np.uint8)
                                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                    
                                    if img is not None and img.size > 0:
                                        frame_count += 1
                                        update_frame(img)
                                        print(f"✓ Frame {frame_count} successfully decoded, length: {len(img_data)} bytes")
                                    else:
                                        print(f"Warning: Could not decode image data, length: {len(img_data)}")
                                except Exception as e:
                                    print(f"Error decoding image: {e}")
                                
                                # Keep only the remainder
                                buffer = buffer[img_start + content_length:]
                            else:
                                # If we don't know the content length, just keep looking for next boundary
                                next_boundary = buffer.find(b'--jpgboundary', img_start)
                                if next_boundary > img_start:
                                    img_data = buffer[img_start:next_boundary]
                                    
                                    # Try to decode the image
                                    try:
                                        nparr = np.frombuffer(img_data, np.uint8)
                                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                        
                                        if img is not None and img.size > 0:
                                            frame_count += 1
                                            update_frame(img)
                                            print(f"✓ Frame {frame_count} successfully decoded, length: {len(img_data)} bytes")
                                        else:
                                            print(f"Warning: Could not decode image data, length: {len(img_data)}")
                                    except Exception as e:
                                        print(f"Error decoding image: {e}")
                                    
                                    # Keep only from the next boundary
                                    buffer = buffer[next_boundary:]
            
            # Avoid buffer growing too large
            if len(buffer) > 1000000:  # 1MB max buffer
                # Just keep the last part where a boundary might be
                last_boundary = buffer.rfind(b'--jpgboundary')
                if last_boundary > 0:
                    buffer = buffer[last_boundary:]
                else:
                    # If no boundary found, keep only the last 10KB
                    buffer = buffer[-10240:]
    
    except Exception as e:
        print(f"Error in stream processing: {e}")
        print(traceback.format_exc())
    finally:
        print("Stream connection closed")

def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Combined webcam streamer and browser viewer for macOS")
    parser.add_argument("remote_host", help="IP address or hostname of the remote processing server")
    parser.add_argument("--webcam-port", type=int, default=8081, help="Port for webcam streaming (default: 8081)")
    parser.add_argument("--remote-port", type=int, default=8082, help="Port for the remote stream (default: 8082)")
    parser.add_argument("--viewer-port", type=int, default=8088, help="Port for the browser viewer (default: 8088)")
    parser.add_argument("--show-preview", action="store_true", help="Show local webcam preview")
    args = parser.parse_args()
    
    # Get local IP for instructions
    local_ip = "unknown"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        pass
    
    print("\n====== Object Detection Client ======")
    print(f"Your MacBook's IP address is: {local_ip}")
    print("Make sure the remote server is running with:")
    print(f"  python remote_processing_server.py --client-host {local_ip} --model yolov8")
    print("======================================\n")
    
    # Set connection active
    global connection_active
    connection_active = True
    
    # Start webcam server
    webcam_server = run_webcam_server(args.webcam_port)
    print(f"Streaming webcam to remote server at: {args.remote_host}:{args.remote_port}")
    print(f"The remote server should connect to: http://{local_ip}:{args.webcam_port}/video.mjpeg")
    
    # Start frame fetching thread
    fetch_thread = threading.Thread(
        target=get_processed_frame_from_stream, 
        args=(args.remote_host, args.remote_port)
    )
    fetch_thread.daemon = True
    fetch_thread.start()
    
    # Start the viewer server
    viewer_server = run_viewer_server(args.viewer_port)
    
    try:
        # Keep the main thread alive
        while connection_active:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Clean up
        connection_active = False
        print("Shutting down servers...")
        webcam_server.shutdown()
        viewer_server.shutdown()
        time.sleep(1)
        print("Done")

if __name__ == "__main__":
    sys.exit(main()) 