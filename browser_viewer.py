#!/usr/bin/env python3
"""
Browser-based viewer for the processed video stream.
This script creates a simple web server that displays the processed video stream in a browser.
"""

import cv2
import os
import time
import threading
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket
import base64
import json
import argparse
import sys
from urllib.parse import parse_qs, urlparse

# Global variables
processed_frame = None
processed_frame_lock = threading.Lock()
frames_received = 0
last_frame_time = 0
connection_active = True

class ViewerHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the browser viewer"""
    
    def do_GET(self):
        """Handle GET requests"""
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
            global processed_frame, processed_frame_lock
            
            with processed_frame_lock:
                if processed_frame is not None:
                    frame_to_send = processed_frame.copy()
                else:
                    # Create a blank frame if no processed frame is available
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
                self.wfile.write(jpg_data.tobytes())
            except Exception as e:
                print(f"Error encoding frame: {e}")
                
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
            
            global frames_received, last_frame_time
            
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

def update_frame(frame):
    """Update the global processed frame"""
    global processed_frame, processed_frame_lock, frames_received, last_frame_time
    
    with processed_frame_lock:
        processed_frame = frame.copy()
    
    frames_received += 1
    last_frame_time = time.time()

def run_server(port=8088):
    """Run the HTTP server"""
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
    print(f"Press Ctrl+C to stop the server")
    print(f"===========================\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down the server...")
        httpd.server_close()
        print("Server stopped")

def get_processed_frame_from_stream(remote_host, remote_port=8082):
    """Get processed frames from the remote MJPEG stream and update the global frame"""
    global connection_active
    
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
        # Create a buffer for the MJPEG stream
        buffer = bytes()
        boundary = b"--jpgboundary"
        
        # Start the streaming request
        response = requests.get(url, stream=True, timeout=30)
        
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            return
        
        print("Stream connected, receiving frames...")
        
        # Process the stream
        for chunk in response.iter_content(chunk_size=1024):
            if not chunk or not connection_active:
                break
            
            # Add the chunk to the buffer
            buffer += chunk
            
            # Process frames if we find a boundary
            if boundary in buffer:
                parts = buffer.split(boundary)
                
                if len(parts) > 1:
                    for part in parts[:-1]:
                        # Find the JPEG data
                        jpg_start = part.find(b'\r\n\r\n') + 4
                        if jpg_start > 4:
                            jpg_data = part[jpg_start:]
                            if jpg_data:
                                try:
                                    # Decode the JPEG data
                                    nparr = np.frombuffer(jpg_data, np.uint8)
                                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                    
                                    if img is not None and img.size > 0:
                                        # Update the global frame
                                        update_frame(img)
                                    else:
                                        print("Warning: Received invalid image data")
                                except Exception as e:
                                    print(f"Error decoding image: {e}")
                    
                    # Keep the last part, which might be incomplete
                    buffer = parts[-1]
        
    except Exception as e:
        print(f"Error in stream processing: {e}")
    finally:
        print("Stream connection closed")
        connection_active = False

def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Browser-based viewer for object detection stream")
    parser.add_argument("remote_host", help="IP address or hostname of the remote processing server")
    parser.add_argument("--remote-port", type=int, default=8082, help="Port for the remote stream (default: 8082)")
    parser.add_argument("--viewer-port", type=int, default=8088, help="Port for the browser viewer (default: 8088)")
    args = parser.parse_args()
    
    # Import requests here to allow the script to run initially even if the module is missing
    global requests
    try:
        import requests
    except ImportError:
        print("Error: The 'requests' module is required.")
        print("Please install it using: pip install requests")
        return 1
    
    # Start the frame fetching thread
    global connection_active
    connection_active = True
    
    fetch_thread = threading.Thread(
        target=get_processed_frame_from_stream, 
        args=(args.remote_host, args.remote_port)
    )
    fetch_thread.daemon = True
    fetch_thread.start()
    
    # Start the HTTP server
    try:
        run_server(args.viewer_port)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        connection_active = False
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 