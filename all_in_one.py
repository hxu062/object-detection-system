#!/usr/bin/env python3
"""
All-in-One Webcam Streamer and Viewer

This script:
1. Captures frames from your webcam
2. Sends them to the remote processing server
3. Receives the processed frames
4. Displays them in a web browser

Usage:
    ./all_in_one.py http://server-ip:port
"""

import cv2
import numpy as np
import time
import threading
import requests
import argparse
import sys
import os
import traceback
import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

# Global variables
stream_frame = None  # Processed frame from server
frame_lock = threading.Lock()
frames_sent = 0
frames_received = 0
connection_active = True

# For test pattern
pattern_count = 0
pattern_start_time = time.time()

class ViewerHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the browser viewer"""
    
    def do_GET(self):
        """Handle GET requests"""
        # Main page
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>All-in-One Webcam Stream</title>
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
                        max-width: 850px;
                        border: 2px solid #333;
                        box-shadow: 0 0 10px rgba(0,0,0,0.2);
                    }
                    #streamImage {
                        width: 100%;
                    }
                    #stats {
                        margin-top: 10px;
                        font-size: 16px;
                        color: #555;
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
                <h1>All-in-One Webcam Stream</h1>
                <div id="videoContainer">
                    <img id="streamImage" src="/frame.jpg" alt="Stream">
                </div>
                <div id="stats">Loading...</div>
                <button class="button" id="refreshBtn">Manual Refresh</button>
                <button class="button" id="autoBtn">Stop Auto-Refresh</button>
                
                <script>
                    let autoRefresh = true;
                    let refreshInterval;
                    const refreshRate = 100; // ms between refreshes
                    
                    function updateImage() {
                        document.getElementById('streamImage').src = 
                            '/frame.jpg?t=' + new Date().getTime();
                    }
                    
                    function updateStats() {
                        fetch('/stats.json?t=' + new Date().getTime())
                            .then(response => response.json())
                            .then(data => {
                                document.getElementById('stats').innerText = 
                                    `Sent: ${data.frames_sent} | Received: ${data.frames_received}`;
                            });
                    }
                    
                    function toggleAutoRefresh() {
                        if (autoRefresh) {
                            autoRefresh = false;
                            document.getElementById('autoBtn').innerText = "Start Auto-Refresh";
                            clearInterval(refreshInterval);
                        } else {
                            autoRefresh = true;
                            document.getElementById('autoBtn').innerText = "Stop Auto-Refresh";
                            startAutoRefresh();
                        }
                    }
                    
                    function startAutoRefresh() {
                        refreshInterval = setInterval(function() {
                            updateImage();
                            updateStats();
                        }, refreshRate);
                    }
                    
                    // Set up button listeners
                    document.getElementById('refreshBtn').addEventListener('click', function() {
                        updateImage();
                        updateStats();
                    });
                    
                    document.getElementById('autoBtn').addEventListener('click', function() {
                        toggleAutoRefresh();
                    });
                    
                    // Start auto-refresh by default
                    startAutoRefresh();
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
            
        # Serve the current frame
        elif self.path.startswith('/frame.jpg'):
            self.send_response(200)
            self.send_header('Content-type', 'image/jpeg')
            self.end_headers()
            
            # Get current frame or create placeholder
            with frame_lock:
                if stream_frame is not None and stream_frame.size > 0:
                    frame_to_send = stream_frame.copy()
                else:
                    # Create a placeholder frame if no frame is available
                    frame_to_send = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame_to_send, "No frame available", (180, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add frame counter
            cv2.putText(frame_to_send, f"Sent: {frames_sent} | Received: {frames_received}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Encode and send
            try:
                # Make sure we're sending a valid numpy array
                if frame_to_send is None or frame_to_send.size == 0:
                    frame_to_send = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame_to_send, "ERROR: Invalid frame", (180, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Ensure correct type for encoding
                if frame_to_send.dtype != np.uint8:
                    frame_to_send = frame_to_send.astype(np.uint8)
                
                # Encode as JPEG
                _, jpg_data = cv2.imencode('.jpg', frame_to_send)
                jpg_bytes = jpg_data.tobytes()
                
                # Send the image data
                self.wfile.write(jpg_bytes)
            except Exception as e:
                print(f"Error encoding/sending frame: {e}")
                print(traceback.format_exc())
            
        # Serve stats as JSON
        elif self.path.startswith('/stats.json'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            stats = {
                "frames_sent": frames_sent,
                "frames_received": frames_received,
            }
            
            self.wfile.write(json.dumps(stats).encode())
            
        else:
            # Not found
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><body><h1>404 Not Found</h1></body></html>')
    
    # Disable request logging to console
    def log_message(self, format, *args):
        return

def run_viewer_server(port=9099):
    """Run the HTTP server for the viewer"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, ViewerHandler)
    
    print(f"\n=== Browser Viewer ===")
    print(f"Open in your browser: http://localhost:{port}/")
    print(f"========================\n")
    
    # Run in a separate thread
    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()
    return httpd

def generate_test_pattern():
    """Generate a test pattern when the stream connection fails"""
    global stream_frame, pattern_start_time, pattern_count, frames_received
    
    pattern_start_time = time.time()
    pattern_count = 0
    
    width, height = 640, 480
    center_x, center_y = width // 2, height // 2
    radius = 50
    
    print("Starting test pattern generation")
    
    while connection_active:
        try:
            # Create a black frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Calculate circle position (moving in a circle)
            elapsed_time = time.time() - pattern_start_time
            angle = (elapsed_time * 30) % 360  # 30 degrees per second
            circle_x = int(center_x + 100 * math.cos(math.radians(angle)))
            circle_y = int(center_y + 100 * math.sin(math.radians(angle)))
            
            # Draw a yellow circle
            cv2.circle(frame, (circle_x, circle_y), radius, (0, 255, 255), -1)
            
            # Draw a rectangle border around the frame
            cv2.rectangle(frame, (10, 10), (width-10, height-10), (0, 0, 255), 2)
            
            # Add some text
            cv2.putText(frame, f"TEST PATTERN", (center_x-100, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"No connection to server", (width//2-150, height-120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Time: {elapsed_time:.1f}s", (width-200, height-50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
            # Update the frame with a lock to ensure thread safety
            with frame_lock:
                stream_frame = frame
                pattern_count += 1
                frames_received += 1  # Increment received frames counter
            
            # Print status periodically
            if pattern_count % 30 == 0:
                print(f"Generated {pattern_count} test pattern frames")
                
            # Sleep to control frame rate
            time.sleep(1/30)  # Aim for 30fps
            
        except Exception as e:
            print(f"Error generating test pattern: {e}")
            time.sleep(1)  # Retry after 1 second

def stream_webcam(server_url, fps=15, debug=False, width=640, height=480):
    """Stream webcam frames to the server"""
    global frames_sent, connection_active
    
    print(f"Initializing webcam...")
    cap = cv2.VideoCapture(0)  # Use default camera
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Get actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam initialized at {actual_width}x{actual_height}")
    
    # Create a session for better performance
    session = requests.Session()
    
    # Calculate frame delay for desired FPS
    frame_delay = 1.0 / fps
    
    print(f"Starting webcam stream to {server_url} at {fps} FPS")
    
    try:
        while connection_active:
            loop_start = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                time.sleep(1)  # Wait before retry
                continue
            
            # Add timestamp and frame counter
            cv2.putText(frame, f"Frame: {frames_sent}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Time: {time.strftime('%H:%M:%S')}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Encode frame to JPEG
            _, img_encoded = cv2.imencode('.jpg', frame)
            
            # Send the frame to server
            try:
                response = session.post(
                    server_url,
                    data=img_encoded.tobytes(),
                    headers={'Content-Type': 'image/jpeg'},
                    timeout=1
                )
                
                if response.status_code == 200:
                    frames_sent += 1
                    if debug or frames_sent % 50 == 0:
                        print(f"Sent frame {frames_sent} ({len(img_encoded)} bytes)")
                else:
                    print(f"Warning: Server returned status {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"Error sending frame: {e}")
                time.sleep(0.5)  # Wait before retry
            
            # Calculate time to sleep to maintain FPS
            elapsed = time.time() - loop_start
            sleep_time = max(0, frame_delay - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cap.release()
        print(f"Webcam stream stopped. Sent {frames_sent} frames.")

def fetch_mjpeg_stream(url, debug=False):
    """Connect to an MJPEG stream and update the global frame"""
    global stream_frame, frames_received, connection_active
    
    print(f"Connecting to stream at {url}")
    
    try:
        # Try a HEAD request first
        try:
            head_response = requests.head(url, timeout=5)
            print(f"HEAD response: {head_response.status_code}")
            
            if head_response.status_code != 200:
                print(f"Error: Server returned status code {head_response.status_code}")
                return
        except Exception as e:
            print(f"HEAD request error: {e}")
            print("Continuing anyway...")
        
        # Connect to the stream
        response = requests.get(url, stream=True, timeout=30)
        
        if response.status_code != 200:
            print(f"Error: GET request failed with status {response.status_code}")
            return
        
        print("Connected to stream, receiving data...")
        
        # Check content type
        content_type = response.headers.get('Content-type', '')
        print(f"Content-Type: {content_type}")
        
        # Look for boundary in content type
        boundary = None
        if 'boundary=' in content_type:
            boundary = content_type.split('boundary=')[1].strip()
            print(f"Found boundary: {boundary}")
        else:
            boundary = "--jpgboundary"  # Default boundary
            print(f"Using default boundary: {boundary}")
        
        # Set up buffer for processing
        buffer = bytes()
        boundary_bytes = boundary.encode() if isinstance(boundary, str) else boundary
        
        # Process frames from the stream
        for chunk in response.iter_content(chunk_size=4096):
            if not chunk or not connection_active:
                break
                
            # Add chunk to buffer
            buffer += chunk
            
            # Debug - print buffer size
            if debug and len(buffer) % 50000 < 4096:
                print(f"Buffer size: {len(buffer)} bytes")
            
            # Look for boundary markers
            if boundary_bytes in buffer:
                # Split on boundary
                parts = buffer.split(boundary_bytes)
                
                # Process complete frame parts
                if len(parts) > 1:
                    for part in parts[:-1]:
                        # Find headers and content
                        headers_end = part.find(b'\r\n\r\n')
                        if headers_end >= 0:
                            # Extract image data after headers
                            img_data = part[headers_end + 4:]
                            
                            if img_data:
                                try:
                                    # Decode image
                                    nparr = np.frombuffer(img_data, np.uint8)
                                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                    
                                    if img is not None and img.size > 0:
                                        # Update global frame
                                        with frame_lock:
                                            stream_frame = img.copy()
                                        
                                        frames_received += 1
                                        
                                        # Log progress
                                        if frames_received == 1 or frames_received % 50 == 0:
                                            print(f"Received frame {frames_received}, shape: {img.shape}")
                                    else:
                                        print(f"Warning: Unable to decode image, data size: {len(img_data)}")
                                except Exception as e:
                                    print(f"Error decoding image: {e}")
                                    if debug:
                                        print(traceback.format_exc())
                    
                    # Keep the last part (may be incomplete)
                    buffer = parts[-1]
            
            # Prevent buffer from growing too large
            if len(buffer) > 500000:  # 500KB limit
                last_boundary = buffer.rfind(boundary_bytes)
                if last_boundary > 0:
                    buffer = buffer[last_boundary:]
                else:
                    # If no boundary found, discard most of the buffer
                    buffer = buffer[-10000:]
    
    except Exception as e:
        print(f"Error in stream processing: {e}")
        print(traceback.format_exc())
    
    print("Stream connection closed")

def main():
    global connection_active, stream_frame, frames_received, frames_sent
    
    # Initialize variables
    connection_active = True
    stream_frame = None
    frames_received = 0
    frames_sent = 0
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='All-in-One Webcam Streamer and Viewer')
    parser.add_argument('server_url', help='URL of the server (e.g., http://100.91.32.59:8082)')
    parser.add_argument('--viewer-port', type=int, default=9099, help='Port for the viewer server')
    parser.add_argument('--fps', type=int, default=15, help='Target frames per second for webcam')
    parser.add_argument('--width', type=int, default=640, help='Width of webcam frames')
    parser.add_argument('--height', type=int, default=480, help='Height of webcam frames')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--test', action='store_true', help='Run in test pattern mode without webcam')
    args = parser.parse_args()
    
    # URL construction
    server_base = args.server_url.rstrip('/')
    webcam_url = f"{server_base}/webcam"
    processed_url = f"{server_base}/processed.mjpeg"
    
    # Start the viewer server in a separate thread
    threading.Thread(target=run_viewer_server, args=(args.viewer_port,), daemon=True).start()
    print(f"Browser viewer started at http://localhost:{args.viewer_port}/")
    
    # Start with a placeholder frame
    with frame_lock:
        stream_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(stream_frame, "Connecting to server...", (150, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    if args.test:
        print("Running in test pattern mode")
        # Start test pattern generator in a separate thread
        threading.Thread(target=generate_test_pattern, daemon=True).start()
    else:
        # Check server connectivity
        print(f"Testing connection to server: {server_base}")
        try:
            response = requests.head(server_base, timeout=5)
            print(f"Server returned status code: {response.status_code}")
            
            if response.status_code == 200:
                print("Server connection successful!")
                
                # Start webcam streaming thread
                print(f"Starting webcam stream to {webcam_url}")
                webcam_thread = threading.Thread(
                    target=stream_webcam,
                    args=(webcam_url, args.fps, args.debug, args.width, args.height),
                    daemon=True
                )
                webcam_thread.start()
                
                # Start processed frame fetching thread
                print(f"Starting to fetch processed frames from {processed_url}")
                threading.Thread(
                    target=fetch_mjpeg_stream,
                    args=(processed_url, args.debug),
                    daemon=True
                ).start()
            else:
                print(f"Server returned unexpected status code: {response.status_code}")
                print("Falling back to test pattern...")
                threading.Thread(target=generate_test_pattern, daemon=True).start()
        except Exception as e:
            print(f"Error connecting to server: {e}")
            print("Falling back to test pattern...")
            threading.Thread(target=generate_test_pattern, daemon=True).start()
    
    # Keep the main thread running
    try:
        print("\nSystem running - Press Ctrl+C to stop")
        while connection_active:
            time.sleep(1)
    except KeyboardInterrupt:
        # Clean up
        connection_active = False
        print("\nExiting...")

if __name__ == "__main__":
    main() 