#!/usr/bin/env python3
"""
Direct Stream Connector
Simplified script to directly connect to the MJPEG stream from the server
and display it in a browser with minimal complexity.
"""

import cv2
import numpy as np
import time
import threading
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
import os
import argparse
import traceback
import requests
import math
import json  # Import json at the top level

# Global variables
stream_frame = None
frame_lock = threading.Lock()
frames_received = 0
connection_active = True
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
                <title>Direct Stream Viewer</title>
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
                <h1>Direct Stream Viewer</h1>
                <div id="videoContainer">
                    <img id="streamImage" src="/placeholder.jpg" alt="Stream">
                </div>
                <div id="stats">Frames received: 0</div>
                <button class="button" id="refreshBtn">Manual Refresh</button>
                <button class="button" id="autoBtn">Start Auto-Refresh</button>
                
                <script>
                    let autoRefresh = false;
                    let refreshInterval;
                    const refreshRate = 200; // ms between refreshes (5 FPS)
                    
                    function updateImage() {
                        document.getElementById('streamImage').src = 
                            '/frame.jpg?t=' + new Date().getTime();
                    }
                    
                    function updateStats() {
                        fetch('/stats.json?t=' + new Date().getTime())
                            .then(response => response.json())
                            .then(data => {
                                document.getElementById('stats').innerText = 
                                    `Frames received: ${data.frames_received}`;
                            });
                    }
                    
                    function startAutoRefresh() {
                        if (!autoRefresh) {
                            autoRefresh = true;
                            document.getElementById('autoBtn').innerText = "Stop Auto-Refresh";
                            refreshInterval = setInterval(function() {
                                updateImage();
                                updateStats();
                            }, refreshRate);
                        } else {
                            autoRefresh = false;
                            document.getElementById('autoBtn').innerText = "Start Auto-Refresh";
                            clearInterval(refreshInterval);
                        }
                    }
                    
                    // Set up button listeners
                    document.getElementById('refreshBtn').addEventListener('click', function() {
                        updateImage();
                        updateStats();
                    });
                    
                    document.getElementById('autoBtn').addEventListener('click', function() {
                        startAutoRefresh();
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
                frame_count = frames_received  # Get atomic copy of frame count
                if stream_frame is not None and stream_frame.size > 0:
                    frame_to_send = stream_frame.copy()
                else:
                    # Create a placeholder frame if no frame is available
                    frame_to_send = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame_to_send, "No frame available", (180, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Always add frame counter to top left
            cv2.putText(frame_to_send, f"Frame: {frame_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
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
                
        # Serve a placeholder image
        elif self.path == '/placeholder.jpg':
            self.send_response(200)
            self.send_header('Content-type', 'image/jpeg')
            self.end_headers()
            
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "Connecting to stream...", (150, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            _, jpg_data = cv2.imencode('.jpg', img)
            self.wfile.write(jpg_data.tobytes())
            
        # Serve stats as JSON
        elif self.path.startswith('/stats.json'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            stats = {
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

def fetch_mjpeg_stream(url, debug=False):
    """Connect to an MJPEG stream and update the global frame"""
    global stream_frame, frames_received, connection_active
    
    print(f"Connecting to stream at {url}")
    
    # Initialize a timeout for receiving the first frame
    frame_timeout = 10  # seconds
    start_time = time.time()
    
    # Try a HEAD request first
    try:
        head_response = requests.head(url, timeout=5)
        print(f"HEAD response: {head_response.status_code}")
        
        if head_response.status_code != 200:
            print(f"Error: Server returned status code {head_response.status_code}")
            print("Falling back to test pattern mode...")
            generate_test_pattern()
            return
    except Exception as e:
        print(f"HEAD request error: {e}")
        print("Continuing anyway...")
    
    try:
        # Connect to the stream
        print(f"Making GET request to {url}...")
        response = requests.get(url, stream=True, timeout=30)
        
        if response.status_code != 200:
            print(f"Error: GET request failed with status {response.status_code}")
            print("Falling back to test pattern mode...")
            generate_test_pattern()
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
                # Save a snippet of the buffer for debugging
                with open("buffer_snippet.bin", "wb") as f:
                    f.write(buffer[:min(5000, len(buffer))])
            
            # Look for boundary markers
            if boundary_bytes in buffer:
                if debug:
                    print(f"Found boundary marker at position {buffer.find(boundary_bytes)}")
                
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
                                        
                                        # Reset timeout since we got a frame
                                        start_time = time.time()
                                        
                                        # Log progress
                                        if frames_received == 1 or frames_received % 10 == 0:
                                            print(f"Received frame {frames_received}, shape: {img.shape}")
                                    else:
                                        print(f"Warning: Unable to decode image, data size: {len(img_data)}")
                                        # Save the raw data for debugging
                                        if debug:
                                            with open(f"debug_frame_{frames_received}.bin", "wb") as f:
                                                f.write(img_data)
                                except Exception as e:
                                    print(f"Error decoding image: {e}")
                                    if debug:
                                        print(traceback.format_exc())
                    
                    # Keep the last part (may be incomplete)
                    buffer = parts[-1]
            
            # Check if we've timed out without receiving any frames
            if frames_received == 0 and time.time() - start_time > frame_timeout:
                print(f"Timeout after {frame_timeout} seconds without receiving any frames")
                print("Falling back to test pattern mode...")
                generate_test_pattern()
                return
            
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
        print("Falling back to test pattern mode...")
        generate_test_pattern()
    
    print("Stream connection closed")

def generate_test_pattern():
    """Generate a test pattern when the stream connection fails"""
    global stream_frame, pattern_start_time, pattern_count
    
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
            cv2.putText(frame, f"Frames: {pattern_count}", (50, height-50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Time: {elapsed_time:.1f}s", (width-200, height-50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
            # Update the frame with a lock to ensure thread safety
            with frame_lock:
                stream_frame = frame
                pattern_count += 1
            
            # Print status periodically
            if pattern_count % 30 == 0:
                print(f"Generated {pattern_count} test pattern frames")
                
            # Sleep to control frame rate
            time.sleep(1/30)  # Aim for 30fps
            
        except Exception as e:
            print(f"Error generating test pattern: {e}")
            time.sleep(1)  # Retry after 1 second

def main():
    """Main function"""
    global connection_active, stream_frame, frames_received, pattern_count, pattern_start_time
    
    # Initialize variables
    connection_active = True
    stream_frame = None
    frames_received = 0
    pattern_count = 0
    pattern_start_time = time.time()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MJPEG Stream Viewer')
    parser.add_argument('url', help='URL of the MJPEG stream', nargs='?')
    parser.add_argument('--port', type=int, default=9099, help='Port for the viewer server')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints')
    parser.add_argument('--test', action='store_true', help='Run in test pattern mode only')
    args = parser.parse_args()
    
    # Validate arguments
    if not args.test and not args.url:
        parser.error("URL is required unless --test is specified")
    
    # Start the viewer server in a separate thread
    threading.Thread(target=run_viewer_server, args=(args.port,), daemon=True).start()
    print(f"Viewer server started at http://localhost:{args.port}/")
    
    if args.test:
        print("Running in test pattern mode")
        # Create an initial test frame
        with frame_lock:
            stream_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(stream_frame, "TEST PATTERN STARTING", (150, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Start test pattern generator in a separate thread
        threading.Thread(target=generate_test_pattern, daemon=True).start()
        
        # Keep main thread alive
        try:
            while connection_active:
                time.sleep(1)
        except KeyboardInterrupt:
            connection_active = False
            print("Exiting...")
        return
    
    # Regular stream mode:
    print(f"Connecting to stream at {args.url}")
    
    # Start with a placeholder frame
    with frame_lock:
        stream_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(stream_frame, "Connecting to stream...", (150, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Start the test pattern thread to show if connection fails
    test_thread = threading.Thread(target=generate_test_pattern, daemon=True)
    
    # Check if we can connect to the server first
    try:
        response = requests.head(args.url, timeout=5)
        if response.status_code != 200:
            print(f"Server returned {response.status_code}, falling back to test pattern")
            test_thread.start()
        else:
            print("Server connection successful, starting stream fetch")
            # Start the stream fetching thread
            threading.Thread(target=fetch_mjpeg_stream, args=(args.url, args.debug), daemon=True).start()
            
            # Set a timeout for the first frame
            timeout = 10  # seconds
            start_time = time.time()
            
            # Wait for the first frame or timeout
            while frames_received == 0 and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            # If no frames received after timeout, start test pattern
            if frames_received == 0:
                print(f"No frames received after {timeout} seconds, falling back to test pattern")
                test_thread.start()
    except Exception as e:
        print(f"Error connecting to server: {e}")
        test_thread.start()
    
    # Keep the main thread running
    try:
        while connection_active:
            time.sleep(1)
            # Print status every 5 seconds
            if frames_received > 0 and frames_received % 50 == 0:
                print(f"Received {frames_received} frames so far")
    except KeyboardInterrupt:
        # Clean up
        connection_active = False
        print("Exiting...")

if __name__ == "__main__":
    main() 