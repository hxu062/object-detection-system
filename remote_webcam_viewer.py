#!/usr/bin/env python3
"""
Remote webcam streaming and viewer for object detection system.
This script runs on the MacBook to:
1. Stream the MacBook webcam to the remote host
2. Receive processed frames from the remote host
3. Display the results live on the MacBook
"""

import cv2
import socket
import threading
import time
import numpy as np
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import requests
from io import BytesIO

# Global variables for the streams
processed_frame = None
processed_frame_lock = threading.Lock()

class WebcamStreamHandler(BaseHTTPRequestHandler):
    """Handle HTTP requests for the webcam stream"""
    
    def do_GET(self):
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
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Warning: Failed to grab frame")
                        break
                    
                    # Encode frame as JPEG
                    ret, jpg = cv2.imencode('.jpg', frame)
                    
                    # Write the JPEG to the stream
                    self.wfile.write(b"--jpgboundary\r\n")
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', str(len(jpg)))
                    self.end_headers()
                    self.wfile.write(jpg.tobytes())
                    self.wfile.write(b"\r\n")
                    
                    # Small delay to prevent overwhelming the network
                    time.sleep(0.05)
                    
            except Exception as e:
                print(f"Streaming error: {e}")
            finally:
                if 'cap' in locals() and cap.isOpened():
                    cap.release()
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><head></head><body><h1>404 Not Found</h1></body></html>')
    
    # Disable request logging to console
    def log_message(self, format, *args):
        return

def run_webcam_server(port=8081):
    """Run the webcam streaming server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, WebcamStreamHandler)
    print(f"Starting webcam stream server on port {port}")
    print(f"Stream URL: http://localhost:{port}/video.mjpeg")
    
    # Run the server in a separate thread
    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()
    return httpd

def receive_processed_stream(remote_host, remote_port=8082):
    """Receive and display the processed video stream from the remote host"""
    url = f"http://{remote_host}:{remote_port}/processed.mjpeg"
    
    print(f"Connecting to processed stream at {url}")
    
    try:
        # Set up the window to display the processed stream
        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
        
        # Create a buffer to store the multipart stream
        stream_buffer = bytes()
        boundary = b"--jpgboundary"
        
        # Make a streaming request
        response = requests.get(url, stream=True, timeout=10)
        
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code} from remote server")
            return
        
        # Process the multipart stream
        for chunk in response.iter_content(chunk_size=1024):
            if not chunk:
                continue
            
            # Add the chunk to our buffer
            stream_buffer += chunk
            
            # Look for the boundary and process frames
            if boundary in stream_buffer:
                parts = stream_buffer.split(boundary)
                
                # Keep the last part which might be incomplete
                if len(parts) > 1:
                    for part in parts[:-1]:
                        # Look for the JPEG data in each part
                        jpg_start = part.find(b'\r\n\r\n') + 4
                        if jpg_start > 4:
                            jpg_data = part[jpg_start:]
                            if jpg_data:
                                # Convert bytes to numpy array
                                try:
                                    nparr = np.frombuffer(jpg_data, np.uint8)
                                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                    if img is not None:
                                        # Update the global frame with lock to avoid race conditions
                                        with processed_frame_lock:
                                            global processed_frame
                                            processed_frame = img
                                except Exception as e:
                                    print(f"Error decoding image: {e}")
                    
                    # Keep the last part which might be incomplete
                    stream_buffer = parts[-1]
                
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to remote server: {e}")
    except Exception as e:
        print(f"Error in receive_processed_stream: {e}")

def display_results():
    """Display the processed frames"""
    try:
        while True:
            # Get the latest processed frame with lock
            with processed_frame_lock:
                frame = processed_frame.copy() if processed_frame is not None else None
            
            if frame is not None:
                cv2.imshow("Object Detection", frame)
            
            # Check for exit key (q)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Slight delay to reduce CPU usage
            time.sleep(0.01)
    except Exception as e:
        print(f"Error in display_results: {e}")
    finally:
        cv2.destroyAllWindows()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Remote webcam streaming and viewing")
    parser.add_argument("remote_host", help="IP address or hostname of the remote processing server")
    parser.add_argument("--webcam-port", type=int, default=8081, help="Port for webcam streaming server (default: 8081)")
    parser.add_argument("--remote-port", type=int, default=8082, help="Port for receiving processed stream (default: 8082)")
    args = parser.parse_args()
    
    # Start the webcam streaming server
    httpd = run_webcam_server(args.webcam_port)
    
    # Start receiving processed stream in a separate thread
    receive_thread = threading.Thread(target=receive_processed_stream, args=(args.remote_host, args.remote_port))
    receive_thread.daemon = True
    receive_thread.start()
    
    try:
        # Display the results
        display_results()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        print("Shutting down webcam server...")
        httpd.shutdown()
        print("Done")

if __name__ == "__main__":
    main() 