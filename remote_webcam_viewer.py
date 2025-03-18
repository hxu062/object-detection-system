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
import sys

# Global variables for the streams
processed_frame = None
processed_frame_lock = threading.Lock()
connection_active = True

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
                
                while connection_active:
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
    global connection_active
    url = f"http://{remote_host}:{remote_port}/processed.mjpeg"
    
    print(f"Connecting to processed stream at {url}")
    print("Waiting for remote server to be ready...")
    
    # Try to connect to the server
    connected = False
    retry_count = 0
    max_retries = 10
    
    while not connected and retry_count < max_retries:
        try:
            # Test connection first
            test_response = requests.head(url, timeout=2)
            if test_response.status_code == 200:
                connected = True
                print("Connected to remote server!")
            else:
                print(f"Server responded with status code: {test_response.status_code}")
                retry_count += 1
                time.sleep(2)
        except requests.exceptions.ConnectionError:
            print(f"Connection failed (attempt {retry_count+1}/{max_retries}). Is the remote server running?")
            retry_count += 1
            time.sleep(2)
        except Exception as e:
            print(f"Error testing connection: {e}")
            retry_count += 1
            time.sleep(2)
    
    if not connected:
        print("Failed to connect to remote server after multiple attempts.")
        print("Please ensure the remote server is running with:")
        print(f"  python remote_processing_server.py --client-host YOUR_MACBOOK_IP --model yolov8")
        connection_active = False
        return
    
    try:
        # Set up the window to display the processed stream
        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
        
        # Create a buffer to store the multipart stream
        stream_buffer = bytes()
        boundary = b"--jpgboundary"
        
        # Make a streaming request
        response = requests.get(url, stream=True, timeout=30)
        
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code} from remote server")
            connection_active = False
            return
        
        # Process the multipart stream
        for chunk in response.iter_content(chunk_size=1024):
            if not chunk or not connection_active:
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
        print("Please check if the server is running and the IP is correct.")
    except Exception as e:
        print(f"Error in receive_processed_stream: {e}")
    finally:
        connection_active = False
        print("Stream connection closed")

def display_results():
    """Display the processed frames"""
    global connection_active
    
    try:
        # Create a blank frame to show initially
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        text = "Waiting for connection to remote server..."
        cv2.putText(blank_frame, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Object Detection", blank_frame)
        cv2.waitKey(1)
        
        last_update = time.time()
        frame_count = 0
        
        while connection_active:
            # Get the latest processed frame with lock
            with processed_frame_lock:
                frame = processed_frame.copy() if processed_frame is not None else None
            
            if frame is not None:
                # Calculate FPS
                frame_count += 1
                current_time = time.time()
                if current_time - last_update >= 1.0:
                    fps = frame_count / (current_time - last_update)
                    cv2.putText(frame, f"Display FPS: {fps:.1f}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    frame_count = 0
                    last_update = current_time
                
                cv2.imshow("Object Detection", frame)
            else:
                # Show waiting message if no frame is available
                cv2.imshow("Object Detection", blank_frame)
            
            # Check for exit key (q)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                connection_active = False
                break
            
            # Slight delay to reduce CPU usage
            time.sleep(0.01)
    except Exception as e:
        print(f"Error in display_results: {e}")
    finally:
        cv2.destroyAllWindows()
        print("Display closed")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Remote webcam streaming and viewing")
    parser.add_argument("remote_host", help="IP address or hostname of the remote processing server")
    parser.add_argument("--webcam-port", type=int, default=8081, help="Port for webcam streaming server (default: 8081)")
    parser.add_argument("--remote-port", type=int, default=8082, help="Port for receiving processed stream (default: 8082)")
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
    
    print("\n====== Remote Object Detection Client ======")
    print(f"Your MacBook's IP address is: {local_ip}")
    print("Make sure the remote server is running with:")
    print(f"  python remote_processing_server.py --client-host {local_ip} --model yolov8")
    print("=============================================\n")
    
    # Start the webcam streaming server
    httpd = run_webcam_server(args.webcam_port)
    
    # Start the display thread first
    display_thread = threading.Thread(target=display_results)
    display_thread.daemon = True
    display_thread.start()
    
    # Start receiving processed stream in a separate thread
    receive_thread = threading.Thread(target=receive_processed_stream, args=(args.remote_host, args.remote_port))
    receive_thread.daemon = True
    receive_thread.start()
    
    try:
        # Keep the main thread alive until user interrupts
        while connection_active:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        connection_active = False
        print("Shutting down webcam server...")
        httpd.shutdown()
        time.sleep(1)
        print("Done")

if __name__ == "__main__":
    main() 