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
import os
import traceback

# Set environment variables to help with OpenCV display issues
os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Try offscreen rendering first

# Global variables for the streams
processed_frame = None
processed_frame_lock = threading.Lock()
connection_active = False  # Initialize to False and set to True in main()

# Track frame stats
frames_received = 0
decode_errors = 0
last_frame_time = 0

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
                        time.sleep(0.1)  # Add delay to avoid busy loop
                        continue
                    
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
    global connection_active, frames_received, decode_errors, last_frame_time
    url = f"http://{remote_host}:{remote_port}/processed.mjpeg"
    
    print(f"Connecting to processed stream at {url}")
    print("Waiting for remote server to be ready...")
    
    # Try to connect to the server
    connected = False
    retry_count = 0
    max_retries = 10
    
    while not connected and retry_count < max_retries and connection_active:
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
        # Create a buffer to store the multipart stream
        stream_buffer = bytes()
        boundary = b"--jpgboundary"
        
        # Make a streaming request
        response = requests.get(url, stream=True, timeout=30)
        
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code} from remote server")
            connection_active = False
            return
        
        print("Stream connected, waiting for frames...")
        
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
                                try:
                                    # Convert bytes to numpy array
                                    nparr = np.frombuffer(jpg_data, np.uint8)
                                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                    
                                    if img is not None and img.size > 0:
                                        frames_received += 1
                                        last_frame_time = time.time()
                                        
                                        # Update the global frame with lock to avoid race conditions
                                        with processed_frame_lock:
                                            global processed_frame
                                            processed_frame = img.copy()  # Make a copy to avoid reference issues
                                    else:
                                        decode_errors += 1
                                        print(f"Warning: Received invalid image data (error #{decode_errors})")
                                except Exception as e:
                                    decode_errors += 1
                                    print(f"Error decoding image: {e} (error #{decode_errors})")
                                    if decode_errors % 10 == 0:  # Print stack trace every 10 errors
                                        print(traceback.format_exc())
                    
                    # Keep the last part which might be incomplete
                    stream_buffer = parts[-1]
                
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to remote server: {e}")
        print("Please check if the server is running and the IP is correct.")
    except Exception as e:
        print(f"Error in receive_processed_stream: {e}")
        print(traceback.format_exc())
    finally:
        connection_active = False
        print("Stream connection closed")

def display_results():
    """Display the processed frames"""
    global connection_active, frames_received, decode_errors, last_frame_time
    
    try:
        # Create a blank frame to show initially
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        text = "Waiting for connection to remote server..."
        cv2.putText(blank_frame, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Try different UI backends if one fails
        window_created = False
        
        for backend in [cv2.WINDOW_NORMAL, cv2.WINDOW_AUTOSIZE]:
            try:
                cv2.namedWindow("Object Detection", backend)
                cv2.imshow("Object Detection", blank_frame)
                cv2.waitKey(1)
                window_created = True
                print("Created display window")
                break
            except Exception as e:
                print(f"Failed to create window with backend {backend}: {e}")
        
        if not window_created:
            print("ERROR: Could not create any display window. Running in headless mode.")
            # Keep running to support streaming, just don't display
        
        last_update = time.time()
        frame_count = 0
        stats_update_time = time.time()
        
        while connection_active:
            # Update stats periodically
            if time.time() - stats_update_time >= 5.0:
                print(f"Stats: {frames_received} frames received, {decode_errors} decode errors")
                stats_update_time = time.time()
                
                # Check for stalled stream
                if frames_received > 0 and time.time() - last_frame_time > 10.0:
                    print("Warning: No frames received in 10 seconds, stream may be stalled")
            
            # Get the latest processed frame with lock
            current_frame = None
            with processed_frame_lock:
                if processed_frame is not None:
                    current_frame = processed_frame.copy()
            
            if current_frame is not None and window_created:
                # Calculate FPS
                frame_count += 1
                current_time = time.time()
                if current_time - last_update >= 1.0:
                    fps = frame_count / (current_time - last_update)
                    last_update = current_time
                    frame_count = 0
                    
                    # Add stats to the frame
                    cv2.putText(current_frame, f"Display FPS: {fps:.1f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(current_frame, f"Frames: {frames_received}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(current_frame, f"Errors: {decode_errors}", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                try:
                    cv2.imshow("Object Detection", current_frame)
                except Exception as e:
                    print(f"Error displaying frame: {e}")
            elif window_created:
                # Show waiting message if no frame is available
                try:
                    cv2.imshow("Object Detection", blank_frame)
                except Exception as e:
                    print(f"Error displaying blank frame: {e}")
            
            # Check for exit key (q) - wrap in try/except to handle headless mode
            try:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    connection_active = False
                    break
            except Exception:
                # In headless mode, just sleep a bit
                time.sleep(0.1)
                pass
            
            # Slight delay to reduce CPU usage
            time.sleep(0.01)
    except Exception as e:
        print(f"Error in display_results: {e}")
        print(traceback.format_exc())
    finally:
        if window_created:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("Display closed")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Remote webcam streaming and viewing")
    parser.add_argument("remote_host", help="IP address or hostname of the remote processing server")
    parser.add_argument("--webcam-port", type=int, default=8081, help="Port for webcam streaming server (default: 8081)")
    parser.add_argument("--remote-port", type=int, default=8082, help="Port for receiving processed stream (default: 8082)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no display)")
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
    
    # Set connection active before starting threads
    global connection_active
    connection_active = True
    
    # Start the webcam streaming server
    httpd = run_webcam_server(args.webcam_port)
    
    threads = []
    
    # Start receiving processed stream in a separate thread
    receive_thread = threading.Thread(target=receive_processed_stream, args=(args.remote_host, args.remote_port))
    receive_thread.daemon = True
    receive_thread.start()
    threads.append(receive_thread)
    
    # Start the display thread if not in headless mode
    if not args.headless:
        display_thread = threading.Thread(target=display_results)
        display_thread.daemon = True
        display_thread.start()
        threads.append(display_thread)
    else:
        print("Running in headless mode (no display)")
    
    try:
        # Keep the main thread alive until user interrupts
        while connection_active:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        connection_active = False
        print("Shutting down...")
        
        # Wait for threads to finish
        for thread in threads:
            thread.join(timeout=1.0)
        
        httpd.shutdown()
        time.sleep(1)
        print("Done")

if __name__ == "__main__":
    main() 