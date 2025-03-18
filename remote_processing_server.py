#!/usr/bin/env python3
"""
Remote processing server for object detection.
This script runs on the remote Linux host to:
1. Receive webcam stream from the MacBook
2. Process frames with object detection
3. Stream the processed frames back to the MacBook
"""

import cv2
import numpy as np
import argparse
import threading
import time
import sys
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.request
import socket

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from our local modules
from webcam_tracking.human_tracker import ObjectTracker
from webcam_tracking.human_tracker_yolo import YOLOObjectTracker
from core.yolov8_wrapper import YOLOv8Wrapper

# Global variables
latest_frame = None
latest_frame_lock = threading.Lock()
processed_frame = None
processed_frame_lock = threading.Lock()
processing_active = True
placeholder_frame = None  # Placeholder frame for when no input is available

class ProcessedStreamHandler(BaseHTTPRequestHandler):
    """Handle HTTP requests for the processed stream"""
    
    def do_HEAD(self):
        """Handle HEAD requests to check if server is running"""
        if self.path == '/processed.mjpeg':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_GET(self):
        if self.path == '/processed.mjpeg':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            
            try:
                while processing_active:
                    # Get the latest processed frame with lock
                    with processed_frame_lock:
                        frame = processed_frame.copy() if processed_frame is not None else placeholder_frame.copy()
                    
                    if frame is not None:
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
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><head></head><body><h1>404 Not Found</h1></body></html>')
    
    # Disable request logging to console
    def log_message(self, format, *args):
        return

def run_processed_stream_server(port=8082):
    """Run the server that streams processed frames back to the client"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, ProcessedStreamHandler)
    print(f"Starting processed stream server on port {port}")
    
    # Run the server in a separate thread
    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()
    return httpd

def receive_webcam_stream(client_url):
    """Receive webcam stream from the client (MacBook)"""
    global latest_frame, processing_active
    
    print(f"Connecting to webcam stream at {client_url}")
    print("Waiting for client to start streaming...")
    
    # Try to connect to the client's webcam stream
    connected = False
    retry_count = 0
    max_retries = 20  # More retries, as the client might take time to start
    
    while not connected and retry_count < max_retries and processing_active:
        try:
            # Test connection first
            test_stream = urllib.request.urlopen(client_url, timeout=2)
            if test_stream.getcode() == 200:
                connected = True
                print("Connected to client webcam stream!")
                test_stream.close()  # Close the test connection
            else:
                print(f"Client responded with status code: {test_stream.getcode()}")
                retry_count += 1
                time.sleep(1)
        except urllib.error.URLError as e:
            print(f"Connection failed (attempt {retry_count+1}/{max_retries}). Is the client streaming?")
            retry_count += 1
            time.sleep(1)
        except Exception as e:
            print(f"Error testing connection: {e}")
            retry_count += 1
            time.sleep(1)
    
    if not connected:
        print("Failed to connect to client webcam stream after multiple attempts.")
        print("Please make sure the client is running and streaming from its webcam.")
        processing_active = False
        return
    
    try:
        # Create a buffer to store the multipart stream
        stream_buffer = bytes()
        boundary = b"--jpgboundary"
        
        # Open the stream
        stream = urllib.request.urlopen(client_url)
        
        # Read stream in chunks
        while processing_active:
            try:
                chunk = stream.read(1024)
                if not chunk:
                    print("End of stream detected")
                    break
                
                # Add the chunk to our buffer
                stream_buffer += chunk
                
                # Look for the boundary and process frames
                if boundary in stream_buffer:
                    parts = stream_buffer.split(boundary)
                    
                    # Process complete parts
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
                                            # Update the global frame with lock
                                            with latest_frame_lock:
                                                latest_frame = img
                                    except Exception as e:
                                        print(f"Error decoding image: {e}")
                        
                        # Keep the last part which might be incomplete
                        stream_buffer = parts[-1]
            except (urllib.error.URLError, ConnectionResetError, socket.timeout) as e:
                print(f"Stream connection error: {e}")
                print("Trying to reconnect...")
                try:
                    stream.close()
                except:
                    pass
                try:
                    stream = urllib.request.urlopen(client_url)
                    print("Reconnected to stream!")
                except Exception as re:
                    print(f"Failed to reconnect: {re}")
                    break
                
    except Exception as e:
        print(f"Error receiving webcam stream: {e}")
    finally:
        print("Webcam stream receiver stopped")
        try:
            stream.close()
        except:
            pass
        
def process_frames(model, confidence=0.5, target_classes=None, yolo_size='m', force_yolo=None, nms=0.4):
    """Process frames with object detection"""
    global latest_frame, processed_frame, placeholder_frame
    
    # Create a placeholder frame
    placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder_frame, "Waiting for webcam stream...", (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Set initial processed frame to placeholder
    with processed_frame_lock:
        processed_frame = placeholder_frame.copy()
    
    print("Initializing object detection model...")
    
    # Initialize the appropriate tracker based on model
    try:
        if model == 'yolov8':
            # Map command-line args to wrapper implementation options
            yolo_impl = force_yolo if force_yolo else 'yolov8core'
            tracker = YOLOv8Wrapper(
                confidence_threshold=confidence,
                target_classes=target_classes,
                model_size=yolo_size,
                nms_threshold=nms,
                implementation=yolo_impl
            )
        elif model == 'yolo':
            # Initialize YOLOv4 tracker
            tracker = YOLOObjectTracker(
                confidence_threshold=confidence,
                target_classes=target_classes,
                nms_threshold=nms
            )
        else:  # mobilenet (default)
            # Initialize MobileNet SSD tracker
            tracker = ObjectTracker(
                confidence_threshold=confidence,
                target_classes=target_classes
            )
            
        print(f"Model initialized: {model}")
    except Exception as e:
        print(f"Error initializing model: {e}")
        # Update placeholder to show error
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, f"Model initialization error: {e}", (20, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        with processed_frame_lock:
            processed_frame = error_frame
        return
    
    # Process frames
    frame_count = 0
    start_time = time.time()
    fps_display_time = start_time
    
    while processing_active:
        # Get the latest frame with lock
        with latest_frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        
        if frame is not None:
            # Process the frame with the tracker
            try:
                if model == 'yolov8':
                    processed, _ = tracker.detect_objects(frame)
                else:
                    processed = tracker.detect_objects(frame)
                
                # Calculate FPS
                frame_count += 1
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # Update FPS every second
                if current_time - fps_display_time >= 1.0:
                    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                    cv2.putText(processed, f"FPS: {fps:.2f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    print(f"Processing FPS: {fps:.2f}", end='\r')
                    fps_display_time = current_time
                
                # Update the processed frame with lock
                with processed_frame_lock:
                    processed_frame = processed
            except Exception as e:
                print(f"Error processing frame: {e}")
                # Create an error frame
                error_frame = frame.copy()
                cv2.putText(error_frame, f"Processing error: {e}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                with processed_frame_lock:
                    processed_frame = error_frame
        else:
            # If no frame is available, show the placeholder
            with processed_frame_lock:
                processed_frame = placeholder_frame.copy()
                
        # Small delay to reduce CPU usage
        time.sleep(0.01)
        
    # Print final stats
    if frame_count > 0:
        total_time = time.time() - start_time
        average_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\nProcessed {frame_count} frames in {total_time:.2f} seconds ({average_fps:.2f} fps)")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Remote processing server for object detection')
    
    # Client connection parameters
    parser.add_argument('--client-host', type=str, required=True, 
                        help='IP address or hostname of the client (MacBook)')
    parser.add_argument('--client-port', type=int, default=8081, 
                        help='Port of the webcam stream on the client (default: 8081)')
    parser.add_argument('--server-port', type=int, default=8082, 
                        help='Port to serve processed stream (default: 8082)')
    
    # The model parameters
    parser.add_argument('--model', type=str, choices=['mobilenet', 'yolo', 'yolov8'], default='yolov8',
                        help='Model to use: mobilenet, yolo, or yolov8 (default: yolov8)')
    parser.add_argument('--confidence', type=float, default=0.5, 
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--nms', type=float, default=0.4, 
                        help='Non-maximum suppression threshold for YOLO (default: 0.4)')
    parser.add_argument('--classes', type=str, 
                        help='Comma-separated list of classes to detect (default: all classes)')
    
    # YOLOv8 specific arguments
    parser.add_argument('--yolo-size', type=str, choices=['n', 's', 'm', 'l', 'x'], default='m',
                        help='YOLOv8 model size: n(ano), s(mall), m(edium), l(arge), or x (default: m)')
    parser.add_argument('--force-yolo', type=str, choices=['ultralytics', 'onnx', 'yolov4-tiny', 'yolov8core'], 
                        help='Force a specific YOLOv8 implementation')
    
    args = parser.parse_args()
    
    # Parse target classes if specified
    target_classes = None
    if args.classes:
        target_classes = [cls.strip() for cls in args.classes.split(',')]
    
    # Client URL for webcam stream
    client_url = f"http://{args.client_host}:{args.client_port}/video.mjpeg"
    
    # Get all IP addresses of this host
    host_ips = []
    try:
        hostname = socket.gethostname()
        host_ips.append(socket.gethostbyname(hostname))
        
        # Get all network interfaces
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 1))  # connect to a public IP
        host_ips.append(s.getsockname()[0])  # get the outgoing IP
        s.close()
    except:
        pass
    
    print("\n====== Remote Object Detection Server ======")
    print(f"Server running on port: {args.server_port}")
    print(f"Server IP addresses: {', '.join(list(set(host_ips)))}")
    print(f"Expecting client connection from: {args.client_host}:{args.client_port}")
    print("=============================================\n")
    
    # Start the processed stream server
    httpd = run_processed_stream_server(args.server_port)
    
    # Start processing frames in a separate thread (start this first to initialize the model)
    process_thread = threading.Thread(
        target=process_frames,
        args=(args.model,),
        kwargs={
            'confidence': args.confidence,
            'target_classes': target_classes,
            'yolo_size': args.yolo_size,
            'force_yolo': args.force_yolo,
            'nms': args.nms
        }
    )
    process_thread.daemon = True
    process_thread.start()
    
    # Start receiving webcam stream in a separate thread
    receive_thread = threading.Thread(target=receive_webcam_stream, args=(client_url,))
    receive_thread.daemon = True
    receive_thread.start()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        global processing_active
        processing_active = False
        time.sleep(1)  # Give threads time to clean up
        httpd.shutdown()
        print("Done")

if __name__ == "__main__":
    main() 