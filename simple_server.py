#!/usr/bin/env python3
"""
Simple HTTP server that processes webcam frames and streams them back
"""

import http.server
import socketserver
import cv2
import numpy as np
import io
import threading
import time
import argparse

# Global variables
latest_frame = None
frame_lock = threading.Lock()
processed_frame = None
processed_count = 0

# Create a placeholder frame
placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(placeholder, "Waiting for webcam...", (120, 240), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

class RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        global processed_frame, processed_count
        
        if self.path == '/processed.mjpeg':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            
            try:
                while True:
                    # Get current frame
                    with frame_lock:
                        if processed_frame is not None:
                            frame = processed_frame.copy()
                        else:
                            frame = placeholder.copy()
                    
                    # Add processed count
                    cv2.putText(frame, f"Processed: {processed_count}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Encode as JPEG
                    _, jpg = cv2.imencode('.jpg', frame)
                    
                    # Send frame
                    self.wfile.write(b"--jpgboundary\r\n")
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', str(len(jpg)))
                    self.end_headers()
                    self.wfile.write(jpg.tobytes())
                    self.wfile.write(b"\r\n")
                    
                    # Small delay
                    time.sleep(0.05)
            except Exception as e:
                print(f"Streaming error: {e}")
                return
                
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><body><h1>404 Not Found</h1></body></html>')
    
    def do_POST(self):
        global latest_frame, processed_count
        
        if self.path == '/webcam':
            content_length = int(self.headers['Content-Length'])
            
            # Read the data
            post_data = self.rfile.read(content_length)
            
            try:
                # Convert to numpy array
                nparr = np.frombuffer(post_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # Update the global frame
                    with frame_lock:
                        latest_frame = img.copy()
                    
                    # Process the frame (do object detection here)
                    process_frame()
                    
                    # Send response
                    self.send_response(200)
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b"OK")
                else:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Failed to decode image")
            except Exception as e:
                print(f"Error processing image: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    # Disable logging
    def log_message(self, format, *args):
        return

def process_frame():
    """Process the latest frame with simple object detection"""
    global latest_frame, processed_frame, processed_count
    
    with frame_lock:
        if latest_frame is not None:
            # Just copy the frame without edge detection processing
            processed = latest_frame.copy()
            
            # Add frame counter to show it's working
            cv2.putText(processed, f"Frame: {processed_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Add timestamp
            cv2.putText(processed, time.strftime("%H:%M:%S"), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Update processed frame and count
            processed_frame = processed
            processed_count += 1

def main():
    parser = argparse.ArgumentParser(description='Simple HTTP server for webcam processing')
    parser.add_argument('--port', type=int, default=8082, help='Port to serve on')
    args = parser.parse_args()
    
    # Create server
    handler = RequestHandler
    httpd = socketserver.ThreadingTCPServer(('0.0.0.0', args.port), handler)
    
    print(f"Starting server on port {args.port}")
    print(f"MJPEG stream available at: http://localhost:{args.port}/processed.mjpeg")
    print(f"Send webcam frames to: http://localhost:{args.port}/webcam")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped")
    finally:
        httpd.server_close()

if __name__ == "__main__":
    main() 