#!/usr/bin/env python3
"""
Simple webcam streaming server for MacOS to make webcam accessible to Docker containers.
Run this on your MacBook to stream the webcam to the Docker container.
"""

import cv2
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import time

class WebcamStreamHandler(BaseHTTPRequestHandler):
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
                
                print("Webcam streaming started. Local preview window opened.")
                print("Press 'q' in the preview window to stop streaming.")
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Warning: Failed to grab frame")
                        break
                    
                    # Show preview
                    cv2.imshow('MacOS Webcam Preview', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
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
                cv2.destroyAllWindows()
                print("Webcam streaming stopped.")
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><head></head><body><h1>404 Not Found</h1></body></html>')

def run_server():
    server_address = ('', 8081)
    httpd = HTTPServer(server_address, WebcamStreamHandler)
    print(f"Starting webcam stream server on port 8081")
    print(f"Stream URL: http://localhost:8081/video.mjpeg")
    print("Press Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Server shutting down...")
        httpd.server_close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_server() 