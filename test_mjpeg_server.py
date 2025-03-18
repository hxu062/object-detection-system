#!/usr/bin/env python3
"""
Simple MJPEG streaming server for testing OpenCV client connections
"""
import cv2
import numpy as np
import time
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import argparse

class MJPEGStreamHandler(BaseHTTPRequestHandler):
    """HTTP request handler to serve MJPEG stream"""
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/processed.mjpeg':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            
            try:
                # Create a synthetic image (animated gradient)
                width, height = 640, 480
                frame_count = 0
                
                while True:
                    # Create test pattern image
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    
                    # Draw moving gradient based on time
                    t = time.time()
                    for y in range(height):
                        for x in range(width):
                            b = int(127.5 * (1 + np.sin(x/30 + t)))
                            g = int(127.5 * (1 + np.sin(y/30 + t)))
                            r = int(127.5 * (1 + np.sin((x+y)/60 + t)))
                            frame[y, x] = [b, g, r]
                    
                    # Draw frame counter
                    frame_count += 1
                    cv2.putText(frame, f"Frame: {frame_count}", (20, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    
                    # Add timestamp
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(frame, timestamp, (20, height-20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
                    # Encode frame as JPEG
                    ret, jpg = cv2.imencode('.jpg', frame)
                    if not ret:
                        print("Error encoding JPEG")
                        continue
                    
                    # Write the JPEG to the stream with proper MIME multipart format
                    self.wfile.write(b"--jpgboundary\r\n")
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', str(len(jpg)))
                    self.end_headers()
                    self.wfile.write(jpg.tobytes())
                    self.wfile.write(b"\r\n")
                    
                    # Add a small delay to control frame rate
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Stream error: {e}")
                return
        else:
            # Serve a simple info page
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><head><title>Test MJPEG Server</title></head><body>')
            self.wfile.write(b'<h1>Test MJPEG Streaming Server</h1>')
            self.wfile.write(b'<p>Access the MJPEG stream at: <a href="/processed.mjpeg">/processed.mjpeg</a></p>')
            self.wfile.write(b'</body></html>')
    
    # Disable logging to console
    def log_message(self, format, *args):
        return

def get_ip_address():
    """Get the local IP address"""
    try:
        # Get local IP by connecting to a public DNS
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def run_server(port):
    """Run the HTTP server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, MJPEGStreamHandler)
    
    ip = get_ip_address()
    print(f"\n=== Test MJPEG Server ===")
    print(f"Server running at:")
    print(f"  http://{ip}:{port}/processed.mjpeg")
    print("Press Ctrl+C to stop the server")
    print("===========================\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down...")
        httpd.server_close()
        print("Server stopped")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple MJPEG streaming server for testing")
    parser.add_argument("--port", type=int, default=8082, help="Port to listen on (default: 8082)")
    args = parser.parse_args()
    
    run_server(args.port) 