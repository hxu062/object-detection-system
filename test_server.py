#!/usr/bin/env python3
"""
Simple HTTP server for testing network connectivity
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
import socket
import argparse

class SimpleHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        message = "Hello from the test server!"
        self.wfile.write(bytes(message, "utf8"))
    
    def log_message(self, format, *args):
        print(f"Request from {self.client_address[0]}: {format % args}")

def run_server(port=8082, host='0.0.0.0'):
    server_address = (host, port)
    httpd = HTTPServer(server_address, SimpleHandler)
    
    # Get all IP addresses
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "unknown"
    
    # Get all network interfaces
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 1))
        outgoing_ip = s.getsockname()[0]
    except:
        outgoing_ip = "unknown"
    finally:
        s.close()
    
    print(f"Starting test server on {host}:{port}")
    print(f"Local hostname: {hostname}")
    print(f"Local IP: {local_ip}")
    print(f"Outgoing IP: {outgoing_ip}")
    print("To test from your MacBook, run:")
    print(f"  curl http://{outgoing_ip}:{port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple HTTP server for testing')
    parser.add_argument('--port', type=int, default=8082, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()
    
    run_server(args.port, args.host) 