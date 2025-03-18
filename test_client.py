#!/usr/bin/env python3
"""
Simple test client to check connectivity with the remote server
"""

import requests
import argparse
import socket
import time

def get_local_ip():
    """Get the local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 1))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "unknown"

def test_server(host, port=8082):
    """Test connection to the server"""
    url = f"http://{host}:{port}"
    local_ip = get_local_ip()
    
    print(f"Testing connection to {url}")
    print(f"Your local IP is: {local_ip}")
    print()
    
    # Try to connect to the server
    try:
        response = requests.get(url, timeout=5)
        print(f"Connection successful!")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return True
    except requests.exceptions.ConnectionError:
        print(f"Connection error: Could not connect to {url}")
        print("Possible reasons:")
        print("1. The server is not running")
        print("2. A firewall is blocking the connection")
        print("3. The IP address or port is incorrect")
        print()
        print("Next steps:")
        print(f"1. On the server, make sure it's running: python test_server.py")
        print(f"2. On the server, check if the port is open: netstat -tuln | grep {port}")
        print(f"3. Try to ping the server: ping {host}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test connection to the remote server")
    parser.add_argument("host", help="Remote host IP address")
    parser.add_argument("--port", type=int, default=8082, help="Remote host port (default: 8082)")
    args = parser.parse_args()
    
    success = test_server(args.host, args.port)
    
    if not success:
        print("\nTrying to connect 5 more times with 2-second intervals...")
        for i in range(5):
            print(f"\nAttempt {i+1}/5...")
            time.sleep(2)
            if test_server(args.host, args.port):
                break 