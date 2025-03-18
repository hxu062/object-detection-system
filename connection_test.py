#!/usr/bin/env python3
"""
Simple connectivity test for MJPEG stream
"""

import argparse
import requests
import time
import socket
import sys

def test_tcp_connection(host, port, timeout=5):
    """Test basic TCP connection to host:port"""
    print(f"Testing TCP connection to {host}:{port}...")
    
    start_time = time.time()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    
    try:
        sock.connect((host, port))
        elapsed = time.time() - start_time
        print(f"✅ TCP connection successful! ({elapsed:.2f}s)")
        return True
    except socket.timeout:
        print(f"❌ Connection timed out after {timeout}s")
        return False
    except ConnectionRefusedError:
        print(f"❌ Connection refused - no service running on {host}:{port}")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {type(e).__name__}: {e}")
        return False
    finally:
        sock.close()

def test_http_connection(url, timeout=5):
    """Test HTTP GET connection to URL"""
    print(f"Testing HTTP connection to {url}...")
    
    start_time = time.time()
    try:
        response = requests.head(url, timeout=timeout)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✅ HTTP connection successful! Status: {response.status_code} ({elapsed:.2f}s)")
            print(f"   Content-Type: {response.headers.get('Content-Type', 'unknown')}")
            return True
        else:
            print(f"⚠️ HTTP connection succeeded but returned status {response.status_code}")
            print(f"   Response headers: {dict(response.headers)}")
            return False
    except requests.exceptions.Timeout:
        print(f"❌ HTTP request timed out after {timeout}s")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ HTTP connection error: {e}")
        return False
    except Exception as e:
        print(f"❌ HTTP request failed: {type(e).__name__}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test connectivity to a remote server')
    parser.add_argument('url', help='URL to test (e.g., http://100.91.32.59:8082/processed.mjpeg)')
    parser.add_argument('--timeout', type=int, default=5, help='Connection timeout in seconds')
    args = parser.parse_args()
    
    # Parse the URL to get host and port
    if not args.url.startswith('http'):
        print("Error: URL must start with http:// or https://")
        sys.exit(1)
    
    try:
        # Extract host and port from URL
        from urllib.parse import urlparse
        parsed_url = urlparse(args.url)
        
        # Get host and port
        host = parsed_url.hostname
        port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)
        
        print(f"=== Connection Test to {host}:{port} ===")
        print(f"Target URL: {args.url}")
        print(f"Timeout: {args.timeout}s")
        print("=" * 50)
        
        # Test TCP first
        tcp_ok = test_tcp_connection(host, port, args.timeout)
        print("-" * 50)
        
        # Then test HTTP if TCP succeeded
        if tcp_ok:
            http_ok = test_http_connection(args.url, args.timeout)
            print("-" * 50)
            
            if http_ok:
                print("✅ All tests PASSED! You should be able to connect to the stream.")
            else:
                print("⚠️ TCP connection succeeded but HTTP test failed.")
                print("   The server might be running but not serving the expected content.")
        else:
            print("❌ Failed to establish TCP connection.")
            print("   Possible issues:")
            print("   - Server might be down")
            print("   - Firewall blocking the connection")
            print("   - Incorrect IP or port")
            print("   - Network issue between client and server")
    
    except Exception as e:
        print(f"Error parsing URL: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 