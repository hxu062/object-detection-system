#!/usr/bin/env python3
"""
Simple connectivity test for MJPEG stream
"""

import argparse
import requests
import time
import socket
import sys
import urllib.parse

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

def test_mjpeg_endpoint(url, timeout=5):
    """Test specific MJPEG endpoint with a short connection"""
    print(f"Testing MJPEG endpoint at {url}...")
    
    try:
        # Try to connect and get just a bit of data
        response = requests.get(url, stream=True, timeout=timeout)
        
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            print(f"✅ Connection to MJPEG endpoint successful!")
            print(f"   Content-Type: {content_type}")
            
            # Check for correct content type
            if 'multipart/x-mixed-replace' in content_type:
                print("✅ Content type is correct for MJPEG stream")
                
                # Try to read a small amount of data
                try:
                    # Read just a bit to see if the stream is working
                    chunk = next(response.iter_content(chunk_size=4096))
                    if chunk:
                        print(f"✅ Successfully read {len(chunk)} bytes from stream")
                        
                        # Check for boundary marker
                        if b'--jpgboundary' in chunk:
                            print("✅ Found boundary marker in data")
                        else:
                            print("⚠️ No boundary marker found in initial data chunk")
                    else:
                        print("⚠️ No data received from stream")
                except StopIteration:
                    print("⚠️ Stream ended immediately")
                except Exception as e:
                    print(f"❌ Error reading from stream: {e}")
            else:
                print(f"⚠️ Content type is not MJPEG stream format")
            
            # Close the connection
            response.close()
            return True
        else:
            print(f"⚠️ Connection succeeded but server returned status {response.status_code}")
            print(f"   Response headers: {dict(response.headers)}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ Request timed out after {timeout}s")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        return False
    except Exception as e:
        print(f"❌ Request failed: {type(e).__name__}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test connectivity to a remote server')
    parser.add_argument('url', help='URL to test (e.g., http://100.91.32.59:8082)')
    parser.add_argument('--timeout', type=int, default=5, help='Connection timeout in seconds')
    parser.add_argument('--check-endpoints', action='store_true', help='Check specific endpoints (webcam and processed.mjpeg)')
    args = parser.parse_args()
    
    # Parse the URL to get host and port
    if not args.url.startswith('http'):
        print("Error: URL must start with http:// or https://")
        sys.exit(1)
    
    try:
        # Extract host and port from URL
        parsed_url = urllib.parse.urlparse(args.url)
        
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
            
            # Check specific endpoints if requested
            if args.check_endpoints and http_ok:
                base_url = args.url.rstrip('/')
                
                # Test the processed MJPEG stream endpoint
                processed_url = f"{base_url}/processed.mjpeg"
                print(f"Testing processed MJPEG endpoint...")
                mjpeg_ok = test_mjpeg_endpoint(processed_url, args.timeout)
                print("-" * 50)
                
                # Test the webcam endpoint
                webcam_url = f"{base_url}/webcam"
                print(f"Testing webcam POST endpoint...")
                try:
                    # Just a HEAD request to check if endpoint exists
                    webcam_response = requests.head(webcam_url, timeout=args.timeout)
                    print(f"Webcam endpoint responded with status: {webcam_response.status_code}")
                    # Note: Many servers will return 405 Method Not Allowed for HEAD on POST-only endpoints
                    if webcam_response.status_code in [200, 405]:
                        print("✅ Webcam endpoint appears to be available (will need POST requests)")
                    else:
                        print(f"⚠️ Webcam endpoint returned unexpected status: {webcam_response.status_code}")
                except Exception as e:
                    print(f"❌ Error checking webcam endpoint: {e}")
                print("-" * 50)
            
            # Print summary
            if http_ok:
                print("✅ Basic connection tests PASSED!")
                print("   TCP and HTTP connections are working.")
                if args.check_endpoints:
                    if mjpeg_ok:
                        print("✅ MJPEG stream endpoint is also working!")
                    else:
                        print("⚠️ MJPEG stream endpoint test failed. Check if the server is streaming.")
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