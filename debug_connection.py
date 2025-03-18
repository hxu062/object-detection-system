#!/usr/bin/env python3
"""
Debug script to test connection and image decoding with OpenCV
"""
import cv2
import numpy as np
import requests
import time
import argparse
import sys

def test_connection(url):
    """Test basic HTTP connectivity to the URL"""
    print(f"Testing connection to {url}")
    
    try:
        # Just make a simple request first
        response = requests.head(url, timeout=5)
        print(f"Connected with status code: {response.status_code}")
        return True
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False

def test_image_stream(url, save_path=None):
    """Test receiving and decoding images from the MJPEG stream"""
    print(f"Attempting to receive and decode images from {url}")
    
    try:
        # Start streaming request
        response = requests.get(url, stream=True, timeout=10)
        print(f"Stream connected with status code: {response.status_code}")
        
        # Create a buffer to store the multipart stream
        stream_buffer = bytes()
        boundary = b"--jpgboundary"
        found_frames = 0
        start_time = time.time()
        
        # Read for up to 5 seconds or until we get 3 frames
        while time.time() - start_time < 5 and found_frames < 3:
            chunk = next(response.iter_content(chunk_size=1024))
            
            if not chunk:
                continue
            
            # Add the chunk to our buffer
            stream_buffer += chunk
            
            # Look for the boundary and process frames
            if boundary in stream_buffer:
                parts = stream_buffer.split(boundary)
                
                # Process completed parts
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
                                
                                if img is not None:
                                    found_frames += 1
                                    print(f"✓ Successfully decoded frame {found_frames} with shape {img.shape}")
                                    
                                    # Save the image if requested
                                    if save_path:
                                        cv2.imwrite(f"{save_path}/frame_{found_frames}.jpg", img)
                                        print(f"  Saved to {save_path}/frame_{found_frames}.jpg")
                                else:
                                    print("✗ Image decode returned None")
                            except Exception as e:
                                print(f"✗ Error decoding image: {str(e)}")
                
                # Keep the last part which might be incomplete
                stream_buffer = parts[-1]
        
        response.close()
        print(f"Received and processed {found_frames} frames")
        return found_frames > 0
        
    except Exception as e:
        print(f"Stream test failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Debug OpenCV connection and image decoding")
    parser.add_argument("server_ip", help="IP address of the remote server")
    parser.add_argument("--port", type=int, default=8082, help="Port number (default: 8082)")
    parser.add_argument("--save", action="store_true", help="Save decoded frames to disk")
    parser.add_argument("--path", default="debug_frames", help="Path to save frames (default: debug_frames)")
    
    args = parser.parse_args()
    
    # Create URL
    url = f"http://{args.server_ip}:{args.port}/processed.mjpeg"
    
    # Create save directory if needed
    import os
    if args.save and not os.path.exists(args.path):
        os.makedirs(args.path)
    
    # Test connection first
    if not test_connection(url):
        print("Connection test failed. Please check if the server is running and accessible.")
        return 1
    
    # Then test image streaming
    success = test_image_stream(url, args.path if args.save else None)
    
    if success:
        print("\n✓ Successfully received and decoded images from the stream.")
        print("  The connection and basic image processing work correctly.")
    else:
        print("\n✗ Failed to receive or decode images from the stream.")
        print("  This indicates a problem with the image stream format or OpenCV compatibility.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 