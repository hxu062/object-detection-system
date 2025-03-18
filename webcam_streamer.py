#!/usr/bin/env python3
"""
Simple webcam streamer to send frames to the remote processing server
"""

import cv2
import requests
import time
import argparse
import numpy as np
import threading
import sys
import os

# Global variables for control
running = True
frames_sent = 0

def stream_webcam(server_url, fps=15, debug=False, width=640, height=480):
    """Stream webcam frames to the server"""
    global running, frames_sent
    
    print(f"Initializing webcam...")
    cap = cv2.VideoCapture(0)  # Use default camera
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Get actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam initialized at {actual_width}x{actual_height}")
    
    # Create a session for better performance
    session = requests.Session()
    
    # Calculate frame delay for desired FPS
    frame_delay = 1.0 / fps
    
    print(f"Starting webcam stream to {server_url} at {fps} FPS")
    
    try:
        while running:
            loop_start = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                time.sleep(1)  # Wait before retry
                continue
            
            # Add timestamp and frame counter
            cv2.putText(frame, f"Frame: {frames_sent}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Time: {time.strftime('%H:%M:%S')}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Encode frame to JPEG
            _, img_encoded = cv2.imencode('.jpg', frame)
            
            # Send the frame to server
            try:
                response = session.post(
                    server_url,
                    data=img_encoded.tobytes(),
                    headers={'Content-Type': 'image/jpeg'},
                    timeout=1
                )
                
                if response.status_code == 200:
                    frames_sent += 1
                    if debug or frames_sent % 50 == 0:
                        print(f"Sent frame {frames_sent} ({len(img_encoded)} bytes)")
                else:
                    print(f"Warning: Server returned status {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"Error sending frame: {e}")
                time.sleep(0.5)  # Wait before retry
            
            # Calculate time to sleep to maintain FPS
            elapsed = time.time() - loop_start
            sleep_time = max(0, frame_delay - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Print actual FPS occasionally
            if frames_sent % 100 == 0:
                actual_fps = 1.0 / (time.time() - loop_start)
                print(f"Actual FPS: {actual_fps:.2f}")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cap.release()
        print(f"Webcam stream stopped. Sent {frames_sent} frames.")

def main():
    global running
    
    parser = argparse.ArgumentParser(description='Stream webcam to remote server')
    parser.add_argument('url', help='URL of the server endpoint (e.g., http://100.91.32.59:8082/webcam)')
    parser.add_argument('--fps', type=int, default=15, help='Target frames per second')
    parser.add_argument('--width', type=int, default=640, help='Width of webcam frames')
    parser.add_argument('--height', type=int, default=480, help='Height of webcam frames')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    try:
        # Start streaming in a thread
        stream_thread = threading.Thread(
            target=stream_webcam,
            args=(args.url, args.fps, args.debug, args.width, args.height)
        )
        stream_thread.daemon = True
        stream_thread.start()
        
        print("Streaming started. Press Ctrl+C to stop.")
        
        # Keep main thread alive
        while stream_thread.is_alive():
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        running = False
        # Allow time for thread to clean up
        time.sleep(1)

if __name__ == "__main__":
    main() 