#!/usr/bin/env python3
"""
Run object detection on webcam or video feed.
"""

import os
import sys
import cv2
import argparse
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from our modules
from webcam_tracking.human_tracker import ObjectTracker
from webcam_tracking.human_tracker_yolo import YOLOObjectTracker
from core.yolov8_wrapper import YOLOv8Wrapper

def main():
    """
    Run the object tracker on webcam or video feed.
    """
    # Parse command line arguments - We need to create a new parser to handle all args
    parser = argparse.ArgumentParser(description='Real-time webcam tracking')
    
    # The model argument and camera arguments
    parser.add_argument('--model', type=str, choices=['mobilenet', 'yolo', 'yolov8'], default='mobilenet',
                        help='Model to use: mobilenet, yolo, or yolov8 (default: mobilenet)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--video', type=str, help='Path to video file (default: use webcam)')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    parser.add_argument('--nms', type=float, default=0.4, help='Non-maximum suppression threshold for YOLO (default: 0.4)')
    parser.add_argument('--classes', type=str, help='Comma-separated list of classes to detect (default: all classes)')
    
    # YOLOv8 specific arguments
    parser.add_argument('--yolo-size', type=str, choices=['n', 's', 'm', 'l', 'x'], default='m',
                        help='YOLOv8 model size: n(ano), s(mall), m(edium), l(arge), or x (default: m)')
    parser.add_argument('--force-yolo', type=str, choices=['ultralytics', 'onnx', 'yolov4-tiny', 'yolov8core'], 
                        default='yolov8core', help='Force a specific YOLOv8 implementation (default: yolov8core)')
    
    args = parser.parse_args()
    
    # Parse target classes if specified
    target_classes = None
    if args.classes:
        target_classes = [cls.strip() for cls in args.classes.split(',')]
    
    # Initialize the appropriate tracker
    if args.model == 'yolov8':
        # Map command-line args to wrapper implementation options
        force_onnx = args.force_yolo == "onnx"
        force_yolov4 = args.force_yolo == "yolov4-tiny"
        force_yolov8core = args.force_yolo == "yolov8core"
        
        try:
            print(f"Initializing YOLOv8 with model size '{args.yolo_size}'")
            tracker = YOLOv8Wrapper(
                target_classes=target_classes,
                confidence_threshold=args.confidence,
                model_size=args.yolo_size,
                force_onnx=force_onnx,
                force_yolov4=force_yolov4,
                force_yolov8core=force_yolov8core
            )
            print(f"Using YOLOv8 {tracker.implementation} implementation")
        except Exception as e:
            print(f"Error initializing YOLOv8: {str(e)}")
            print("Falling back to YOLOv4-tiny...")
            args.model = 'yolo'
    
    if args.model == 'yolo':
        tracker = YOLOObjectTracker(
            confidence_threshold=args.confidence, 
            nms_threshold=args.nms,
            target_classes=target_classes
        )
        print(f"Using YOLOv4-tiny detector with confidence threshold {args.confidence}")
    elif args.model == 'mobilenet':
        tracker = ObjectTracker(
            confidence_threshold=args.confidence,
            target_classes=target_classes
        )
        print(f"Using MobileNet SSD detector with confidence threshold {args.confidence}")
    
    # Open webcam or video file
    if args.video:
        print(f"Opening video file: {args.video}")
        cap = cv2.VideoCapture(args.video)
    else:
        print(f"Opening webcam with index: {args.camera}")
        cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print("Press 'q' to exit")
    
    # Process frames
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Read a frame
        ret, frame = cap.read()
        
        if not ret:
            if args.video:
                # End of video file
                break
            else:
                # Camera error, try to reopen
                print("Warning: Could not read frame, trying to reopen camera...")
                cap.release()
                cap = cv2.VideoCapture(args.camera)
                if not cap.isOpened():
                    print("Error: Could not reopen camera")
                    break
                continue
        
        # Process the frame with the tracker
        if args.model == 'yolov8':
            processed_frame, _ = tracker.detect_objects(frame)
        else:
            processed_frame = tracker.detect_objects(frame)
        
        # Display the result
        cv2.imshow('Object Tracking', processed_frame)
        
        # Calculate FPS every 10 frames
        frame_count += 1
        if frame_count % 10 == 0:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            print(f"FPS: {fps:.2f}", end='\r')
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    if frame_count > 0:
        total_time = time.time() - start_time
        average_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\nProcessed {frame_count} frames in {total_time:.2f} seconds ({average_fps:.2f} fps)")

if __name__ == "__main__":
    main() 