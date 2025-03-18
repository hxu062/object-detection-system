#!/usr/bin/env python3

import os
import cv2
import argparse
import time
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path so we can import from other modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from webcam_tracking.human_tracker import ObjectTracker
from webcam_tracking.human_tracker_yolo import YOLOObjectTracker
from core.yolov8_wrapper import YOLOv8Wrapper

def get_video_writer_config(output_path):
    """
    Get the appropriate VideoWriter configuration based on the output file extension.
    
    Args:
        output_path (str): Path to save the output video file
    
    Returns:
        tuple: (fourcc, extension) where fourcc is the FourCC code and extension is the file extension
    """
    # Get the file extension
    _, ext = os.path.splitext(output_path.lower())
    
    if ext in ['.mp4', '.m4v']:
        # For MP4 output, try H.264 codec
        try:
            return cv2.VideoWriter_fourcc(*'H264'), ext
        except:
            try:
                return cv2.VideoWriter_fourcc(*'X264'), ext
            except:
                try:
                    return cv2.VideoWriter_fourcc(*'DIVX'), ext
                except:
                    # Fall back to XVID with .avi if H.264 is not available
                    print("Warning: H.264 codec not available, falling back to XVID codec with .avi extension")
                    return cv2.VideoWriter_fourcc(*'XVID'), '.avi'
    elif ext in ['.avi']:
        # For AVI output, use XVID codec
        return cv2.VideoWriter_fourcc(*'XVID'), ext
    else:
        # For any other extension, default to XVID with .avi
        print(f"Warning: Unsupported output extension '{ext}', falling back to XVID codec with .avi extension")
        return cv2.VideoWriter_fourcc(*'XVID'), '.avi'

def process_video(input_path, output_path, model_type='mobilenet', confidence=0.5, 
                 nms=0.4, target_classes=None, no_progress=False, yolo_model_size='m',
                 force_yolo_implementation=None):
    """
    Process a video file with object detection and save results as a new video.
    
    Args:
        input_path (str): Path to the input video file
        output_path (str): Path to save the output video file
        model_type (str): Type of model to use ('mobilenet', 'yolo', or 'yolov8')
        confidence (float): Confidence threshold for detections
        nms (float): Non-maximum suppression threshold for YOLO
        target_classes (list): List of target class names to detect
        no_progress (bool): Do not show progress information
        yolo_model_size (str): YOLOv8 model size ('n', 's', 'm', 'l', or 'x')
        force_yolo_implementation (str): Force a specific YOLOv8 implementation
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Check that the input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found")
        return False
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{input_path}'")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize the appropriate tracker
    if model_type.lower() == 'yolov8':
        # Map command-line args to wrapper implementation options
        force_onnx = force_yolo_implementation == "onnx"
        force_yolov4 = force_yolo_implementation == "yolov4-tiny"
        force_yolov8core = force_yolo_implementation == "yolov8core"
        
        try:
            print(f"Initializing YOLOv8 with model size '{yolo_model_size}'")
            tracker = YOLOv8Wrapper(
                target_classes=target_classes,
                confidence_threshold=confidence,
                model_size=yolo_model_size,
                force_onnx=force_onnx,
                force_yolov4=force_yolov4,
                force_yolov8core=force_yolov8core
            )
            print(f"Using YOLOv8 {tracker.implementation} implementation")
        except Exception as e:
            print(f"Error initializing YOLOv8: {str(e)}")
            print("Falling back to YOLOv4-tiny...")
            model_type = 'yolo'
            
    if model_type.lower() == 'yolo':
        tracker = YOLOObjectTracker(target_classes=target_classes, confidence_threshold=confidence, nms_threshold=nms)
    elif model_type.lower() == 'mobilenet':
        tracker = ObjectTracker(target_classes=target_classes, confidence_threshold=confidence)
    # YOLOv8 is handled above
    
    # Process the video
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Read a frame from the video
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame with object detection
            if model_type.lower() == 'yolov8':
                # YOLOv8Wrapper.detect_objects returns (frame, detections)
                processed_frame, _ = tracker.detect_objects(frame)
            else:
                # Other trackers return just the frame
                processed_frame = tracker.detect_objects(frame)
            
            # Write the processed frame to the output video
            out.write(processed_frame)
            
            # Update progress
            frame_count += 1
            if not no_progress and frame_count % 10 == 0:
                elapsed_time = time.time() - start_time
                frames_per_second = frame_count / elapsed_time if elapsed_time > 0 else 0
                percent_complete = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Processed {frame_count} frames ({percent_complete:.1f}%), {frames_per_second:.2f} fps", end='\r')
    
    except Exception as e:
        print(f"\nError processing video: {str(e)}")
        cap.release()
        out.release()
        return False
    
    # Release resources
    cap.release()
    out.release()
    
    # Print final statistics
    total_time = time.time() - start_time
    average_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\nProcessed {frame_count} frames in {total_time:.2f} seconds ({average_fps:.2f} fps)")
    print(f"Output saved to: {output_path}")
    
    return True

def list_available_classes():
    """
    List all available classes for detection with the MobileNet SSD and YOLO models.
    
    Returns:
        tuple: (mobilenet_classes, yolo_classes)
    """
    # MobileNet SSD classes
    mobilenet_classes = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    # YOLO classes (COCO dataset)
    yolo_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
        'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
        'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    return mobilenet_classes, yolo_classes

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process a video file with object detection')
    parser.add_argument('input', nargs='?', type=str, help='Path to the input video file')
    parser.add_argument('--output', type=str, help='Path to save the output video file (default: "output_[input].mp4")')
    parser.add_argument('--model', type=str, choices=['mobilenet', 'yolo', 'yolov8'], default='mobilenet', 
                        help='Model to use (mobilenet, yolo, or yolov8)')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold (0.0 to 1.0)')
    parser.add_argument('--nms', type=float, default=0.4, help='Non-maximum suppression threshold for YOLO (0.0 to 1.0)')
    parser.add_argument('--classes', type=str, help='Comma-separated list of classes to detect (default: all classes)')
    parser.add_argument('--list-classes', action='store_true', help='List available classes for detection and exit')
    parser.add_argument('--no-progress', action='store_true', help='Do not display progress information')
    parser.add_argument('--yolo-size', type=str, choices=['n', 's', 'm', 'l', 'x'], default='m', 
                        help='YOLOv8 model size: n(ano), s(mall), m(edium), l(arge), or x (default: m)')
    parser.add_argument('--force-yolo', type=str, choices=['ultralytics', 'onnx', 'yolov4-tiny', 'yolov8core'], default='yolov8core',
                        help='Force a specific YOLOv8 implementation (default: yolov8core)')
    
    args = parser.parse_args()
    
    # List available classes if requested
    if args.list_classes:
        mobilenet_classes, yolo_classes = list_available_classes()
        print("\nAvailable classes for MobileNet SSD model:")
        print(", ".join([f"'{cls}'" for cls in mobilenet_classes if cls != 'background']))
        print("\nAvailable classes for YOLO models:")
        print(", ".join([f"'{cls}'" for cls in yolo_classes]))
        return
    
    # Check if input file is specified
    if not args.input:
        parser.print_help()
        print("\nError: Input file is required")
        return
    
    # Parse target classes if specified
    target_classes = None
    if args.classes:
        target_classes = [cls.strip() for cls in args.classes.split(',')]
    
    # Determine output path if not specified
    if args.output:
        output_path = args.output
    else:
        input_basename = os.path.basename(args.input)
        input_name, input_ext = os.path.splitext(input_basename)
        output_path = f"output_{input_name}.mp4"
    
    # Process the video
    success = process_video(
        input_path=args.input,
        output_path=output_path,
        model_type=args.model,
        confidence=args.confidence,
        nms=args.nms,
        target_classes=target_classes,
        no_progress=args.no_progress,
        yolo_model_size=args.yolo_size,
        force_yolo_implementation=args.force_yolo
    )
    
    if not success:
        print("Video processing failed")
        exit(1)

if __name__ == "__main__":
    main() 