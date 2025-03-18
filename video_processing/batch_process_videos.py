#!/usr/bin/env python3

import os
import argparse
import time
import glob
import sys
from pathlib import Path

# Either use relative import from the same package, or add the parent directory to sys.path
from .video_processor import process_video

def batch_process_videos(input_dir, output_dir, model_type='yolov8', confidence=0.5, 
                         nms=0.4, target_classes=None, skip_existing=True, 
                         limit=None, yolo_model_size='m', force_yolo_implementation='yolov8core'):
    """
    Process all videos in the input directory and save results to the output directory.
    
    Args:
        input_dir (str): Directory containing input videos
        output_dir (str): Directory to save processed videos
        model_type (str): Type of model to use ('mobilenet', 'yolo', or 'yolov8')
        confidence (float): Confidence threshold for detections
        nms (float): Non-maximum suppression threshold for YOLO
        target_classes (list): List of target class names to detect
        skip_existing (bool): Skip videos that already have processed versions
        limit (int): Maximum number of videos to process (None for all)
        yolo_model_size (str): YOLOv8 model size ('n', 's', 'm', 'l', or 'x')
        force_yolo_implementation (str): Force a specific YOLOv8 implementation
                                       (default: 'yolov8core' for better compatibility)
        
    Returns:
        dict: Statistics about the processing
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all video files in the input directory
    video_files = []
    for ext in ['.mp4', '.avi', '.mov']:
        video_files.extend(glob.glob(os.path.join(input_dir, f'*{ext}')))
    
    # Sort videos by name for consistent processing order
    video_files.sort()
    
    # Limit the number of videos if specified
    if limit is not None and limit > 0:
        video_files = video_files[:limit]
    
    # Print summary before starting
    print(f"Found {len(video_files)} videos in {input_dir}")
    print(f"Using {model_type} model with confidence threshold {confidence}")
    if force_yolo_implementation:
        print(f"Forcing {force_yolo_implementation} implementation for YOLOv8")
    if target_classes:
        print(f"Detecting the following classes: {', '.join(target_classes)}")
    else:
        print("Detecting all available classes")
    
    # Process each video
    start_time = time.time()
    successful = 0
    skipped = 0
    failed = 0
    
    for i, video_path in enumerate(video_files, 1):
        # Get the base filename (without directory)
        video_basename = os.path.basename(video_path)
        output_path = os.path.join(output_dir, video_basename)
        
        # Check if output already exists
        if skip_existing and os.path.exists(output_path):
            print(f"[{i}/{len(video_files)}] Skipping {video_basename} (output already exists)")
            skipped += 1
            continue
        
        print(f"\n[{i}/{len(video_files)}] Processing {video_basename}")
        
        # Process the video
        success = process_video(
            input_path=video_path,
            output_path=output_path,
            model_type=model_type,
            confidence=confidence,
            nms=nms,
            target_classes=target_classes,
            no_progress=False,
            yolo_model_size=yolo_model_size,
            force_yolo_implementation=force_yolo_implementation
        )
        
        if success:
            successful += 1
        else:
            failed += 1
            print(f"Failed to process {video_basename}")
    
    # Print final summary
    total_time = time.time() - start_time
    print("\n" + "="*50)
    print(f"Batch processing complete in {total_time:.2f} seconds")
    print(f"Total videos: {len(video_files)}")
    print(f"Successfully processed: {successful}")
    print(f"Skipped (already exist): {skipped}")
    print(f"Failed: {failed}")
    print("="*50)
    
    return {
        "total": len(video_files),
        "successful": successful,
        "skipped": skipped,
        "failed": failed,
        "time_seconds": total_time
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch process videos with object detection')
    parser.add_argument('--input-dir', type=str, default='nissan_videos', 
                        help='Directory containing input videos (default: nissan_videos)')
    parser.add_argument('--output-dir', type=str, default='nissan_videos/output', 
                        help='Directory to save processed videos (default: nissan_videos/output)')
    parser.add_argument('--model', type=str, choices=['mobilenet', 'yolo', 'yolov8'], default='yolov8', 
                        help='Model to use (mobilenet, yolo, or yolov8, default: yolov8)')
    parser.add_argument('--confidence', type=float, default=0.5, 
                        help='Confidence threshold (0.0 to 1.0, default: 0.5)')
    parser.add_argument('--nms', type=float, default=0.4, 
                        help='Non-maximum suppression threshold for YOLO (0.0 to 1.0, default: 0.4)')
    parser.add_argument('--classes', type=str, 
                        help='Comma-separated list of classes to detect (default: all classes)')
    parser.add_argument('--no-skip', action='store_true', 
                        help='Do not skip videos that already have processed versions')
    parser.add_argument('--limit', type=int, 
                        help='Maximum number of videos to process (default: all)')
    parser.add_argument('--yolo-size', type=str, choices=['n', 's', 'm', 'l', 'x'], default='m', 
                        help='YOLOv8 model size: n(ano), s(mall), m(edium), l(arge), or x (default: m)')
    parser.add_argument('--force-yolo', type=str, choices=['ultralytics', 'onnx', 'yolov4-tiny', 'yolov8core'], default='yolov8core',
                        help='Force a specific YOLOv8 implementation (default: yolov8core)')
    
    args = parser.parse_args()
    
    # Parse target classes if specified
    target_classes = None
    if args.classes:
        target_classes = [cls.strip() for cls in args.classes.split(',')]
    
    # Process videos
    batch_process_videos(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_type=args.model,
        confidence=args.confidence,
        nms=args.nms,
        target_classes=target_classes,
        skip_existing=not args.no_skip,
        limit=args.limit,
        yolo_model_size=args.yolo_size,
        force_yolo_implementation=args.force_yolo
    )

if __name__ == "__main__":
    main() 