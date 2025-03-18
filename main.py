#!/usr/bin/env python3
"""
Main entry point for the object detection system.
Provides access to both real-time webcam tracking and video processing.
"""

import argparse
import os
import sys

def main():
    # Create a simple parser for the main command
    main_parser = argparse.ArgumentParser(description="Object Detection System")
    main_parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    
    # First check if we're just listing models
    if "--list-models" in sys.argv:
        print("Available models:")
        print("  - mobilenet: MobileNet SSD (fastest, but less accurate)")
        print("  - yolo: YOLOv4-tiny (good balance of speed and accuracy)")
        print("  - yolov8: YOLOv8 with multiple implementations (most accurate)")
        return
    
    # Check if we have a task argument (webcam or video)
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        main_parser.add_argument("task", choices=["webcam", "video"], help="Task to run: webcam (real-time tracking) or video (video processing)")
        main_parser.print_help()
        sys.exit(1)
    
    # Get the task (first argument)
    task = sys.argv[1]
    
    # Add the root directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Remove the task from argv to pass clean arguments to the submodule
    original_argv = sys.argv.copy()
    sys.argv = [original_argv[0]] + original_argv[2:]
    
    # Run the appropriate task
    if task == "webcam":
        # Import and run webcam tracking
        from webcam_tracking.run_tracker import main as run_webcam
        run_webcam()
    
    elif task == "video":
        # Check if we should do batch processing
        if "--batch" in original_argv:
            # Import and run batch video processing
            from video_processing.batch_process_videos import main as run_batch
            run_batch()
        else:
            # Import and run single video processing
            from video_processing.video_processor import main as process_video
            process_video()
    else:
        print(f"Unknown task: {task}")
        main_parser.add_argument("task", choices=["webcam", "video"], help="Task to run: webcam (real-time tracking) or video (video processing)")
        main_parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 