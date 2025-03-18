#!/usr/bin/env python3

import os
import glob
import cv2
import numpy as np
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_video(video_path):
    """
    Analyze a processed video file to count detections and extract statistics.
    
    Args:
        video_path: Path to the processed video with bounding boxes
        
    Returns:
        Dictionary with detection statistics
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Sampling rate - check every 10th frame to speed up analysis
    sampling_rate = 10
    
    # Statistics
    stats = {
        'filename': os.path.basename(video_path),
        'frame_count': total_frames,
        'width': width,
        'height': height,
        'fps': fps,
        'duration': total_frames / fps if fps > 0 else 0,
        'objects_per_frame': [],
        'total_detections': 0,
        'objects': defaultdict(int)
    }
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process only every Nth frame to speed up analysis
        if frame_idx % sampling_rate == 0:
            # Detect bounding box annotations in the frame
            # This assumes the format that our video_processor.py uses:
            # - White/colored rectangles for bounding boxes
            # - Text labels for class names
            
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Get text using simple thresholding to find white/bright text on darker backgrounds
            # This is a heuristic approach - may need adjustment for different videos
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # Find contours that might be bounding boxes
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count rectangles that are likely bounding boxes (filter by size)
            valid_contours = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 20 and h > 20 and w < width//2 and h < height//2:
                    valid_contours += 1
                    
                    # Try to read the class name by looking at bright pixels near the top of the bounding box
                    # This is a heuristic and might need adjustment
                    roi = frame[max(0, y-20):y, max(0, x):min(x+w, width)]
                    if roi.size > 0:
                        # Simple heuristic - the brightest areas in the ROI are likely text
                        bright_pixels = np.sum(roi > 200)
                        if bright_pixels > 50:
                            # Assuming the class name is represented by the presence of bright pixels
                            # We increment a generic counter here
                            stats['objects']['vehicle'] += 1
            
            stats['objects_per_frame'].append(valid_contours)
            stats['total_detections'] += valid_contours
        
        frame_idx += 1
    
    cap.release()
    
    # Calculate averages
    if stats['objects_per_frame']:
        stats['avg_objects_per_frame'] = np.mean(stats['objects_per_frame'])
        stats['max_objects_per_frame'] = np.max(stats['objects_per_frame'])
    else:
        stats['avg_objects_per_frame'] = 0
        stats['max_objects_per_frame'] = 0
        
    return stats


def generate_summary(input_dir, output_file=None, generate_plots=True):
    """
    Generate a summary of all processed videos in the input directory.
    
    Args:
        input_dir: Directory containing processed videos
        output_file: Path to save the summary report (optional)
        generate_plots: Whether to generate plots
    """
    # Get all video files
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov']:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    video_files.sort()
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} videos in {input_dir}")
    print("Analyzing videos...")
    
    all_stats = []
    total_detections = 0
    total_frames = 0
    total_duration = 0
    
    for i, video_path in enumerate(video_files):
        print(f"[{i+1}/{len(video_files)}] Analyzing {os.path.basename(video_path)}")
        stats = analyze_video(video_path)
        
        if stats:
            all_stats.append(stats)
            total_detections += stats['total_detections']
            total_frames += stats['frame_count']
            total_duration += stats['duration']
    
    # Summary calculations
    if not all_stats:
        print("No valid statistics found for any video.")
        return
    
    avg_detections_per_frame = total_detections / total_frames if total_frames > 0 else 0
    
    # Print summary
    summary = []
    summary.append("=== DETECTION SUMMARY ===")
    summary.append(f"Total videos analyzed: {len(all_stats)}")
    summary.append(f"Total video duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    summary.append(f"Total frames: {total_frames}")
    summary.append(f"Total detections (sampled): {total_detections}")
    summary.append(f"Average detections per frame: {avg_detections_per_frame:.2f}")
    
    # Per video stats
    summary.append("\n=== PER VIDEO STATISTICS ===")
    for stats in all_stats:
        summary.append(f"\nFilename: {stats['filename']}")
        summary.append(f"  Duration: {stats['duration']:.2f} seconds")
        summary.append(f"  Frame count: {stats['frame_count']}")
        summary.append(f"  Resolution: {stats['width']}x{stats['height']}")
        summary.append(f"  Detections (sampled): {stats['total_detections']}")
        summary.append(f"  Avg objects per frame: {stats['avg_objects_per_frame']:.2f}")
        summary.append(f"  Max objects per frame: {stats['max_objects_per_frame']}")
    
    # Print and save report
    report = "\n".join(summary)
    print(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Report saved to {output_file}")
    
    # Generate plots if requested
    if generate_plots and all_stats:
        plot_dir = os.path.join(os.path.dirname(input_dir), "analysis_plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot detections per video
        video_names = [stats['filename'] for stats in all_stats]
        detections = [stats['total_detections'] for stats in all_stats]
        
        plt.figure(figsize=(15, 8))
        plt.bar(range(len(video_names)), detections)
        plt.xticks(range(len(video_names)), video_names, rotation=90)
        plt.xlabel('Video')
        plt.ylabel('Number of Detections (Sampled)')
        plt.title('Detections per Video')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'detections_per_video.png'))
        
        # Plot average objects per frame
        avg_objects = [stats['avg_objects_per_frame'] for stats in all_stats]
        
        plt.figure(figsize=(15, 8))
        plt.bar(range(len(video_names)), avg_objects)
        plt.xticks(range(len(video_names)), video_names, rotation=90)
        plt.xlabel('Video')
        plt.ylabel('Average Objects per Frame')
        plt.title('Average Objects per Frame per Video')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'avg_objects_per_video.png'))
        
        print(f"Plots saved to {plot_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate summary of processed videos with object detection")
    parser.add_argument("--input", "-i", default="nissan_videos/output", 
                        help="Directory containing processed videos (default: nissan_videos/output)")
    parser.add_argument("--output", "-o", default="video_analysis_report.txt",
                        help="Output file for the summary report (default: video_analysis_report.txt)")
    parser.add_argument("--no-plots", action="store_true", 
                        help="Disable generation of plots")
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.isdir(args.input):
        print(f"Error: Input directory {args.input} does not exist")
        return
    
    generate_summary(args.input, args.output, not args.no_plots)


if __name__ == "__main__":
    main() 