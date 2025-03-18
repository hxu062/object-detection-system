import cv2
import numpy as np
import time
import argparse
import os
import platform
import subprocess

class ObjectTracker:
    def __init__(self, confidence_threshold=0.5, target_classes=None):
        self.confidence_threshold = confidence_threshold
        
        # Load the MobileNet SSD model
        self.model = cv2.dnn.readNetFromCaffe(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        'core', 'models', 'MobileNetSSD_deploy.prototxt'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        'core', 'models', 'MobileNetSSD_deploy.caffemodel')
        )
        
        # List of class names the model can detect
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        
        # Classes we want to detect
        if target_classes and len(target_classes) > 0:
            self.target_classes = target_classes
            
            # Validate that all target classes are in the available classes
            for target_class in self.target_classes:
                if target_class not in self.classes:
                    print(f"Warning: '{target_class}' is not a valid class for MobileNet SSD model. It will be ignored.")
            
            # Filter to only include valid classes
            self.target_classes = [cls for cls in self.target_classes if cls in self.classes]
            
            if not self.target_classes:
                print("Warning: No valid target classes specified. Using all available classes.")
                self.target_classes = [cls for cls in self.classes if cls != "background"]
        else:
            # If no target classes specified, use all available classes
            self.target_classes = [cls for cls in self.classes if cls != "background"]
            
        print(f"Detecting the following classes: {', '.join(self.target_classes)}")
        
        # Colors for visualization (one color per class)
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
    def detect_objects(self, frame):
        """
        Detect objects in the given frame and return the frame with bounding boxes.
        """
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Create a blob from the frame
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            0.007843, 
            (300, 300), 
            127.5
        )
        
        # Set the blob as input to the model
        self.model.setInput(blob)
        
        # Forward pass through the model to get detections
        detections = self.model.forward()
        
        # Process each detection
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter out weak detections
            if confidence > self.confidence_threshold:
                # Get the class ID
                class_id = int(detections[0, 0, i, 1])
                class_name = self.classes[class_id]
                
                # We're only interested in the target classes
                if class_name in self.target_classes:
                    # Get bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Draw the bounding box
                    cv2.rectangle(frame, (startX, startY), (endX, endY), 
                                 self.colors[class_id].tolist(), 4)
                    
                    # Add a label
                    label = f"{class_name}: {confidence:.2f}"
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[class_id].tolist(), 2)
        
        return frame

def list_available_classes():
    """List all available classes for the models."""
    mobilenet_classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
    
    # Read YOLO/COCO classes
    coco_classes = []
    try:
        with open('models/coco.names', 'r') as f:
            coco_classes = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        coco_classes = ["File 'models/coco.names' not found"]
    
    return mobilenet_classes, coco_classes

def list_available_cameras():
    """List all available camera devices."""
    available_cameras = []
    
    # On macOS, try to detect Continuity Camera
    if platform.system() == 'Darwin':
        try:
            # Use system_profiler to get camera information
            result = subprocess.run(['system_profiler', 'SPCameraDataType'], 
                                   capture_output=True, text=True)
            camera_info = result.stdout
            
            # Parse the output to find camera names
            cameras = []
            for line in camera_info.split('\n'):
                if ':' in line and not line.strip().startswith('Camera'):
                    name = line.split(':')[0].strip()
                    if name and 'FaceTime' in name or 'iPhone' in name or 'iPad' in name:
                        cameras.append(name)
            
            # Add detected cameras to the list
            for i, name in enumerate(cameras):
                available_cameras.append((i, name))
            
            # If no cameras were found with system_profiler, fall back to OpenCV detection
            if not available_cameras:
                print("No cameras detected with system_profiler, falling back to OpenCV detection.")
        except Exception as e:
            print(f"Error detecting cameras with system_profiler: {e}")
            print("Falling back to OpenCV detection.")
    
    # If no cameras were found or we're not on macOS, use OpenCV detection
    if not available_cameras:
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                break
            ret, _ = cap.read()
            if ret:
                camera_name = f"Camera {index}"
                if index == 0:
                    camera_name = "Default Camera"
                elif index == 1:
                    camera_name = "External Camera (possibly iPhone)"
                available_cameras.append((index, camera_name))
            cap.release()
            index += 1
            # Limit to checking first 5 indices to avoid long delays
            if index >= 5:
                break
    
    return available_cameras

def setup_iphone_camera():
    """
    Provide instructions for setting up iPhone as a webcam using Continuity Camera.
    """
    print("\n=== Setting up iPhone as a webcam using Continuity Camera ===")
    print("1. Make sure your iPhone is running iOS 16 or later")
    print("2. Make sure your Mac is running macOS Ventura or later")
    print("3. Ensure both devices are signed in to the same Apple ID")
    print("4. Ensure both devices have Bluetooth and Wi-Fi turned on")
    print("5. Place your iPhone near your Mac")
    print("6. Your iPhone should automatically be detected as a camera option")
    print("\nIf your iPhone is not being detected:")
    print("- On your iPhone, go to Settings > General > AirPlay & Handoff and ensure Continuity Camera is enabled")
    print("- On your Mac, go to System Preferences > General > AirDrop & Handoff and ensure Continuity Camera is enabled")
    print("- Make sure both devices are on the same Wi-Fi network")
    print("- Restart both devices if necessary")
    
    input("\nPress Enter to continue...")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Object Movement Tracker')
    parser.add_argument('--output', type=str, help='Path to output video file')
    parser.add_argument('--no-display', action='store_true', help='Do not display the output')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold (0.0 to 1.0)')
    parser.add_argument('--camera', type=int, default=None, help='Camera index to use')
    parser.add_argument('--setup-iphone', action='store_true', help='Show instructions for setting up iPhone as webcam')
    parser.add_argument('--classes', type=str, help='Comma-separated list of classes to detect')
    parser.add_argument('--list-classes', action='store_true', help='List available classes for detection')
    args = parser.parse_args()
    
    # List available classes if requested
    if args.list_classes:
        mobilenet_classes, coco_classes = list_available_classes()
        print("\nAvailable classes for MobileNet SSD model:")
        print(", ".join([f"'{cls}'" for cls in mobilenet_classes if cls != 'background']))
        print("\nAvailable classes for YOLO model:")
        print(", ".join([f"'{cls}'" for cls in coco_classes]))
        return
    
    # Show iPhone setup instructions if requested
    if args.setup_iphone:
        setup_iphone_camera()
    
    # Parse target classes
    target_classes = None
    if args.classes:
        target_classes = [cls.strip() for cls in args.classes.split(',')]
    
    # Initialize the tracker
    tracker = ObjectTracker(confidence_threshold=args.confidence, target_classes=target_classes)
    
    # List available cameras if no camera index is provided
    camera_index = args.camera
    if camera_index is None:
        available_cameras = list_available_cameras()
        if not available_cameras:
            print("Error: No cameras detected.")
            setup_iphone = input("Would you like to see instructions for setting up iPhone as a webcam? (y/n): ").lower()
            if setup_iphone == 'y':
                setup_iphone_camera()
            return
        
        print("\nAvailable cameras:")
        for idx, name in available_cameras:
            print(f"{idx}: {name}")
        
        try:
            camera_index = int(input("\nSelect camera index: "))
        except ValueError:
            print("Invalid input. Using default camera (0).")
            camera_index = 0
    
    # Initialize the selected camera
    cap = cv2.VideoCapture(camera_index)
    
    # Check if the camera is opened correctly
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}.")
        setup_iphone = input("Would you like to see instructions for setting up iPhone as a webcam? (y/n): ").lower()
        if setup_iphone == 'y':
            setup_iphone_camera()
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output path is provided
    video_writer = None
    if args.output:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
        print(f"Saving output to {args.output}")
    
    print("Press 'q' to quit")
    
    # FPS calculation variables
    fps_start_time = time.time()
    fps_value = 0
    frame_count = 0
    
    try:
        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            # Detect objects and draw bounding boxes
            processed_frame = tracker.detect_objects(frame)
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time > 1:
                fps_value = frame_count / elapsed_time
                frame_count = 0
                fps_start_time = time.time()
            
            # Display FPS on the frame
            cv2.putText(processed_frame, f"FPS: {fps_value:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write the frame to the output video if specified
            if video_writer:
                video_writer.write(processed_frame)
            
            # Display the processed frame if not disabled
            if not args.no_display:
                cv2.imshow("Object Tracker (MobileNet SSD)", processed_frame)
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Print a dot to show progress
                print(".", end="", flush=True)
                
                # Check if user pressed Ctrl+C
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Release resources
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print("\nApplication closed")

if __name__ == "__main__":
    main() 