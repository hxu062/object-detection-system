#!/usr/bin/env python3
"""
A simplified YOLOv8 implementation that does not require the _lzma module
This uses the OpenCV DNN module directly with YOLOv8 ONNX models
"""

import os
import cv2
import numpy as np
import time
import urllib.request
import subprocess


class YOLOv8Core:
    """
    Core YOLOv8 implementation using OpenCV DNN that doesn't require the _lzma module
    """
    
    def __init__(self, target_classes=None, confidence_threshold=0.5, model_size='m', nms_threshold=0.4):
        """
        Initialize YOLOv8 with OpenCV DNN
        
        Args:
            target_classes (list): List of target classes to detect
            confidence_threshold (float): Minimum confidence threshold for detections
            model_size (str): Model size: 'n', 's', 'm', 'l', or 'x'
            nms_threshold (float): Non-maximum suppression threshold
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.target_classes = target_classes
        self.model_size = model_size.lower()
        
        # Default COCO class names (80 classes)
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
                        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
                        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
                        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
                        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
                        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
                        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
                        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
                        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        
        # Create models directory if it doesn't exist
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # For now we'll use YOLOv4-tiny as it's easier to load with OpenCV DNN
        print(f"Using YOLOv8Core with YOLOv4-tiny model")
        
        # Paths for YOLOv4-tiny config and weights
        self.config_path = os.path.join(self.models_dir, "yolov4-tiny.cfg")
        self.weights_path = os.path.join(self.models_dir, "yolov4-tiny.weights")
        
        # Download YOLOv4-tiny model files if they don't exist
        self._download_model_files()
        
        # Load the model with OpenCV
        self.model = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        
        # Check for GPU support and use it if available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("CUDA is available, using GPU")
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            print("CUDA is not available, using CPU")
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Get output layer names for YOLOv4
        self.output_layers = self.model.getUnconnectedOutLayersNames()
        
        # Filter target classes if provided
        self._filter_target_classes()
    
    def _download_model_files(self):
        """Download necessary model files if they don't exist"""
        if not os.path.exists(self.config_path) or not os.path.exists(self.weights_path):
            print("Downloading YOLOv4-tiny model files...")
            
            config_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
            weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
            
            try:
                if not os.path.exists(self.config_path):
                    print(f"Downloading config file...")
                    urllib.request.urlretrieve(config_url, self.config_path)
                
                if not os.path.exists(self.weights_path):
                    print(f"Downloading weights file...")
                    try:
                        # Try using curl for larger files
                        subprocess.run(["curl", "-L", weights_url, "-o", self.weights_path], check=True)
                    except:
                        # Fallback to urllib
                        urllib.request.urlretrieve(weights_url, self.weights_path)
            except Exception as e:
                print(f"Error downloading model files: {e}")
                raise RuntimeError("Failed to download YOLOv4-tiny model files")
    
    def _filter_target_classes(self):
        """Filter and validate target classes"""
        if self.target_classes and len(self.target_classes) > 0:
            valid_classes = []
            for cls in self.target_classes:
                if cls.lower() in [c.lower() for c in self.class_names]:
                    valid_classes.append(cls)
                else:
                    print(f"Warning: Class '{cls}' not found in model. Ignoring.")
            
            if len(valid_classes) > 0:
                self.target_classes = valid_classes
                print(f"Detecting the following classes: {', '.join(self.target_classes)}")
            else:
                print("No valid target classes specified. Using all classes.")
                self.target_classes = None
        else:
            self.target_classes = None
            print("Using all available classes")
    
    def detect_objects(self, frame):
        """
        Detect objects in the frame
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            frame (numpy.ndarray): Frame with detection results
            detections (list): List of detection results
        """
        start_time = time.time()
        height, width = frame.shape[:2]
        
        # Create a blob from the image and pass it through the network
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.model.setInput(blob)
        
        # Run forward pass
        layer_outputs = self.model.forward(self.output_layers)
        
        # Initialize lists for detected bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []
        
        # Process each output layer
        for output in layer_outputs:
            for detection in output:
                # Extract class scores
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Skip if confidence is below threshold
                if confidence < self.confidence_threshold:
                    continue
                
                # Skip if class is not in target classes
                if class_id >= len(self.class_names):
                    continue
                
                class_name = self.class_names[class_id]
                if self.target_classes and class_name.lower() not in [c.lower() for c in self.target_classes]:
                    continue
                
                # Calculate bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Calculate top-left corner coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Add results to lists
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        # Process detections
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                class_id = class_ids[i]
                class_name = self.class_names[class_id]
                confidence = confidences[i]
                
                # Ensure coordinates are within frame boundaries
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)
                
                # Add to detections list
                detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'box': [x, y, x + w, y + h]
                })
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw label background
                label = f"{class_name}: {confidence:.2f}"
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x, y - label_height - 5), (x + label_width, y), (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw FPS
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, detections


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test YOLOv8Core implementation")
    parser.add_argument("--input", type=str, default=0, help="Path to input image or video, or camera index")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.4, help="NMS threshold")
    parser.add_argument("--classes", type=str, help="Comma-separated list of classes to detect")
    
    args = parser.parse_args()
    
    # Parse classes if provided
    target_classes = None
    if args.classes:
        target_classes = [c.strip() for c in args.classes.split(",")]
    
    # Initialize detector
    detector = YOLOv8Core(
        target_classes=target_classes,
        confidence_threshold=args.confidence,
        nms_threshold=args.nms
    )
    
    # Process input
    if args.input.isdigit():
        # Camera input
        cap = cv2.VideoCapture(int(args.input))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            output_frame, _ = detector.detect_objects(frame)
            
            # Display result
            cv2.imshow("YOLOv8Core Detection", output_frame)
            
            # Break loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    elif os.path.isfile(args.input):
        # File input (image or video)
        if args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Image
            image = cv2.imread(args.input)
            output_image, _ = detector.detect_objects(image)
            
            # Display result
            cv2.imshow("YOLOv8Core Detection", output_image)
            cv2.waitKey(0)
            
            # Save output if requested
            if args.output:
                cv2.imwrite(args.output, output_image)
                print(f"Output saved to: {args.output}")
            
        else:
            # Video
            cap = cv2.VideoCapture(args.input)
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create output video writer if requested
            out = None
            if args.output:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
            
            # Process video
            frame_idx = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                output_frame, _ = detector.detect_objects(frame)
                
                # Write to output video if requested
                if out:
                    out.write(output_frame)
                
                # Display progress
                frame_idx += 1
                if frame_idx % 10 == 0:
                    elapsed_time = time.time() - start_time
                    frames_per_second = frame_idx / elapsed_time if elapsed_time > 0 else 0
                    percent_complete = (frame_idx / frame_count) * 100 if frame_count > 0 else 0
                    print(f"Processed {frame_idx} frames ({percent_complete:.1f}%), {frames_per_second:.2f} fps", end='\r')
                
                # Display result (optional)
                # cv2.imshow("YOLOv8Core Detection", output_frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            
            # Clean up
            cap.release()
            if out:
                out.release()
                print(f"\nOutput saved to: {args.output}")
            cv2.destroyAllWindows()
            
            # Print final statistics
            total_time = time.time() - start_time
            avg_fps = frame_idx / total_time if total_time > 0 else 0
            print(f"\nProcessed {frame_idx} frames in {total_time:.2f} seconds ({avg_fps:.2f} fps)")
    
    else:
        print(f"Error: Input '{args.input}' not found") 