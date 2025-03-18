#!/usr/bin/env python3
"""
YOLOv8 wrapper class that handles different ways of using YOLOv8:
1. Using the ultralytics package if available
2. Using the YOLOv8Core class (our custom implementation)
3. Using the ONNX version of YOLOv8 with OpenCV DNN if ultralytics fails
4. Falling back to YOLOv4-tiny if all else fails
"""

import os
import cv2
import numpy as np
import time
import urllib.request
import subprocess
from pathlib import Path
import sys


class YOLOv8Wrapper:
    """
    Wrapper class for YOLOv8 object detection with multiple fallback options
    """
    
    def __init__(self, target_classes=None, confidence_threshold=0.5, model_size='m', 
                 force_onnx=False, force_yolov4=False, force_yolov8core=False):
        """
        Initialize YOLOv8 detection with fallbacks
        
        Args:
            target_classes (list): List of target classes to detect
            confidence_threshold (float): Minimum confidence threshold for detections
            model_size (str): Model size: 'n', 's', 'm', 'l', or 'x'
            force_onnx (bool): Force using ONNX version even if ultralytics is available
            force_yolov4 (bool): Force using YOLOv4-tiny fallback
            force_yolov8core (bool): Force using YOLOv8Core implementation
        """
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes
        self.model_size = model_size.lower()
        self.force_onnx = force_onnx
        self.force_yolov4 = force_yolov4
        self.force_yolov8core = force_yolov8core
        
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
        
        # Try different implementations
        self.implementation = None
        self.model = None
        
        # 1. Try YOLOv8Core if forced (our custom implementation that doesn't require _lzma)
        if self.force_yolov8core:
            try:
                self._init_yolov8core()
                self.implementation = "yolov8core"
                return
            except Exception as e:
                print(f"Failed to initialize YOLOv8Core: {str(e)}")
        
        # 2. Try ultralytics if not forced to use alternatives
        if not self.force_onnx and not self.force_yolov4 and not self.force_yolov8core:
            try:
                self._init_ultralytics()
                self.implementation = "ultralytics"
                return
            except Exception as e:
                print(f"Failed to initialize ultralytics YOLOv8: {str(e)}")
                # Try YOLOv8Core as first fallback
                try:
                    self._init_yolov8core()
                    self.implementation = "yolov8core"
                    return
                except Exception as e:
                    print(f"Failed to initialize YOLOv8Core as fallback: {str(e)}")
        
        # 3. Try ONNX implementation if not forced to use YOLOv4
        if not self.force_yolov4:
            try:
                self._init_onnx()
                self.implementation = "onnx"
                return
            except Exception as e:
                print(f"Failed to initialize ONNX YOLOv8: {str(e)}")
        
        # 4. Fall back to YOLOv4-tiny
        try:
            self._init_yolov4_tiny()
            self.implementation = "yolov4-tiny"
            return
        except Exception as e:
            print(f"Failed to initialize YOLOv4-tiny: {str(e)}")
            raise RuntimeError("All YOLOv8 implementations failed")
    
    def _init_yolov8core(self):
        """Initialize using our custom YOLOv8Core implementation"""
        try:
            from yolov8_core import YOLOv8Core
        except ImportError:
            print("YOLOv8Core module not found!")
            raise
        
        print(f"Using YOLOv8Core implementation with model size '{self.model_size}'")
        self.model = YOLOv8Core(
            target_classes=self.target_classes,
            confidence_threshold=self.confidence_threshold,
            model_size=self.model_size
        )
        
        # We don't need to filter target classes as YOLOv8Core handles that internally
        # but we should update our target_classes to match what YOLOv8Core is using
        self.target_classes = self.model.target_classes
        
    def _init_ultralytics(self):
        """Initialize using the ultralytics package"""
        try:
            from ultralytics import YOLO
        except ImportError:
            print("Ultralytics package not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
            from ultralytics import YOLO
        
        print(f"Using ultralytics YOLOv8-{self.model_size} implementation")
        
        # Look for model in the models directory
        model_path = os.path.join(self.models_dir, f"yolov8{self.model_size}.pt")
        
        # If model doesn't exist, download it
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}")
            print(f"Attempting to use the default YOLO model path...")
            model_path = f"yolov8{self.model_size}.pt"
        
        self.model = YOLO(model_path)
        
        # Filter target classes if provided
        if self.target_classes and len(self.target_classes) > 0:
            print(f"Detecting the following classes: {', '.join(self.target_classes)}")
        else:
            print("Using all available classes")
    
    def _init_onnx(self):
        """Initialize using ONNX model with OpenCV DNN"""
        print(f"Using YOLOv8-{self.model_size} ONNX implementation with OpenCV DNN")
        
        # Define model paths
        onnx_filename = f"yolov8{self.model_size}.onnx"
        self.onnx_path = os.path.join(self.models_dir, onnx_filename)
        
        # Download the ONNX model if it doesn't exist
        if not os.path.exists(self.onnx_path):
            print(f"Downloading YOLOv8-{self.model_size} ONNX model...")
            
            # Use a mirror that provides YOLOv8 ONNX files
            onnx_url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8{self.model_size}.onnx"
            
            try:
                # Try using curl for larger files
                subprocess.run(["curl", "-L", onnx_url, "-o", self.onnx_path], check=True)
            except:
                # Fallback to urllib
                urllib.request.urlretrieve(onnx_url, self.onnx_path)
        
        # Load the model with OpenCV DNN
        self.model = cv2.dnn.readNetFromONNX(self.onnx_path)
        
        # Use GPU if available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("CUDA is available, using GPU")
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            print("CUDA is not available, using CPU")
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Filter target classes if provided
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
    
    def _init_yolov4_tiny(self):
        """Initialize using YOLOv4-tiny with OpenCV DNN as a fallback"""
        print("Falling back to YOLOv4-tiny implementation with OpenCV DNN")
        
        # Paths for YOLOv4-tiny config and weights
        self.config_path = os.path.join(self.models_dir, "yolov4-tiny.cfg")
        self.weights_path = os.path.join(self.models_dir, "yolov4-tiny.weights")
        
        # Download YOLOv4-tiny model files if they don't exist
        if not os.path.exists(self.config_path) or not os.path.exists(self.weights_path):
            print("Downloading YOLOv4-tiny model files...")
            
            config_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
            weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
            
            try:
                if not os.path.exists(self.config_path):
                    print(f"Downloading config file to {self.config_path}...")
                    urllib.request.urlretrieve(config_url, self.config_path)
                
                if not os.path.exists(self.weights_path):
                    print(f"Downloading weights file to {self.weights_path}...")
                    try:
                        # Try using curl for larger files
                        subprocess.run(["curl", "-L", weights_url, "-o", self.weights_path], check=True)
                    except:
                        # Fallback to urllib
                        urllib.request.urlretrieve(weights_url, self.weights_path)
            except Exception as e:
                print(f"Error downloading model files: {e}")
                raise RuntimeError("Failed to download YOLOv4-tiny model files")
        
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
        if self.implementation == "ultralytics":
            return self._detect_ultralytics(frame)
        elif self.implementation == "onnx":
            return self._detect_onnx(frame)
        elif self.implementation == "yolov8core":
            return self.model.detect_objects(frame)
        else:  # YOLOv4-tiny
            return self._detect_yolov4_tiny(frame)
    
    def _detect_ultralytics(self, frame):
        """Detect using ultralytics package"""
        # Run YOLOv8 inference
        results = self.model(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        # Extract detection information
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get detection details
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names[class_id]
                
                # Skip if not in target classes
                if self.target_classes and class_name.lower() not in [c.lower() for c in self.target_classes]:
                    continue
                
                # Add to detections list
                detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'box': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        return annotated_frame, detections
    
    def _detect_onnx(self, frame):
        """Detect using ONNX model with OpenCV DNN"""
        start_time = time.time()
        height, width = frame.shape[:2]
        
        # Prepare image for ONNX model (letterbox)
        input_size = (640, 640)  # Default YOLOv8 input size
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, input_size, swapRB=True, crop=False)
        self.model.setInput(blob)
        
        # Run inference
        outputs = self.model.forward(None)  # YOLOv8 ONNX model has a single output
        
        # Process detections
        detections = []
        
        # YOLOv8 ONNX output format: [batch, num_detections, num_classes+5]
        # num_classes+5 : [x, y, w, h, confidence, class_scores...]
        
        # Extract each detection
        for i in range(outputs.shape[1]):
            # Get box coordinates, confidence, and class scores
            detection = outputs[0, i, :]
            confidence = float(detection[4])
            
            if confidence < self.confidence_threshold:
                continue
            
            # Get class scores
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_confidence = float(class_scores[class_id])
            
            if class_confidence < self.confidence_threshold:
                continue
            
            # Get class name
            class_name = self.class_names[class_id]
            
            # Skip if not in target classes
            if self.target_classes and class_name.lower() not in [c.lower() for c in self.target_classes]:
                continue
            
            # Denormalize box coordinates
            x, y, w, h = detection[0:4]
            
            # ONNX models typically output centerX, centerY, width, height
            x1 = int((x - w/2) * width)
            y1 = int((y - h/2) * height)
            x2 = int((x + w/2) * width)
            y2 = int((y + h/2) * height)
            
            # Ensure coordinates are within frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            # Add to detections list
            detections.append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': class_confidence * confidence,
                'box': [x1, y1, x2, y2]
            })
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes([d['box'] for d in detections], 
                                  [d['confidence'] for d in detections], 
                                  self.confidence_threshold, 0.4)
        
        filtered_detections = []
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                det = detections[i]
                x1, y1, x2, y2 = det['box']
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label background
                label = f"{det['class_name']}: {det['confidence']:.2f}"
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - label_height - 5), (x1 + label_width, y1), (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                filtered_detections.append(det)
        
        # Draw FPS
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, filtered_detections
    
    def _detect_yolov4_tiny(self, frame):
        """Detect using YOLOv4-tiny with OpenCV DNN"""
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
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
        
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


# For testing as standalone script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test YOLOv8 wrapper")
    parser.add_argument("--input", type=str, default=0, help="Path to input image or video, or camera index")
    parser.add_argument("--model", type=str, choices=["ultralytics", "onnx", "yolov4-tiny", "yolov8core"], 
                        help="Force specific implementation")
    parser.add_argument("--size", type=str, choices=['n', 's', 'm', 'l', 'x'], default='m',
                        help="YOLOv8 model size (default: m)")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--classes", type=str, help="Comma-separated list of classes to detect")
    
    args = parser.parse_args()
    
    # Parse classes if provided
    target_classes = None
    if args.classes:
        target_classes = [c.strip() for c in args.classes.split(",")]
    
    # Determine which implementation to force
    force_onnx = args.model == "onnx"
    force_yolov4 = args.model == "yolov4-tiny"
    force_yolov8core = args.model == "yolov8core"
    
    # Initialize detector
    try:
        detector = YOLOv8Wrapper(
            target_classes=target_classes,
            confidence_threshold=args.confidence,
            model_size=args.size,
            force_onnx=force_onnx,
            force_yolov4=force_yolov4,
            force_yolov8core=force_yolov8core
        )
        
        print(f"Successfully initialized YOLOv8 wrapper using {detector.implementation} implementation")
        
        # Determine if input is an image, video, or camera
        if args.input.isdigit():
            # Camera input
            cap = cv2.VideoCapture(int(args.input))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                output_frame, detections = detector.detect_objects(frame)
                
                # Display result
                cv2.imshow("YOLOv8 Detection", output_frame)
                
                # Break loop on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        
        elif os.path.isfile(args.input):
            # Check if it's an image or video
            if args.input.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                # Image input
                image = cv2.imread(args.input)
                output_image, detections = detector.detect_objects(image)
                
                # Display result
                cv2.imshow("YOLOv8 Detection", output_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                # Save output
                output_path = f"output_{os.path.basename(args.input)}"
                cv2.imwrite(output_path, output_image)
                print(f"Output saved to {output_path}")
            
            else:
                # Video input
                cap = cv2.VideoCapture(args.input)
                
                # Get video properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Create output video writer
                output_path = f"output_{os.path.basename(args.input)}"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    output_frame, detections = detector.detect_objects(frame)
                    
                    # Write to output video
                    out.write(output_frame)
                    
                    # Display result
                    cv2.imshow("YOLOv8 Detection", output_frame)
                    
                    # Break loop on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                print(f"Output saved to {output_path}")
        
        else:
            print(f"Error: Input '{args.input}' is not a valid file or camera index")
    
    except Exception as e:
        print(f"Error: {str(e)}") 