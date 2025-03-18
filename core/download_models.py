import os
import urllib.request
import sys

def download_file(url, destination):
    """
    Download a file from a URL to a destination path.
    """
    print(f"Downloading {url} to {destination}...")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"Successfully downloaded to {destination}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create models directory within the core directory if it doesn't exist
    models_dir = os.path.join(script_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created '{models_dir}' directory")
    
    # URLs for the model files
    model_files = {
        # MobileNet SSD model files
        "MobileNetSSD_deploy.prototxt": "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
        "MobileNetSSD_deploy.caffemodel": "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel",
        
        # YOLOv4-tiny model files
        "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
        "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
        "coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
    }
    
    # Download each file
    success_count = 0
    total_files = len(model_files)
    
    for filename, url in model_files.items():
        destination = os.path.join(models_dir, filename)
        if download_file(url, destination):
            success_count += 1
    
    # Report results
    if success_count == total_files:
        print("\nAll model files downloaded successfully!")
        print("Models are now stored in the core/models directory.")
        print("\nYou can run the object detection with:")
        print("  - python main.py webcam --model mobilenet (for MobileNet SSD)")
        print("  - python main.py webcam --model yolo (for YOLOv4-tiny)")
        print("  - python main.py webcam --model yolov8 (for YOLOv8)")
        return 0
    else:
        print(f"\n{success_count} out of {total_files} files downloaded successfully.")
        print("Some downloads failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 