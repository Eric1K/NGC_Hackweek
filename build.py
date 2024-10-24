# python yolov5/detect.py --weights yolov5/runs/train/exp9/weights/best.pt --img 640 --conf 0.25 --source prenotedimage/Image1.jpg
# python yolov5/detect.py --weights yolov5/runs/train/exp9/weights/last.pt --img 640 --conf 0.25 --source prenotedimage/Image4.jpg --conf 0.1
# python yolov5/detect.py --weights yolov5/runs/train/exp10/weights/best.pt --source 0                               # webcam



import torch
from pathlib import Path
import cv2
import matplotlib.pyplot as plt


# Load the Custom YOLOv5 model
def load_model(weights_path):
    model = torch.hub.load('yolov5', 'custom', path=weights_path, source='local')  # Use local YOLOv5 repo
    return model

# Run detection on a single image
def detect(model, img_path):
    # Load the image
    img = cv2.imread(img_path)  # Read the image with OpenCV (BGR format)
    
    # Perform inference
    results = model(img)
    
    # Show results
    results.show()  # Display image with detected boxes
    return results

if __name__ == "__main__":
    # Set the path to your trained model and image
    weights_path = 'yolov5/runs/train/exp9/weights/best.pt'  # model path
    image_path = 'prenotedimage/Image1.jpg'  # Replace with the path to your image

    # Load the yolov5 model
    model = load_model(weights_path)
    
    # Run inference on the image
    results = detect(model, image_path)
    
    # Optionally: Print or process the results
    print(results.pandas().xyxy[0])
    print(model.names)  # Print the classes the model was trained on
