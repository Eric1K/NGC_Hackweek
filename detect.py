import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# Load pre-trained Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn()
model.eval()

# Load and preprocess the image
# image_path = 'Images/image12.jpg'  # Path to the image
image_path = 'prenotedimage/Image1.jpg' 
img = Image.open(image_path)

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a tensor
])
img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Perform object detection
with torch.no_grad():
    outputs = model(img_tensor)

# Parse the outputs (bounding boxes, labels, and scores)
boxes = outputs[0]['boxes'].numpy()
labels = outputs[0]['labels'].numpy()
scores = outputs[0]['scores'].detach().numpy()

# Load the COCO labels for object names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'book', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    # Add the rest of the COCO labels here...
    'book', 'bookshelf', 'clock', 'globe'  # Example for items in your image
]

# OpenCV - Draw bounding boxes on the image
img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # Convert PIL image to OpenCV

for box, label, score in zip(boxes, labels, scores):
    if score > 0.5:  # Filter by confidence score
        x1, y1, x2, y2 = box
        label_name = COCO_INSTANCE_CATEGORY_NAMES[label]
        cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img_cv, f'{label_name}: {score:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Show the image with bounding boxes
cv2.imshow('Detected Objects', img_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
