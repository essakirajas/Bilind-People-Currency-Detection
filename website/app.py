import cv2
from ultralytics import YOLO
import pyttsx3
from flask import Flask, render_template, request
import numpy as np

# Load the YOLOv8 model
model = YOLO('yolov8.pt')

# Initialize the text-to-speech engine
engine = pyttsx3.init()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if image file is uploaded
    if 'image' not in request.files:
        return "No image uploaded", 400
    
    image_file = request.files['image']
    
    # Read image from file
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Run YOLOv8 inference on the image
    results = model(image)

# # Access the actual detection results
    detections = results[0].boxes.data
    response = ""

    class_names = []
    for det in detections:
        class_index = int(det[5])
        class_name = model.names[class_index]
        if class_name != "background":  # Exclude background class
            class_names.append(class_name)

# Concatenate the class names into a single string
    class_names_str = '\n'.join(class_names)

    return render_template('index.html', class_names=class_names_str)


if __name__ == '__main__':
    app.run(debug=True)
