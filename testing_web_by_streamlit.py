# The yolov8 have only the model for 100 and 50 only.
# The YoloV8x has the mmodel for all Currency.


import cv2
import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8.pt')

# Streamlit app
st.title("Currency Detector Web App")

# File uploader for image selection
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image file
    image = Image.open(uploaded_file)
    
    # Convert the image to an array
    image_np = np.array(image)

    # Convert the image array to BGR (OpenCV format)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Run YOLOv8 inference on the image
    results = model(image_bgr)

    # Visualize the results on the image
    annotated_image = results[0].plot()

    # Display the original and annotated images
    st.image([image, annotated_image], caption=['Original Image', 'YOLOv8 Inference'], use_column_width=True)

