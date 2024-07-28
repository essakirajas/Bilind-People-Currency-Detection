# The yolov8 have only the model for 100 and 50 only.
# The YoloV8x has the mmodel for all Currency.


import cv2
from ultralytics import YOLO
import pyttsx3

# Load the YOLOv8 model
model = YOLO('yolov8.pt')

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Path to the image file
image_path = 'D:\\projects\\currency_detection\\dataset\\10 annot\\IMG_1632.JPG'


# Read the image
frame = cv2.imread(image_path)

# Check if the image is loaded successfully
if frame is None:
    print(f"Error: Could not read the image at {image_path}. Please check the file path.")
    exit()

# Run YOLOv8 inference on the image
results = model(frame)

# Access the actual detection results
detections = results.pred[0]

# Iterate over detections
for detection in detections:
    # Get class ID and confidence
    class_id = int(detection[5])
    confidence = float(detection[4])
    
    # Filter for currency class (modify according to your class index)
    if class_id == 1 and confidence > 0.5:  # Assuming currency class ID is 1
        # Get currency label
        currency_label = model.names[class_id]
        # Speak the detected currency
        engine.say(f"This is {currency_label}")
        engine.runAndWait()

# Visualize the results on the image
annotated_frame = results.imgs[0]

# Display the annotated image
cv2.imshow("YOLOv8 Inference", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
