# The yolov8 have only the model for 100 and 50 only.
# The YoloV8x has the mmodel for all Currency.

import cv2
from ultralytics import YOLO

model = YOLO('yolov8.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:

    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if reading the frame fails
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()




# import cv2
# from ultralytics import YOLO
# import pyttsx3

# # Load the YOLOv8 model
# model = YOLO('yolov8x.pt')

# # Initialize the text-to-speech engine
# engine = pyttsx3.init()

# # Path to the image file
# image_path = 'IMG_20240209_102942.jpg'

# # Read the image
# frame = cv2.imread(image_path)

# # Check if the image is loaded successfully
# if frame is None:
#     print(f"Error: Could not read the image at {image_path}. Please check the file path.")
#     exit()

# # Run YOLOv8 inference on the image
# results = model(frame)

# # Access the actual detection results
# detections = results[0].boxes.data

# # Print class names and bounding box coordinates
# for det in detections:
#     class_name = model.names[int(det[5])]
#     confidence = det[4]
#     print(f"Class: {class_name}, Confidence: {confidence}, BBox: {det[:4]}")

#     # Speak the class name
#     engine.say(f"The detected object is {class_name}")
#     engine.runAndWait()

# # Visualize the results on the image
# annotated_frame = results[0].plot()

# # Display the annotated image
# cv2.imshow("YOLOv8 Inference", annotated_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
