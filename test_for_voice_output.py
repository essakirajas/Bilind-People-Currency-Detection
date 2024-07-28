# The yolov8 have only the model for 100 and 50 only.
# The YoloV8x has the mmodel for all Currency.

import cv2
from ultralytics import YOLO
import pyttsx3

# Load the YOLOv8 model
model = YOLO('yolov8.pt')

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Open the default camera (usually the webcam)
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Loop through the video frames
while True:
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
      
        # Access the actual detection results
        detections = results[0].boxes.data

        # Print class names and bounding box coordinates
        for det in detections:
            class_name = model.names[int(det[5])]
            confidence = det[4]
            print(f"Class: {class_name}, Confidence: {confidence}, BBox: {det[:4]}")

            # Draw bounding box on the frame
            bbox = det[:4].cpu().numpy().astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name}: {confidence:.2f}", (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Speak the class name
            engine.say(f"The detected object is {class_name}")
            engine.runAndWait()

        # Display the frame with bounding boxes
        cv2.imshow("YOLOv8 Inference", frame)

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

# # Open the default camera (usually the webcam)
# cap = cv2.VideoCapture(0)

# # Check if the camera is opened successfully
# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()

# # Loop through the video frames
# while True:
#     # Read a frame from the video
#     success, frame = cap.read()

#     if success:
#         # Run YOLOv8 inference on the frame
#         results = model(frame)

#         # Access the actual detection results
#         detections = results[0].boxes.data

#         # Print class names and bounding box coordinates (excluding background class)
#         for det in detections:
#             class_index = int(det[5])
#             # Exclude background class (class index 0)
#             if class_index != 0:
#                 class_name = model.names[class_index]
#                 confidence = det[4]
#                 print(f"Class: {class_name}, Confidence: {confidence}, BBox: {det[:4]}")

#                 # Draw bounding box on the frame
#                 bbox = det[:4].cpu().numpy().astype(int)
#                 cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
#                 cv2.putText(frame, f"{class_name}: {confidence:.2f}", (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#                 # Speak the class name
#                 engine.say(f"The detected object is {class_name}")
#                 engine.runAndWait()

#         # Display the frame with bounding boxes
#         cv2.imshow("YOLOv8 Inference", frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if reading the frame fails
#         break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()
