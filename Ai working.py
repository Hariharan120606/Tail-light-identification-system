import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")



DISTANCE_THRESHOLD = 10.0

def calculate_distance(bbox_width, focal_length=1000, real_width=1.5):
    return (focal_length * real_width) / bbox_width if bbox_width > 0 else float("inf")

# Provide the path to your MP4 file
video_path = 'dashcam_footage.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

window_width, window_height = 1280, 720

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = frame.copy()

    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            if cls in [2]:  # Cars, Buses, Trucks
                bbox_width = x2 - x1
                distance = calculate_distance(bbox_width)
                
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Distance: {distance:.2f} m", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if distance < DISTANCE_THRESHOLD:
                    cv2.putText(annotated_frame, "TOO CLOSE!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 0, 255), 3)
                    print("ALERT: Car is too close!")

                car_region = annotated_frame[int(y1):int(y2), int(x1):int(x2)]
                hsv_frame = cv2.cvtColor(car_region, cv2.COLOR_BGR2HSV)

                lower_red1, upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
                lower_red2, upper_red2 = np.array([170, 120, 70]), np.array([180, 255, 255])
                mask_red = cv2.inRange(hsv_frame, lower_red1, upper_red1) | cv2.inRange(hsv_frame, lower_red2, upper_red2)
                
                # Split into left and right halves
                mid_x = car_region.shape[1] // 2
                left_half, right_half = mask_red[:, :mid_x], mask_red[:, mid_x:]
                
                # Find contours in each half
                contours_left, _ = cv2.findContours(left_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_right, _ = cv2.findContours(right_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Check for symmetry in tail lights
                def detect_taillights(contours, offset=0):
                    detected_lights = []
                    for cnt in contours:
                        if cv2.contourArea(cnt) > (bbox_width * 0.01):
                            x, y, w, h = cv2.boundingRect(cnt)
                            detected_lights.append((x + offset, y, w, h))
                    return detected_lights
                
                left_lights = detect_taillights(contours_left)
                right_lights = detect_taillights(contours_right, offset=mid_x)
                
                if 1 <= len(left_lights) + len(right_lights) <= 4:  # Max of 4 tail lights
                    for (x, y, w, h) in left_lights + right_lights:
                        cv2.rectangle(car_region, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(car_region, "Tail Light", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    resized_frame = cv2.resize(annotated_frame, (window_width, window_height))
    cv2.imshow("Tail Light Detection", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.imwrite("sample_output.png", frame)