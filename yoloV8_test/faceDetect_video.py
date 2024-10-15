# yolo偵測人臉畫框
from ultralytics import YOLO
import cv2
import time
import torch
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = YOLO("../yoloV8_test/yolov8m-face.pt").to(device)

cap = cv2.VideoCapture("C:\yolov8_awsRekognition\C0896.mp4")
if not cap.isOpened():
    raise Exception("影片無法開啟")

# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# print(f"w: {frame_width}")
# print(f"h: {frame_height}")
results_list = []

frame_count = 0
face_count = 1
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    time_start = time.time()
    results = model(frame)

    for result in results:
        face_count = 1
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            label = f"{confidence:.2f}"

            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # detection = {
            #     "Index": face_count,
            #     "confidence": confidence,
            #     "BoundingBox": {
            #         "x1": x1,
            #         "y1": y1,
            #         "x2": x2,
            #         "y2": y2
            #     }
            # }
            face_count += 1
            # results_list.append(detection)
    frame_count += 1

    frame_resized = cv2.resize(frame, (1920, 1080))
    cv2.imshow("YOLO Detection", frame_resized)

    print("success", time.time() - time_start)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# with open('detection_results.json', 'w') as f:
#     json.dump(results_list, f, indent=4)
