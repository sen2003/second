from ultralytics import YOLO
import cv2
import time
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = YOLO("../yoloV8_test/yolov8m-face.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    resized_frame = cv2.resize(frame, (640, 640))

    frame_tensor = torch.from_numpy(resized_frame).permute(
        2, 0, 1).unsqueeze(0).float().to(device)
    frame_tensor /= 255.0

    time_start = time.time()

    result = model.predict(frame_tensor, show=False)

    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls = int(box.cls.item())
            conf = box.conf.item()
            label = f"{conf:.2f}"
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(resized_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    cv2.imshow("YOLO Detection", cv2.resize(resized_frame, (1280, 720)))
    # cv2.imshow("YOLO Detection", resized_frame)

    print("success", time.time() - time_start)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
