import boto3
import json
import sys
import time
import torch
import cv2
from ultralytics import YOLO

# 定義中文名稱映射


def chinese_name(externalImageId):
    name_map = {
        "Huang": "黃士熏",
        "Ke": "柯信汶",
        "Shen": "沈宏勳",
        "Tsou": "鄒博森"
    }
    return name_map.get(externalImageId, "Unknow person")


# 檢查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加載YOLO模型
model = YOLO("../yoloV8_test/yolov8m-face.pt").to(device)

# 打開視頻文件
cap = cv2.VideoCapture("D:\\yolov8_awsRekognition\\yoloV8_test\\C0885v2.MP4")
if not cap.isOpened():
    raise Exception("影片無法開啟")

# 讀取AWS Rekognition的結果
with open('rekognition_results.json', 'r', encoding='utf-8') as f:
    rekognition_results = json.load(f)

# 函數: 檢查YOLO檢測框是否與Rekognition框重疊


def check_overlap(yolo_box, rek_box):
    x1, y1, x2, y2 = yolo_box
    rx1 = rek_box["Left"]
    ry1 = rek_box["Top"]
    rx2 = rek_box["Left"] + rek_box["Width"]
    ry2 = rek_box["Top"] + rek_box["Height"]

    overlap = not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2)
    return overlap


# 開始處理視頻
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    time_start = time.time()
    results = model(frame)

    # 繪製YOLO檢測到的框
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0].item()
            label = f"{confidence:.2f}"

            # 檢查是否有Rekognition結果重疊
            matched = False
            for rek_result in rekognition_results:
                timestamp = rek_result["Timestamp"]
                if frame_count == int(timestamp / 33.33):  # assuming 30 fps
                    rek_box = rek_result["BoundingBox"]
                    name = rek_result["Name"]
                    if name != "Unknow" and check_overlap((x1, y1, x2, y2), rek_box):
                        similarity = float(rek_result["Similarity"])
                        label = f"{name}: {similarity:.2f}%"
                        cv2.rectangle(frame, (x1, y1),
                                      (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        matched = True
                        break

            # 如果沒有匹配到Rekognition結果，使用綠色框
            if not matched:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    frame_count += 1

    frame_resized = cv2.resize(frame, (1920, 1080))
    cv2.imshow("YOLO Detection", frame_resized)

    print("success", time.time() - time_start)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
