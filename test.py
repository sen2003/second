# import torch

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
# print(torch.__version__)

#從本地端
import cv2
import boto3
import os
import time
from datetime import datetime
from ultralytics import YOLO
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def chinese_name(ExternalImageId):
    name_map = {
        "Huang": "黃士熏",
        "Ko": "柯信汶",
        "Shen": "沈宏勳",
        "Tsou": "鄒博森"
    }
    return name_map.get(ExternalImageId)

def cv2ChineseText(frame, text, position, textColor, textSize):
    if (isinstance(frame, np.ndarray)):
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame)
    fontStyle = ImageFont.truetype("C:\cvyolo\yoloV8_test\msjhbd.ttc", textSize, encoding="utf-8")
    draw.text(position, text, textColor, font=fontStyle)
    return cv2.cvtColor(np.asanyarray(frame), cv2.COLOR_RGB2BGR)
    

# AWS Rekognition 初始化
rekognition = boto3.client('rekognition', region_name='us-east-1')  
collection_id = "myCollection1"  

# YOLO 模型載入
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using device:{device}')
model = YOLO("../yoloV8_test/yolov8n-face.pt").to(device)

# 輸入影片及輸出影片設定
input_video_path = "C:\\cvyolo\\c1029v2.mp4"
output_video_path = "from_local.mp4"
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    raise Exception("影片無法開啟")

# 計時器開始
start_time = time.time()

# 建立unknoe_faces資料夾
unknown_faces_dir = "unknown_faces"
os.makedirs(unknown_faces_dir, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

frame_count = 0
face_count = 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # 擴大框選範圍
            padding = 15  # 可調整
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)
            face_img = frame[y1:y2, x1:x2]

            # 本地檢測人臉有效性
            gray_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray_face_img, scaleFactor=1.05, minNeighbors=3)

            if len(detected_faces) == 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                frame = cv2ChineseText(frame, "辨識中...", (x1, y1 - 25), (0, 0, 255), 20)
                continue
            
            try:
                # 使用 Rekognition 本地檢測人臉
                _, encoded_image = cv2.imencode('.jpg', face_img)
                response = rekognition.search_faces_by_image(
                    CollectionId=collection_id,
                    Image={'Bytes': encoded_image.tobytes()},
                    MaxFaces=10,
                    FaceMatchThreshold=20
                )

                if response['FaceMatches']:
                    best_match = response['FaceMatches'][0]
                    name = chinese_name(best_match['Face']['ExternalImageId'])
                    confidence = best_match['Similarity']
                    label = f"{name} ({confidence:.2f}%)"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    frame = cv2ChineseText(frame, label, (x1, y1 - 30), (0, 255, 0), 24)
                else:
                    label = "Unknown"
                    unknown_face_path = os.path.join(unknown_faces_dir, f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.jpg")
                    cv2.imwrite(unknown_face_path, face_img)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            except rekognition.exceptions.InvalidParameterException:
                print(f"Skipping face {face_count} at frame {frame_count}: No faces detected in image.")
            face_count += 1

    out.write(frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# 計時器結束並輸出總時間
end_time = time.time()
total_time = end_time - start_time
print(f"總共處理時間: {total_time:.2f} 秒")





