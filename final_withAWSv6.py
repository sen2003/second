# ok2
import cv2
import boto3
import os
from datetime import datetime
from ultralytics import YOLO
import torch

# AWS Rekognition 和 S3 初始化
rekognition = boto3.client('rekognition', region_name='us-east-1')  
s3 = boto3.client('s3')
bucket_name = "img-face-rekognition"  
collection_id = "myCollection1"  

# YOLO 模型載入
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO("../yoloV8_test/yolov8n-face.pt").to(device)

# 輸入影片及輸出影片設定
input_video_path = "C:\\cvyolo\\c1029v3.mp4"
output_video_path = "output.mp4"
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    raise Exception("影片無法開啟")

# 建立資料夾存放未知和成功臉部
unknown_faces_dir = "unknown_faces"
# success_faces_dir = "success_faces"
os.makedirs(unknown_faces_dir, exist_ok=True)
# os.makedirs(success_faces_dir, exist_ok=True)

# 初始化影片寫出物件
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

            # 使用 timestamp 作為 S3 檔名
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            face_path = f"{timestamp}.jpg"
            cv2.imwrite(face_path, face_img)

            # 上傳到 S3
            s3.upload_file(face_path, bucket_name, face_path)

            try:
                # Rekognition Search Faces By Image
                response = rekognition.search_faces_by_image(
                    CollectionId=collection_id,
                    Image={'S3Object': {'Bucket': bucket_name, 'Name': face_path}},
                    MaxFaces=1,
                    FaceMatchThreshold=70  
                )

                if response['FaceMatches']:
                    # 成功辨識的情況
                    best_match = response['FaceMatches'][0]
                    name = best_match['Face']['ExternalImageId']
                    confidence = best_match['Similarity']
                    label = f"{name} ({confidence:.2f}%)"

                    # # 將成功辨識的臉儲存到 success 資料夾
                    # success_face_path = os.path.join(success_faces_dir, f"{timestamp}_{name}.jpg")
                    # cv2.imwrite(success_face_path, face_img)

                    # 標記綠色框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # 未知人臉的情況
                    label = "Unknown"
                    unknown_face_path = os.path.join(unknown_faces_dir, f"unknown_{frame_count}_{face_count}.jpg")
                    cv2.imwrite(unknown_face_path, face_img)

                    # 標記紅色框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            except rekognition.exceptions.InvalidParameterException:
                # 如果 AWS Rekognition 無法找到臉部，跳過處理
                print(f"Skipping face {face_count} at frame {frame_count}: No faces detected in image.")
            
            # 刪除臨時圖片
            os.remove(face_path)
            face_count += 1

    out.write(frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
