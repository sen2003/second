from ultralytics import YOLO
import cv2
import numpy as np
import time
import torch

class AreaTracker:
    def __init__(self, relative_x, relative_y, relative_width, relative_height, threshold_distance=50):
        self.relative_x = max(0.0, min(1.0, relative_x))
        self.relative_y = max(0.0, min(1.0, relative_y))
        self.relative_width = max(0.0, min(1.0, relative_width))
        self.relative_height = max(0.0, min(1.0, relative_height))
        
        self.threshold_distance = threshold_distance
        self.tracker = cv2.TrackerCSRT_create()
        self.is_tracking = False
        self.initial_area = None
        self.current_area = None
        self.frame_width = None
        self.frame_height = None

    def initialize_with_frame(self, frame):
        self.frame_height, self.frame_width = frame.shape[:2]
        
        area_x = int(self.relative_x * self.frame_width)
        area_y = int(self.relative_y * self.frame_height)
        area_width = int(self.relative_width * self.frame_width)
        area_height = int(self.relative_height * self.frame_height)
        
        self.initial_area = [area_x, area_y, area_width, area_height]
        self.current_area = self.initial_area.copy()
        
    def calculate_distance(self):
        current_center_x = self.current_area[0] + self.current_area[2]/2
        current_center_y = self.current_area[1] + self.current_area[3]/2
        initial_center_x = self.initial_area[0] + self.initial_area[2]/2
        initial_center_y = self.initial_area[1] + self.initial_area[3]/2
        
        distance = np.sqrt((current_center_x - initial_center_x)**2 + 
                         (current_center_y - initial_center_y)**2)
        return distance
    def check_significant_movement(self):
        return self.calculate_distance() > self.threshold_distance

    def get_door_center(self):
        # 返回門的中心點座標
        center_x = self.initial_area[0] + self.initial_area[2]/2
        center_y = self.initial_area[1] + self.initial_area[3]/2
        return (center_x, center_y)

def find_closest_face_fast(faces, door_x):
    min_distance = float('inf')
    closest_face = None
    
    for face in faces:
        # 計算人臉中心點的X座標
        face_center_x = (face[0] + face[2]) / 2
        # 計算與門的X軸距離
        distance = abs(face_center_x - door_x)
        
        if distance < min_distance:
            min_distance = distance
            closest_face = face
            
    return closest_face

def main():
    # 初始化YOLO模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = YOLO("../yoloV8_test/yolov8n-face.pt").to(device)

    # 開啟視頻
    cap = cv2.VideoCapture("C:\cvyolo\C0896.mp4")
    if not cap.isOpened():
        raise Exception("影片無法開啟")

    # 創建門的追蹤器
    tracker = AreaTracker(
        relative_x=0.74,
        relative_y=0.046,
        relative_width=0.031,
        relative_height=0.028,
        threshold_distance=70
    )

    # 初始化追蹤器
    ret, frame = cap.read()
    if ret:
        tracker.initialize_with_frame(frame)
        tracker.tracker.init(frame, tuple(tracker.initial_area))
        tracker.is_tracking = True

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        time_start = time.time()
        
        # 更新門的追蹤
        door_opened = False
        if tracker.is_tracking:
            success, box = tracker.tracker.update(frame)
            if success:
                tracker.current_area = [int(v) for v in box]
                
                # 繪製當前追蹤框（紅色）
                p1 = (int(box[0]), int(box[1]))
                p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)
                
                # 繪製初始區域（綠色）
                cv2.rectangle(frame, 
                            (tracker.initial_area[0], tracker.initial_area[1]),
                            (tracker.initial_area[0] + tracker.initial_area[2], 
                             tracker.initial_area[1] + tracker.initial_area[3]),
                            (0, 255, 0), 2)
                
                door_opened = tracker.check_significant_movement()
                if door_opened:
                    cv2.putText(frame, "Door is opened!", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 執行人臉偵測
        results = model(frame)
        
        # 獲取門的X座標位置（門的中心點）
        door_x = tracker.initial_area[0] + tracker.initial_area[2]/2
        
        # 收集所有人臉座標
        faces = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                faces.append([x1, y1, x2, y2])
                
                # 如果門沒開，所有人臉都畫綠框
                if not door_opened:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 如果門開了，找出最近的人臉並標示
        if door_opened and faces:
            closest_face = find_closest_face_fast(faces, door_x)
            
            # 畫出所有人臉
            for face in faces:
                # 判斷是否為最近的人臉
                is_closest = face == closest_face
                color = (0, 0, 255) if is_closest else (0, 255, 0)
                thickness = 3 if is_closest else 2
                
                cv2.rectangle(frame, 
                            (face[0], face[1]),
                            (face[2], face[3]),
                            color, thickness)
                
                # 為最近的人臉加上標籤
                if is_closest:
                    cv2.putText(frame, "Door Opener", 
                              (face[0], face[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 顯示處理後的影像
        frame_resized = cv2.resize(frame, (1920, 1080))
        cv2.imshow("Detection", frame_resized)
        
        print(f"Frame processed in {time.time() - time_start:.3f} seconds")
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()