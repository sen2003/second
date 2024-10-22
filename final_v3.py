from ultralytics import YOLO
import cv2
import numpy as np
import time
import torch
from datetime import datetime
import json
import sys

class AreaTracker:
    """追蹤特定區域（門）的類別"""
    def __init__(self, relative_x, relative_y, relative_width, relative_height, threshold_distance=50):
        """
        初始化追蹤器
        @param relative_x: 相對X座標 (0.0 ~ 1.0)
        @param relative_y: 相對Y座標 (0.0 ~ 1.0)
        @param relative_width: 相對寬度 (0.0 ~ 1.0)
        @param relative_height: 相對高度 (0.0 ~ 1.0)
        @param threshold_distance: 移動閾值（像素）
        """
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
        """根據影片尺寸初始化追蹤區域"""
        self.frame_height, self.frame_width = frame.shape[:2]
        
        area_x = int(self.relative_x * self.frame_width)
        area_y = int(self.relative_y * self.frame_height)
        area_width = int(self.relative_width * self.frame_width)
        area_height = int(self.relative_height * self.frame_height)
        
        self.initial_area = [area_x, area_y, area_width, area_height]
        self.current_area = self.initial_area.copy()
        
    def calculate_distance(self):
        """計算當前位置與初始位置的距離"""
        current_center_x = self.current_area[0] + self.current_area[2]/2
        current_center_y = self.current_area[1] + self.current_area[3]/2
        initial_center_x = self.initial_area[0] + self.initial_area[2]/2
        initial_center_y = self.initial_area[1] + self.initial_area[3]/2
        
        distance = np.sqrt((current_center_x - initial_center_x)**2 + 
                         (current_center_y - initial_center_y)**2)
        return distance

    def check_significant_movement(self):
        """檢查移動是否超過閾值"""
        return self.calculate_distance() > self.threshold_distance

    def get_door_center(self):
        """獲取門的中心點座標"""
        center_x = self.initial_area[0] + self.initial_area[2]/2
        center_y = self.initial_area[1] + self.initial_area[3]/2
        return (center_x, center_y)

class YOLOProcessor:
    """YOLO模型處理器"""
    def __init__(self, model_path, process_width=640, process_every_n_frames=3):
        """
        初始化YOLO處理器
        @param model_path: YOLO模型路徑
        @param process_width: 處理寬度
        @param process_every_n_frames: 每N幀處理一次
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        try:
            self.model = YOLO(model_path).to(self.device)
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            sys.exit(1)
            
        self.process_width = process_width
        self.last_faces = []
        self.frame_count = 0
        self.process_every_n_frames = process_every_n_frames
        
    def process_frame(self, frame):
        """
        處理單一幀
        @param frame: 輸入幀
        @return: 人臉位置列表
        """
        self.frame_count += 1
        
        # 只在特定幀數進行處理
        if self.frame_count % self.process_every_n_frames != 0:
            return self.last_faces
            
        # 計算縮放比例
        height, width = frame.shape[:2]
        scale = self.process_width / width
        process_height = int(height * scale)
        
        # 縮放圖片
        small_frame = cv2.resize(frame, (self.process_width, process_height))
        
        try:
            # 執行偵測
            results = self.model(small_frame)
            
            # 處理結果
            self.last_faces = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    # 轉換回原始解析度
                    x1, y1, x2, y2 = map(lambda x: int(x/scale), [x1, y1, x2, y2])
                    confidence = box.conf[0].item()
                    self.last_faces.append({
                        'coords': [x1, y1, x2, y2],
                        'confidence': confidence
                    })
        except Exception as e:
            print(f"Error processing frame: {e}")
            
        return self.last_faces
def find_closest_face_fast(faces, door_x):
    """
    快速找出最接近門的人臉
    @param faces: 人臉列表
    @param door_x: 門的X座標
    @return: 最接近的人臉資訊
    """
    min_distance = float('inf')
    closest_face = None
    
    for face in faces:
        coords = face['coords']
        # 計算人臉中心點的X座標
        face_center_x = (coords[0] + coords[2]) / 2
        # 計算與門的X軸距離
        distance = abs(face_center_x - door_x)
        
        if distance < min_distance:
            min_distance = distance
            closest_face = face
            
    return closest_face

class DetectionLogger:
    """偵測結果記錄器"""
    def __init__(self, filename_prefix):
        self.filename_prefix = filename_prefix
        self.detection_results = []
        self.start_time = time.time()
        
    def log_frame(self, frame_number, door_status, faces, closest_face):
        """記錄單一幀的偵測結果"""
        frame_result = {
            "frame_number": frame_number,
            "timestamp": time.time() - self.start_time,
            "door_status": "open" if door_status else "closed",
            "faces": [{
                "face_id": i,
                "confidence": face['confidence'],
                "coordinates": {
                    "x1": face['coords'][0],
                    "y1": face['coords'][1],
                    "x2": face['coords'][2],
                    "y2": face['coords'][3]
                },
                "is_door_opener": (face == closest_face)
            } for i, face in enumerate(faces)]
        }
        self.detection_results.append(frame_result)
        
    def save_results(self):
        """儲存偵測結果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.filename_prefix}_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.detection_results, f, indent=4)
            print(f"Detection results saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")

def main():
    """主程式"""
    # 初始化YOLO處理器
    yolo_processor = YOLOProcessor(
        model_path="../yoloV8_test/yolov8n-face.pt",
        process_width=640,
        process_every_n_frames=3
    )
    
    # 創建門的追蹤器
    tracker = AreaTracker(
        relative_x=0.74,
        relative_y=0.046,
        relative_width=0.031,
        relative_height=0.028,
        threshold_distance=70
    )
    
    # 創建記錄器
    logger = DetectionLogger("detection_results")
    
    # 開啟視頻
    try:
        cap = cv2.VideoCapture("C:\cvyolo\C0896.mp4")
        if not cap.isOpened():
            raise Exception("影片無法開啟")
    except Exception as e:
        print(f"Error opening video: {e}")
        return
    
    # 初始化追蹤器
    ret, frame = cap.read()
    if ret:
        tracker.initialize_with_frame(frame)
        tracker.tracker.init(frame, tuple(tracker.initial_area))
        tracker.is_tracking = True
    
    frame_count = 0
    fps_start_time = time.time()
    fps_counter = 0
    current_fps = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            fps_counter += 1
            current_time = time.time()
            
            # 計算FPS
            if current_time - fps_start_time >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_start_time = current_time
            
            # 更新門的追蹤
            door_opened = False
            closest_face = None
            
            if tracker.is_tracking:
                success, box = tracker.tracker.update(frame)
                if success:
                    tracker.current_area = [int(v) for v in box]
                    door_opened = tracker.check_significant_movement()
                    
                    # 繪製門的框
                    cv2.rectangle(frame, 
                                (int(box[0]), int(box[1])),
                                (int(box[0] + box[2]), int(box[1] + box[3])),
                                (0, 0, 255), 2)
            # 處理人臉偵測
            faces = yolo_processor.process_frame(frame)
            
            # 如果門開啟，找出最近的人臉
            if door_opened and faces:
                door_x = tracker.initial_area[0] + tracker.initial_area[2]/2
                closest_face = find_closest_face_fast(faces, door_x)
                
                # 繪製人臉
                for face in faces:
                    is_closest = face == closest_face
                    color = (0, 0, 255) if is_closest else (0, 255, 0)
                    thickness = 3 if is_closest else 2
                    coords = face['coords']
                    
                    cv2.rectangle(frame, 
                                (coords[0], coords[1]),
                                (coords[2], coords[3]),
                                color, thickness)
                    
                    if is_closest:
                        cv2.putText(frame, "Door Opener", 
                                  (coords[0], coords[1] - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 記錄這一幀的結果
            logger.log_frame(frame_count, door_opened, faces, closest_face)
            
            # 顯示狀態資訊
            cv2.putText(frame, f"FPS: {current_fps}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if door_opened:
                cv2.putText(frame, "Door is opened!", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 顯示處理後的影像
            frame_resized = cv2.resize(frame, (1920, 1080))
            cv2.imshow("Detection", frame_resized)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
    except Exception as e:
        print(f"Error during processing: {e}")
        
    finally:
        # 儲存結果並釋放資源
        logger.save_results()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()