from ultralytics import YOLO
import cv2
import numpy as np
import time
import torch
from datetime import datetime
import json
import sys
import boto3
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor

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
    def __init__(self, model_path, process_width=480, process_every_n_frames=6):
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
            # 設定推論參數
            self.model.conf = 0.5  # 提高信心度閾值
            self.model.iou = 0.5   # 設定IOU閾值
            if self.device.type == 'cuda':
                self.model.half()  # 使用半精度推論
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

class FaceRecognition:
    def __init__(self, collection_id, region_name='us-east-1'):
        """
        初始化 AWS Rekognition 客戶端
        @param collection_id: AWS Rekognition Collection ID
        @param region_name: AWS 區域
        """
        self.collection_id = collection_id
        self.rekognition = boto3.client('rekognition', region_name=region_name)
        self.recognition_cache = {}
        self.cache_timeout = 30  # 快取超時時間（秒）
        self.last_clean_time = time.time()
        self.executor = ThreadPoolExecutor(max_workers=3)
        
    def clean_cache(self):
        """清理過期的快取"""
        current_time = time.time()
        if current_time - self.last_clean_time > 60:  # 每分鐘清理一次
            expired_keys = [k for k, v in self.recognition_cache.items() 
                          if current_time - v['timestamp'] > self.cache_timeout]
            for k in expired_keys:
                del self.recognition_cache[k]
            self.last_clean_time = current_time
    def recognize_face(self, face_image, face_key):
        """
        辨識人臉身份
        @param face_image: numpy array 格式的人臉圖片
        @param face_key: 人臉位置的唯一標識
        @return: Future 物件
        """
        # 檢查快取
        if face_key in self.recognition_cache:
            cache_data = self.recognition_cache[face_key]
            if time.time() - cache_data['timestamp'] < self.cache_timeout:
                return None, cache_data['result']
        
        # 提交非同步任務
        future = self.executor.submit(self._perform_recognition, face_image, face_key)
        return future, None
        
    def _perform_recognition(self, face_image, face_key):
        """執行實際的辨識任務"""
        try:
            # 將 numpy array 轉換為 bytes
            img = Image.fromarray(face_image)
            imgByteArr = io.BytesIO()
            img.save(imgByteArr, format='JPEG')
            imgBytes = imgByteArr.getvalue()
            
            # 呼叫 AWS Rekognition API
            response = self.rekognition.search_faces_by_image(
                CollectionId=self.collection_id,
                Image={'Bytes': imgBytes},
                MaxFaces=1,
                FaceMatchThreshold=80
            )
            
            # 處理辨識結果
            if response['FaceMatches']:
                match = response['FaceMatches'][0]
                face_id = match['Face']['ExternalImageId']
                confidence = match['Similarity']
                result = (face_id, confidence)
            else:
                result = ("Unknown", 0)
                
            # 更新快取
            self.recognition_cache[face_key] = {
                'result': result,
                'timestamp': time.time()
            }
            
            return result
            
        except Exception as e:
            print(f"Recognition error: {e}")
            return "Error", 0

def find_door_openers(faces, door_center, radius=300):
    """
    找出在門附近圓形範圍內的所有可能開門者
    """
    door_openers = []
    
    for face in faces:
        coords = face['coords']
        # 計算人臉中心點
        face_center_x = (coords[0] + coords[2]) / 2
        face_center_y = (coords[1] + coords[3]) / 2
        
        # 判斷是否在圓形範圍內
        if (face_center_x - door_center[0])**2 + (face_center_y - door_center[1])**2 <= radius**2:
            door_openers.append(face)
            
    return door_openers

class DetectionLogger:
    """偵測結果記錄器"""
    def __init__(self, filename_prefix, save_interval=300):
        self.filename_prefix = filename_prefix
        self.detection_results = []
        self.start_time = time.time()
        self.save_interval = save_interval
        self.last_save_frame = 0
        
    def log_frame(self, frame_number, door_status, faces, door_openers, recognized_faces=None):
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
                "is_door_opener": face in door_openers,
                "identity": recognized_faces.get(
                    f"{face['coords'][0]}_{face['coords'][1]}", 
                    ("Unknown", 0)
                ) if recognized_faces and face in door_openers else None
            } for i, face in enumerate(faces)]
        }
        self.detection_results.append(frame_result)
        
        # 定期儲存
        if frame_number - self.last_save_frame >= self.save_interval:
            self.save_results()
            self.last_save_frame = frame_number
        
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
    # 初始化YOLO處理器
    yolo_processor = YOLOProcessor(
        model_path="../yoloV8_test/yolov8n-face.pt",
        process_width=480,
        process_every_n_frames=6
    )
    
    # 初始化 AWS Rekognition
    face_recognition = FaceRecognition(
        collection_id='myCollection1'  # 請替換為您的 Collection ID
    )
    
    # 創建門的追蹤器
    tracker = AreaTracker(
        relative_x=0.74,
        relative_y=0.046,
        relative_width=0.031,
        relative_height=0.028,
        threshold_distance=50
    )
    
    # 創建記錄器
    logger = DetectionLogger("detection_results", save_interval=300)
    
    # 設定顯示的縮放比例
    DISPLAY_SCALE = 0.5
    DETECTION_RADIUS = 300
    
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
    fps = 0
    
    # 儲存辨識結果和正在處理的任務
    recognition_results = {}
    pending_recognitions = {}
    
    try:
        while cap.isOpened():
            frame_start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            fps_counter += 1
            
            # 計算FPS
            if time.time() - fps_start_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            # 更新門的追蹤
            door_opened = False
            door_openers = []
            
            if tracker.is_tracking:
                success, box = tracker.tracker.update(frame)
                if success:
                    tracker.current_area = [int(v) for v in box]
                    door_opened = tracker.check_significant_movement()
                    # cv2.rectangle(frame, 
                    #             (int(box[0]), int(box[1])),
                    #             (int(box[0] + box[2]), int(box[1] + box[3])),
                    #             (0, 0, 255), 2)
                    
                    # 如果門開啟，顯示偵測範圍
                    # if door_opened:
                    #     door_center = tracker.get_door_center()
                    #     overlay = frame.copy()
                    #     cv2.circle(overlay, 
                    #              (int(door_center[0]), int(door_center[1])), 
                    #              DETECTION_RADIUS,
                    #              (0, 255, 255), 
                    #              -1)
                    #     cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                                
            # 處理人臉偵測
            faces = yolo_processor.process_frame(frame)
            
            # 檢查待處理的辨識任務
            completed_keys = []
            for face_key, future in pending_recognitions.items():
                if future.done():
                    recognition_results[face_key] = future.result()
                    completed_keys.append(face_key)
            
            # 移除已完成的任務
            for key in completed_keys:
                del pending_recognitions[key]
            
            # 如果門開啟，找出所有在範圍內的開門者
            if door_opened and faces:
                door_center = tracker.get_door_center()
                door_openers = find_door_openers(faces, door_center, DETECTION_RADIUS)
                
                # 繪製人臉
                for face in faces:
                    is_opener = face in door_openers
                    color = (0, 0, 255) if is_opener else (0, 255, 0)
                    thickness = 2 if is_opener else 1
                    coords = face['coords']
                    
                    # 繪製人臉框
                    cv2.rectangle(frame, 
                                (coords[0], coords[1]),
                                (coords[2], coords[3]),
                                color, thickness)
                    
                    if is_opener:
                        # 取得人臉影像
                        face_img = frame[coords[1]:coords[3], 
                                       coords[0]:coords[2]]
                        
                        # 檢查是否需要進行辨識
                        face_key = f"{coords[0]}_{coords[1]}"
                        if (face_key not in recognition_results and 
                            face_key not in pending_recognitions):
                            # 啟動新的辨識任務
                            future, cached_result = face_recognition.recognize_face(
                                face_img, face_key)
                            if future:
                                pending_recognitions[face_key] = future
                            elif cached_result:
                                recognition_results[face_key] = cached_result
                        
                        # 顯示辨識結果
                        if face_key in recognition_results:
                            name, confidence = recognition_results[face_key]
                            if confidence > 80:
                                identity_text = f"{name} ({confidence:.1f}%)"
                            else:
                                identity_text = "Unknown"
                        else:
                            identity_text = "Identifying..."
                        
                        cv2.putText(frame, 
                                  f"Door Opener: {identity_text}",
                                  (coords[0], coords[1] - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, color, 2)
            
            # 顯示FPS和門的狀態
            cv2.putText(frame, f"FPS: {fps}", 
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if door_opened:
                cv2.putText(frame, 
                           f"Door is opened! {len(door_openers)} opener", 
                           (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 記錄這一幀的結果
            logger.log_frame(frame_count, door_opened, faces, door_openers, recognition_results)
            
            # 顯示處理後的影像
            display_width = int(frame.shape[1] * DISPLAY_SCALE)
            display_height = int(frame.shape[0] * DISPLAY_SCALE)
            frame_resized = cv2.resize(frame, (display_width, display_height))
            cv2.imshow("Detection", frame_resized)
            
            # 控制播放速度
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
    except Exception as e:
        print(f"Error during processing: {e}")
        
    finally:
        # 清理資源
        logger.save_results()
        face_recognition.executor.shutdown()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
                    # 繪製門的框