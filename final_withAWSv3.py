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
import os

class AreaTracker:
    """追蹤特定區域（門）的類別"""
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
        center_x = self.initial_area[0] + self.initial_area[2]/2
        center_y = self.initial_area[1] + self.initial_area[3]/2
        return (center_x, center_y)

class YOLOProcessor:
    """YOLO模型處理器"""
    def __init__(self, model_path, process_width=480, process_every_n_frames=6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        try:
            self.model = YOLO(model_path).to(self.device)
            self.model.conf = 0.5
            self.model.iou = 0.5
            if self.device.type == 'cuda':
                self.model.half()
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            sys.exit(1)
            
        self.process_width = process_width
        self.last_faces = []
        self.frame_count = 0
        self.process_every_n_frames = process_every_n_frames
    def process_frame(self, frame):
        self.frame_count += 1
        
        if self.frame_count % self.process_every_n_frames != 0:
            return self.last_faces
            
        height, width = frame.shape[:2]
        scale = self.process_width / width
        process_height = int(height * scale)
        
        small_frame = cv2.resize(frame, (self.process_width, process_height))
        
        try:
            results = self.model(small_frame)
            
            self.last_faces = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
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
    def __init__(self, collection_id, region_name='ap-northeast-1'):
        self.collection_id = collection_id
        self.rekognition = boto3.client('rekognition', region_name=region_name)
        self.recognition_cache = {}
        self.cache_timeout = 30
        self.last_clean_time = time.time()
        self.executor = ThreadPoolExecutor(max_workers=3)
        
    def clean_cache(self):
        current_time = time.time()
        if current_time - self.last_clean_time > 60:
            expired_keys = [k for k, v in self.recognition_cache.items() 
                          if current_time - v['timestamp'] > self.cache_timeout]
            for k in expired_keys:
                del self.recognition_cache[k]
            self.last_clean_time = current_time
    
    def recognize_face(self, face_image, face_key):
        if face_key in self.recognition_cache:
            cache_data = self.recognition_cache[face_key]
            if time.time() - cache_data['timestamp'] < self.cache_timeout:
                return None, cache_data['result']
        
        future = self.executor.submit(self._perform_recognition, face_image, face_key)
        return future, None
    
    def _perform_recognition(self, face_image, face_key):
        try:
            img = Image.fromarray(face_image)
            imgByteArr = io.BytesIO()
            img.save(imgByteArr, format='JPEG', quality=95)  # 提高JPEG品質
            imgBytes = imgByteArr.getvalue()
            
            response = self.rekognition.search_faces_by_image(
                CollectionId=self.collection_id,
                Image={'Bytes': imgBytes},
                MaxFaces=1,
                FaceMatchThreshold=80
            )
            
            if response['FaceMatches']:
                match = response['FaceMatches'][0]
                face_id = match['Face']['ExternalImageId']
                confidence = match['Similarity']
                result = (face_id, confidence)
            else:
                result = ("Unknown", 0)
                
            self.recognition_cache[face_key] = {
                'result': result,
                'timestamp': time.time()
            }
            
            return result
            
        except Exception as e:
            print(f"Recognition error: {e}")
            return "Error", 0

class FaceImageSaver:
    def __init__(self, base_folder="door_opener_faces", min_face_size=200):
        self.base_folder = base_folder
        self.saved_faces = set()
        self.face_count = 0
        self.min_face_size = min_face_size
        
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        
        self.today_folder = os.path.join(
            base_folder, 
            datetime.now().strftime("%Y%m%d")
        )
        if not os.path.exists(self.today_folder):
            os.makedirs(self.today_folder)
    def enhance_face_image(self, face_img):
        """增強人臉影像品質"""
        try:
            # 調整大小，確保最小邊長不小於指定大小
            height, width = face_img.shape[:2]
            scale = max(self.min_face_size / min(height, width), 1.0)
            
            if scale > 1.0:
                new_width = int(width * scale)
                new_height = int(height * scale)
                face_img = cv2.resize(face_img, (new_width, new_height), 
                                    interpolation=cv2.INTER_LANCZOS4)
            return face_img
            # 增強影像品質
            # 1. 轉換到LAB色彩空間
            # lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
            # l, a, b = cv2.split(lab)
            
            # # 2. 應用CLAHE
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            # cl = clahe.apply(l)
            
            # # 3. 合併通道
            # enhanced_lab = cv2.merge([cl, a, b])
            
            # # 4. 轉換回BGR
            # enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # # 5. 銳化
            # kernel = np.array([[-1,-1,-1],
            #                  [-1, 9,-1],
            #                  [-1,-1,-1]])
            # sharpened = cv2.filter2D(enhanced_bgr, -1, kernel)
            
            # # 6. 去噪
            # denoised = cv2.fastNlMeansDenoisingColored(sharpened,
            #                                           None,
            #                                           10,
            #                                           10,
            #                                           7,
            #                                           21)
            
            # return denoised
        except Exception as e:
            print(f"Error enhancing face image: {e}")
            # return face_img
    
    def save_face(self, face_img, face_key, additional_info=None):
        """儲存人臉影像和相關資訊"""
        try:
            if face_key not in self.saved_faces:
                # 檢查影像是否過小
                height, width = face_img.shape[:2]
                if min(height, width) < 30:
                    return
                
                self.face_count += 1
                timestamp = datetime.now().strftime("%H%M%S")
                
                # 增強影像品質
                enhanced_face = self.enhance_face_image(face_img)
                
                # 生成檔案名稱
                image_filename = f"face_{self.face_count}_{timestamp}.jpg"
                info_filename = f"face_{self.face_count}_{timestamp}.json"
                
                # 完整檔案路徑
                image_path = os.path.join(self.today_folder, image_filename)
                info_path = os.path.join(self.today_folder, info_filename)
                
                # 使用較高的JPEG品質儲存
                cv2.imwrite(image_path, enhanced_face, 
                          [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                
                # 記錄資訊
                info = {
                    "face_id": self.face_count,
                    "timestamp": datetime.now().isoformat(),
                    "image_file": image_filename,
                    "face_key": face_key,
                    "original_size": {
                        "width": width,
                        "height": height
                    },
                    "enhanced_size": {
                        "width": enhanced_face.shape[1],
                        "height": enhanced_face.shape[0]
                    }
                }
                if additional_info:
                    info.update(additional_info)
                
                with open(info_path, 'w', encoding='utf-8') as f:
                    json.dump(info, f, indent=4, ensure_ascii=False)
                
                self.saved_faces.add(face_key)
                print(f"Saved enhanced face image: {image_filename}")
                
        except Exception as e:
            print(f"Error saving face: {e}")

def find_door_openers(faces, door_center, radius=300):
    """找出在門附近圓形範圍內的所有可能開門者"""
    door_openers = []
    
    for face in faces:
        coords = face['coords']
        face_center_x = (coords[0] + coords[2]) / 2
        face_center_y = (coords[1] + coords[3]) / 2
        
        if (face_center_x - door_center[0])**2 + (face_center_y - door_center[1])**2 <= radius**2:
            door_openers.append(face)
            
    return door_openers

class DetectionLogger:
    def __init__(self, filename_prefix, save_interval=300):
        self.filename_prefix = filename_prefix
        self.detection_results = []
        self.start_time = time.time()
        self.save_interval = save_interval
        self.last_save_frame = 0
    def log_frame(self, frame_number, door_status, faces, door_openers, recognized_faces=None):
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
        
        if frame_number - self.last_save_frame >= self.save_interval:
            self.save_results()
            self.last_save_frame = frame_number
    
    def save_results(self):
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
        process_width=640,  # 增加處理寬度
        process_every_n_frames=6
    )
    
    # 初始化人臉辨識
    face_recognition = FaceRecognition(
        collection_id='your_collection_id'  # 替換為您的Collection ID
    )
    
    # 創建門的追蹤器
    tracker = AreaTracker(
        relative_x=0.74,
        relative_y=0.046,
        relative_width=0.031,
        relative_height=0.028,
        threshold_distance=50
    )
    
    # 創建記錄器和人臉儲存器
    logger = DetectionLogger("detection_results", save_interval=300)
    face_saver = FaceImageSaver(min_face_size=200)  # 設定最小人臉大小
    
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
            
            if time.time() - fps_start_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            door_opened = False
            door_openers = []
            
            if tracker.is_tracking:
                success, box = tracker.tracker.update(frame)
                if success:
                    tracker.current_area = [int(v) for v in box]
                    door_opened = tracker.check_significant_movement()
                    
                    cv2.rectangle(frame, 
                                (int(box[0]), int(box[1])),
                                (int(box[0] + box[2]), int(box[1] + box[3])),
                                (0, 0, 255), 2)
                    
                    # 畫透明半圓
                    # if door_opened:
                    #     door_center = tracker.get_door_center()
                    #     overlay = frame.copy()
                    #     cv2.circle(overlay, 
                    #              (int(door_center[0]), int(door_center[1])), 
                    #              300,
                    #              (0, 255, 255), 
                    #              -1)
                    #     cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
            faces = yolo_processor.process_frame(frame)
            
            completed_keys = []
            for face_key, future in pending_recognitions.items():
                if future.done():
                    recognition_results[face_key] = future.result()
                    completed_keys.append(face_key)
            
            for key in completed_keys:
                del pending_recognitions[key]
            
            if door_opened and faces:
                door_center = tracker.get_door_center()
                door_openers = find_door_openers(faces, door_center, 300)
                
                for face in faces:
                    is_opener = face in door_openers
                    color = (0, 0, 255) if is_opener else (0, 255, 0)
                    thickness = 3 if is_opener else 2
                    coords = face['coords']
                    
                    if is_opener:
                        # 擷取較大範圍的人臉區域
                        padding = 30
                        y1 = max(0, coords[1] - padding)
                        y2 = min(frame.shape[0], coords[3] + padding)
                        x1 = max(0, coords[0] - padding)
                        x2 = min(frame.shape[1], coords[2] + padding)
                        
                        face_img = frame[y1:y2, x1:x2]
                        
                        face_key = f"{coords[0]}_{coords[1]}"
                        
                        # 儲存人臉影像
                        additional_info = {
                            "confidence": face['confidence'],
                            "coordinates": coords,
                            "door_status": "opened",
                            "frame_number": frame_count
                        }
                        face_saver.save_face(face_img, face_key, additional_info)
                        
                        # 執行人臉辨識
                        if face_key not in recognition_results and face_key not in pending_recognitions:
                            future, cached_result = face_recognition.recognize_face(
                                face_img, face_key)
                            if future:
                                pending_recognitions[face_key] = future
                            elif cached_result:
                                recognition_results[face_key] = cached_result
                    
                    # 繪製人臉框
                    cv2.rectangle(frame, 
                                (coords[0], coords[1]),
                                (coords[2], coords[3]),
                                color, thickness)
                    
                    if is_opener:
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
                           f"Door is opened! {len(door_openers)} opener(s)", 
                           (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 記錄這一幀的結果
            logger.log_frame(frame_count, door_opened, faces, door_openers, recognition_results)
            
            # 顯示處理後的影像
            frame_resized = cv2.resize(frame, (1920,1080))  # 調整顯示大小
            cv2.imshow("Detection", frame_resized)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
    except Exception as e:
        print(f"Error during processing: {e}")
        
    finally:
        logger.save_results()
        face_recognition.executor.shutdown()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()