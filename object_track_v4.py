import cv2
import numpy as np
import os

class AreaTracker:
    def __init__(self, relative_x, relative_y, relative_width, relative_height, threshold_distance=50):
        """
        初始化追蹤器
        
        參數:
        relative_x: float, 相對X座標 (0.0 ~ 1.0)
        relative_y: float, 相對Y座標 (0.0 ~ 1.0)
        relative_width: float, 相對寬度 (0.0 ~ 1.0)
        relative_height: float, 相對高度 (0.0 ~ 1.0)
        threshold_distance: int, 移動閾值（像素）
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
        
        # 初始化人臉檢測器
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # 是否已儲存人臉
        self.face_saved = False
        
    def initialize_with_frame(self, frame):
        """根據影片尺寸初始化追蹤區域"""
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # 將相對座標轉換為像素座標
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
    
    def find_nearest_face(self, frame):
        """檢測最近的人臉"""
        if self.face_saved:  # 如果已經儲存過人臉，就不再檢測
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
            
        # 計算每個人臉到初始區域的距離
        door_center_x = self.initial_area[0] + self.initial_area[2]/2
        door_center_y = self.initial_area[1] + self.initial_area[3]/2
        
        min_distance = float('inf')
        nearest_face = None
        
        for face in faces:
            x, y, w, h = face
            face_center_x = x + w/2
            face_center_y = y + h/2
            
            # 計算距離
            distance = np.sqrt((face_center_x - door_center_x)**2 + 
                             (face_center_y - door_center_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_face = face
                
        return nearest_face

    def save_face_image(self, frame, face):
        """儲存最近的人臉"""
        if not self.face_saved:  # 只在還沒儲存過時儲存
            x, y, w, h = face
            face_img = frame[y:y+h, x:x+w]
            
            # 確保face_images資料夾存在
            if not os.path.exists('face_images'):
                os.makedirs('face_images')
                
            filename = 'face_images/nearest_face.jpg'
            cv2.imwrite(filename, face_img)
            self.face_saved = True
            return filename
        return None

def main():
    cap = cv2.VideoCapture("C:\cvyolo\C0896.mp4")
    if not cap.isOpened():
        raise Exception("影片無法開啟")

    tracker = AreaTracker(
        relative_x=0.74,      # 相對X座標
        relative_y=0.046,     # 相對Y座標
        relative_width=0.031, # 相對寬度
        relative_height=0.028,# 相對高度
        threshold_distance=70  # 移動閾值
    )

    ret, frame = cap.read()
    if ret:
        tracker.initialize_with_frame(frame)
        tracker.tracker.init(frame, tuple(tracker.initial_area))
        tracker.is_tracking = True

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法接收影像")
            break

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
                
                # 檢查是否有顯著移動
                if tracker.check_significant_movement():
                    cv2.putText(frame, "Door is opened!", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # 尋找最近的人臉
                    nearest_face = tracker.find_nearest_face(frame)
                    if nearest_face is not None:
                        x, y, w, h = nearest_face
                        # 繪製人臉框（藍色）
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        
                        # 儲存人臉圖片
                        filename = tracker.save_face_image(frame, nearest_face)
                        if filename:
                            cv2.putText(frame, "Face saved", (x, y-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # 顯示移動距離
                distance = tracker.calculate_distance()
                cv2.putText(frame, f"Distance: {distance:.2f}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 顯示影像
        cv2.imshow('Tracking', frame)
        
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()