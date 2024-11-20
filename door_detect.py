import cv2
import numpy as np
from datetime import datetime

class DoorDetector:
    def __init__(self):
        """初始化門偵測器"""
        self.is_initialized = False
        self.reference_values = None
        self.door_regions = []
        self.door_status = False
        self.consecutive_detections = 0
        self.detection_threshold = 3
        self.change_threshold = 30  # 判斷變化的閾值
        self.grid_size = (9, 4)  # 網格大小
        self.stable_frames = 0
        self.stable_threshold = 40  # 穩定幀數閾值
        
    def initialize(self, frame):
        """初始化參考值"""
        height, width = frame.shape[:2]
        cell_height = height // self.grid_size[1]
        cell_width = width // self.grid_size[0]
        
        # 在影像右上方建立一個檢測區域
        self.door_regions = []
        for i in range(1):  
            x = width - (i + 2) * cell_width
            y = i * cell_height
            self.door_regions.append((
                x, y,
                cell_width, cell_height
            ))
        
        # 計算初始參考值
        self.reference_values = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for x, y, w, h in self.door_regions:
            roi = gray[y:y+h, x:x+w]
            avg_value = np.mean(roi)
            self.reference_values.append(avg_value)
            
        self.is_initialized = True
        
    def update_reference(self, frame):
        """更新參考值"""
        if not self.door_status:
            self.stable_frames += 1
            if self.stable_frames >= self.stable_threshold:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for i, (x, y, w, h) in enumerate(self.door_regions):
                    roi = gray[y:y+h, x:x+w]
                    self.reference_values[i] = np.mean(roi)
                self.stable_frames = 0
                
    def process_frame(self, frame):
        """處理每一幀影像"""
        if not self.is_initialized:
            self.initialize(frame)
            
        output_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 檢查所有區域的變化
        changes_detected = 0
        
        for i, (x, y, w, h) in enumerate(self.door_regions):
            # 計算當前區域的平均值
            roi = gray[y:y+h, x:x+w]
            current_value = np.mean(roi)
            
            # 計算與參考值的差異
            difference = abs(current_value - self.reference_values[i])
            
            # 繪製檢測區域
            color = (0, 255, 0) if difference < self.change_threshold else (0, 0, 255)
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), color, 2)
            
            # 顯示區域的平均值和差異
            cv2.putText(output_frame, 
                    f"Gray: {current_value:.1f}", 
                    (10, 90),  # 灰度值
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (255, 0, 0), 
                    2)
            cv2.putText(output_frame, 
                    f"Difference: {difference:.1f}", 
                    (10, 60),  # 差異值
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    color, 
                    2)

            
            if difference > self.change_threshold:
                changes_detected += 1
                
        # 更新門的狀態
        if changes_detected >= len(self.door_regions):
            self.consecutive_detections += 1
            if self.consecutive_detections >= self.detection_threshold:
                if not self.door_status:
                    print(f"Door opened at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.door_status = True
        else:
            self.consecutive_detections = 0
            self.door_status = False
            
        # 當門關閉且場景穩定時更新參考值
        # self.update_reference(frame)
        
        # 門開啟時顯示警示區域
        if self.door_status:
            overlay = output_frame.copy()
            warning_area = self.get_warning_area(frame.shape[1], frame.shape[0])
            cv2.rectangle(overlay, 
                        (warning_area[0], warning_area[1]),
                        (warning_area[2], warning_area[3]),
                        (0, 255, 255), 
                        -1)
            cv2.addWeighted(overlay, 0.3, output_frame, 0.7, 0, output_frame)
            
        # 顯示門的狀態
        status_text = "Door is OPENED" if self.door_status else "Door: CLOSED"
        cv2.putText(output_frame, 
                   status_text, 
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1, 
                   (0, 0, 255) if self.door_status else (0, 255, 0), 
                   2)
                   
        return output_frame
        
    def get_warning_area(self, frame_width, frame_height):
        """計算警示區域"""
        # 在門的位置右側創建警示區域
        margin_x = int(frame_width * 0.2)
        margin_y = int(frame_height * 0.08)
        
        left = frame_width - int(frame_width * 0.34)
        top = margin_y
        right = frame_width - margin_x
        bottom = frame_height - margin_y-500
        
        return (left, top, right, bottom)

def main():
    # 開啟影片
    cap = cv2.VideoCapture("C:\\cvyolo\\C1029v2.mp4")
    if not cap.isOpened():
        raise Exception("無法開啟影片")
        
    # 創建門偵測器
    detector = DoorDetector()

    # 獲取輸入影片的幀率和分辨率
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    # 初始化視頻寫入器
    output_path = "demo.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 編碼器選擇，使用 mp4 格式
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("結束")
            break
            
        # 處理影像
        output_frame = detector.process_frame(frame)
        
        # 寫入處理後的影像到輸出視頻
        out.write(output_frame)

        # 顯示結果
        cv2.imshow('Door Detection', output_frame)
        
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # 釋放資源
    cap.release()
    out.release()  # 確保視頻文件保存完成
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
