import cv2
import numpy as np

class AreaTracker:
    def __init__(self, relative_x, relative_y, relative_width, relative_height, threshold_distance=50):
        """
        @param relative_x: 相對X座標 (0.0 ~ 1.0)
        @param relative_y: 相對Y座標 (0.0 ~ 1.0)
        @param relative_width: 相對寬度 (0.0 ~ 1.0)
        @param relative_height: 相對高度 (0.0 ~ 1.0)
        @param threshold_distance: 移動閾值（像素）
        """
        # 確保所有輸入值在0.0到1.0之間
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
        
        # 將相對座標轉換為像素座標
        area_x = int(self.relative_x * self.frame_width)
        area_y = int(self.relative_y * self.frame_height)
        area_width = int(self.relative_width * self.frame_width)
        area_height = int(self.relative_height * self.frame_height)
        
        self.initial_area = [area_x, area_y, area_width, area_height]
        self.current_area = self.initial_area.copy()
        
    def calculate_distance(self):
        # 計算當前位置與初始位置的距離
        current_center_x = self.current_area[0] + self.current_area[2]/2
        current_center_y = self.current_area[1] + self.current_area[3]/2
        initial_center_x = self.initial_area[0] + self.initial_area[2]/2
        initial_center_y = self.initial_area[1] + self.initial_area[3]/2
        
        distance = np.sqrt((current_center_x - initial_center_x)**2 + 
                         (current_center_y - initial_center_y)**2)
        return distance

    def check_significant_movement(self):
        # 檢查移動是否超過閾值
        return self.calculate_distance() > self.threshold_distance

def main():
    # 開啟視頻
    cap = cv2.VideoCapture("C:\cvyolo\C0896.mp4")
    if not cap.isOpened():
        raise Exception("影片無法開啟")

    # 創建追蹤器實例（使用0.0~1.0的比例值）
    tracker = AreaTracker(
        relative_x=0.74,      # 相對X座標 (原本1420位置約為0.74)
        relative_y=0.046,     # 相對Y座標 (原本50位置約為0.046)
        relative_width=0.031, # 相對寬度 (原本60像素約為0.031)
        relative_height=0.028,# 相對高度 (原本30像素約為0.028)
        threshold_distance=70  # 移動閾值（像素）
    )

    # 讀取第一幀並初始化追蹤器
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
            # 更新追蹤器
            success, box = tracker.tracker.update(frame)
            
            if success:
                # 更新當前位置
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
                    cv2.putText(frame, "door is opened!", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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