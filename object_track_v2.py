import cv2
import numpy as np

class AreaTracker:
    def __init__(self, area_x, area_y, area_width, area_height, threshold_distance=50):
        # 初始化固定區域的位置和大小
        self.initial_area = [area_x, area_y, area_width, area_height]
        self.current_area = self.initial_area.copy()
        self.threshold_distance = threshold_distance
        self.tracker = cv2.TrackerCSRT_create()
        self.is_tracking = False
        
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

    # 創建追蹤器實例（這裡設定固定區域，可依需求修改）
    tracker = AreaTracker(
        area_x=1420,     # 起始X座標
        area_y=50,     # 起始Y座標
        area_width=60, # 寬度
        area_height=30,# 高度
        threshold_distance=50  # 移動閾值（像素）
    )

    # 讀取第一幀並初始化追蹤器
    ret, frame = cap.read()
    if ret:
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
                    # 在畫面上顯示警告
                    cv2.putText(frame, "DOOR IS OPEN!", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # 這裡可以添加其他動作，例如：
                    # - 保存截圖
                    # - 觸發警報
                    # - 記錄時間戳
                    # - 發送通知
                    # 等等...

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