import supervision as sv
from ultralytics import YOLO 
from tqdm import tqdm
import argparse
import numpy as np
import cv2
import time
from datetime import datetime
import json

class EventLogger:
    def __init__(self):
        self.events = {}
        self.frame_events = []
        self.event_counter = 0
        
    def add_event(self, track_id, event_type):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        event_id = str(self.event_counter)
        if event_id not in self.events:
            self.events[event_id] = []
            
        self.events[event_id].append({
            "ID": str(track_id),
            "event": event_type,
            "time": current_time
        })
        
        self.frame_events.append(f"ID {track_id} {event_type}")
        if len(self.frame_events) > 5:
            self.frame_events.pop(0)
            
        self.event_counter += 1
        
    def save_to_json(self, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.events, f, ensure_ascii=False, indent=2)
            
    def get_display_text(self):
        return '\n'.join(self.frame_events)

class LineCrossDetector:
    def __init__(self, line_x: int):
        self.line_x = line_x
        self.previous_positions = {}  # 記錄每個ID的上一次位置
        
    def update(self, detections):
        current_ids = detections.tracker_id
        in_count = 0
        out_count = 0
        triggered_events = []  # [(id, event_type), ...]
        
        # 計算每個檢測框的中心x座標
        boxes = detections.xyxy
        center_x = (boxes[:, 0] + boxes[:, 2]) / 2
        
        for idx, track_id in enumerate(current_ids):
            current_x = center_x[idx]
            
            if track_id in self.previous_positions:
                prev_x = self.previous_positions[track_id]
                
                # 檢查是否穿過檢測線
                if prev_x < self.line_x and current_x > self.line_x:
                    triggered_events.append((track_id, "out"))
                    in_count += 1
                elif prev_x >= self.line_x and current_x < self.line_x:
                    triggered_events.append((track_id, "in"))
                    out_count += 1
            
            # 更新位置記錄
            self.previous_positions[track_id] = current_x
            
        return triggered_events, in_count, out_count

def process_video(
        source_weights_path: str, 
        source_video_path: str,
        output_video: str, 
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.3 
) -> None:
    model = YOLO(source_weights_path)
    classes = list(model.names.values())
    
    # 設定檢測線位置（x座標）
    LINE_X1 = 1560
    LINE_X2= 1460
    midpointX=(LINE_X1+LINE_X2)/2
    LINE_START = sv.Point(LINE_X2, 500)
    LINE_END = sv.Point(LINE_X1, 0)
    
    tracker = sv.ByteTrack()    
    smoother = sv.DetectionsSmoother()
    box_annotator = sv.RoundBoxAnnotator(color=sv.Color.WHITE)
    label_annotator = sv.LabelAnnotator(color=sv.Color.YELLOW, text_color=sv.Color.BLACK, text_thickness=1, text_scale=0.6)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=0.6, text_color=sv.Color.WHITE, color=sv.Color.RED,display_in_count=False, display_out_count=False)
    line_counter = sv.LineZone(start=LINE_START, end = LINE_END)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    
    # 初始化事件記錄器和線檢測器
    event_logger = EventLogger()
    line_detector = LineCrossDetector(midpointX)
    
    # 建立虛擬的LineZone用於顯示
    # line_zone = sv.LineZone(start=LINE_START, end=LINE_END)
    total_in = 0
    total_out = 0

    fourcc = cv2.VideoWriter_fourcc(*'H264') 
    out = cv2.VideoWriter(output_video, fourcc, video_info.fps, (video_info.width, video_info.height))
    
    for frame in tqdm(frame_generator, total=video_info.total_frames):
        results = model(frame, verbose=False, conf=confidence_threshold, iou=iou_threshold)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[np.where((detections.class_id==0))]
        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        
        triggered_events, in_count, out_count = line_detector.update(detections)
        
        # 更新總計數並記錄事件
        
        for track_id, event_type in triggered_events:
            event_logger.add_event(track_id, event_type)
        
        # 更新虛擬LineZone的計數（用於顯示）
       
        
        # 繪製標註
        
        
        # 準備標籤
        labels = [
            f"id{tracker_id} {classes[class_id]} {confidence:.2f}"
            for tracker_id, class_id, confidence
            in zip(detections.tracker_id, detections.class_id, detections.confidence)
        ]
        line_counter.trigger(detections=detections)
        
        # 檢查穿越事件
        
        # 添加各種標註
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        annotated_frame = line_annotator.annotate(frame=annotated_frame,line_counter=line_counter)
        
        # 添加事件文字
        event_text = event_logger.get_display_text()
        y_position = 30
        for line in event_text.split('\n'):
            # color=(0,0,0) if line[-3]=="" else (255,255,255)
            cv2.putText(
                annotated_frame,
                line,
                (10, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,255),
                2
            )
            y_position += 30
            
        out.write(annotated_frame)

    # 保存事件記錄
    output_json = output_video.rsplit('.', 1)[0] + '_events.json'
    event_logger.save_to_json(output_json)
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("video processing with YOLO and ByteTrack")
    parser.add_argument(
        "--source_weights_path",
        required=True,
        help="Path to the source weights file",
        type=str
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str
    )
    parser.add_argument(
        "--output_video",
        required=True,
        help="Path to the target video file",
        type=str
    )
    args = parser.parse_args()
    process_video(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        output_video=args.output_video
    )