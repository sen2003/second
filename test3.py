import supervision as sv
from ultralytics import YOLO 
from tqdm import tqdm
import numpy as np
import cv2
import time
from datetime import datetime
import json
import boto3
from PIL import Image, ImageDraw, ImageFont
import os
import torch

def chinese_name(ExternalImageId):
    name_map = {
        "Huang": "黃士熏",
        "Ko": "柯信汶",
        "Shen": "沈宏勳",
        "Tsou": "鄒博森"
    }
    return name_map.get(ExternalImageId)

def displayChineseText(frame, text, position, textColor, textSize):
    if (isinstance(frame, np.ndarray)):
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame)
    fontStyle = ImageFont.truetype("msjhbd.ttc", textSize, encoding="utf-8")
    draw.text(position, text, textColor, font=fontStyle)
    return cv2.cvtColor(np.asanyarray(frame), cv2.COLOR_RGB2BGR)

class EventLogger:
    def __init__(self):
        self.events = {}
        self.frame_events = []
        self.event_counter = 0
        self.known_faces = {}  # 儲存已識別的ID與對應的姓名和信心度
        
    def add_event(self, track_id, event_type, name=None, confidence=None):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        event_id = str(self.event_counter)
        if event_id not in self.events:
            self.events[event_id] = []
            
        self.events[event_id].append({
            "Name": name if name else "Unknown",
            "Confidence": confidence if confidence else 0.0,
            "Event": event_type,
            "Time": current_time
        })
        
        display_name = name if name else f"ID {track_id}"
        self.frame_events.append((f"{current_time} : {display_name} {event_type}", event_type))
        if len(self.frame_events) > 5:
            self.frame_events.pop(0)
            
        self.event_counter += 1
        
    def update_face_info(self, track_id, name, confidence):
        self.known_faces[track_id] = {"name": name, "confidence": confidence}
        
    def get_face_info(self, track_id):
        return self.known_faces.get(track_id, None)
        
    def save_to_json(self, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.events, f, ensure_ascii=False, indent=2)
            
    def get_display_text(self):
        return self.frame_events

class LineCrossDetector:
    def __init__(self, line_x: int):
        self.line_x = line_x
        self.previous_positions = {}
        self.current_state = None
        
    def update(self, detections, event_logger):
        current_ids = detections.tracker_id
        in_count = 0
        out_count = 0
        triggered_events = []
        boxes = detections.xyxy
        left_x = boxes[:, 0]
        right_x = boxes[:, 2]
        self.current_state = None

        for idx, track_id in enumerate(current_ids):
            current_left = left_x[idx]
            current_right = right_x[idx]
            
            if track_id in self.previous_positions:
                prev_left, prev_right = self.previous_positions[track_id]
                if self.line_x<=current_right and self.line_x>=current_left:
                    continue
                
                face_info = event_logger.get_face_info(track_id)
                name = face_info["name"] if face_info else None
                confidence = face_info["confidence"] if face_info else None
                
                if prev_right > self.line_x and current_right <= self.line_x:
                    triggered_events.append((track_id, "in"))
                    in_count += 1
                    self.current_state = "in"
                elif prev_left <= self.line_x and current_left > self.line_x:
                    triggered_events.append((track_id, "out"))
                    out_count += 1
                    self.current_state = "out"
            
            self.previous_positions[track_id] = (current_left, current_right)
        return triggered_events, in_count, out_count

    def get_line_color(self):
        if self.current_state == "in":
            return sv.Color.RED
        elif self.current_state == "out":
            return sv.Color.GREEN
        return sv.Color.WHITE

def process_video(
        source_weights_path: str,
        face_weights_path: str,
        source_video_path: str,
        output_video: str,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.3
) -> None:
    # Initialize AWS Rekognition
    rekognition = boto3.client('rekognition', region_name='us-east-1')
    collection_id = "myCollection1"
    
    # Initialize models
    model = YOLO(source_weights_path)
    face_model = YOLO(face_weights_path)
    classes = list(model.names.values())
    
    # Set up line detection
    LINE_X1 = 1500
    LINE_X2 = 1500
    midpointX = (LINE_X1 + LINE_X2)/2
    LINE_START = sv.Point(LINE_X2, 500)
    LINE_END = sv.Point(LINE_X1, 0)
    
    # Initialize trackers and annotators
    tracker = sv.ByteTrack()
    smoother = sv.DetectionsSmoother()
    box_annotator = sv.RoundBoxAnnotator(color=sv.Color.WHITE)
    label_annotator = sv.LabelAnnotator(color=sv.Color.YELLOW, text_color=sv.Color.BLACK, text_thickness=1, text_scale=0.6)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=0.6, text_color=sv.Color.WHITE, color=sv.Color.WHITE,display_in_count=False, display_out_count=False)
    line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
    
    # Initialize video processing
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    
    # Initialize loggers and detectors
    event_logger = EventLogger()
    line_detector = LineCrossDetector(midpointX)
    
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_video, fourcc, video_info.fps, (video_info.width, video_info.height))
    
    for frame in tqdm(frame_generator, total=video_info.total_frames):
        # Detect faces
        face_results = model(frame, verbose=False, conf=confidence_threshold, iou=iou_threshold)[0]
        detections = sv.Detections.from_ultralytics(face_results)
        detections = detections[np.where((detections.class_id==0))]
        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)
        
        annotated_frame = frame.copy()
        
        # Process each detection
        for idx, track_id in enumerate(detections.tracker_id):
            if event_logger.get_face_info(track_id) is None:
                # Extract face image
                box = detections.xyxy[idx]
                x1, y1, x2, y2 = map(int, box)
                face_img = frame[y1:y2, x1:x2]
                
                try:
                    # AWS Rekognition
                    _, encoded_image = cv2.imencode('.jpg', face_img)
                    response = rekognition.search_faces_by_image(
                        CollectionId=collection_id,
                        Image={'Bytes': encoded_image.tobytes()},
                        MaxFaces=1,
                        FaceMatchThreshold=15
                    )
                    
                    if response['FaceMatches']:
                        match = response['FaceMatches'][0]
                        name = chinese_name(match['Face']['ExternalImageId'])
                        confidence = match['Similarity']
                        event_logger.update_face_info(track_id, name, confidence)
                except Exception as e:
                    print(f"Error processing face {track_id}: {str(e)}")
        
        # Update line crossing events
        triggered_events, _, _ = line_detector.update(detections, event_logger)
        
        # Record events
        for track_id, event_type in triggered_events:
            face_info = event_logger.get_face_info(track_id)
            if face_info:
                event_logger.add_event(track_id, event_type, face_info["name"], face_info["confidence"])
            else:
                event_logger.add_event(track_id, event_type)
        
        # Draw annotations
        labels = []
        for idx, track_id in enumerate(detections.tracker_id):
            face_info = event_logger.get_face_info(track_id)
            if face_info:
                label = f"{chinese_name(face_info['name'])} ({face_info['confidence']:.1f}%)"
            else:
                label = f"ID{track_id}"
            labels.append(label)
        
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        annotated_frame = line_annotator.annotate(frame=annotated_frame, line_counter=line_counter)
        line_annotator.color = line_detector.get_line_color()
        
        # Add event text
        event_text = event_logger.get_display_text()
        y_position = 32
        for text, event_type in event_text:
            color = (0, 0, 255) if event_type == "in" else (0, 128, 0)
            annotated_frame = displayChineseText(
                annotated_frame,
                text,
                (20, y_position),
                color,
                24
            )
            y_position += 32
        
        out.write(annotated_frame)
    
    # Save event log
    output_json = output_video.rsplit('.', 1)[0] + '_events.json'
    event_logger.save_to_json(output_json)
    out.release()

if __name__ == "__main__":
    # 直接設定參數
    source_weights_path = "C:\cvyolo\yoloV8_test\yolo11m.pt"  # 一般物件偵測模型
    face_weights_path = "C:\cvyolo\yoloV8_test\yolov8n-face.pt"  # 人臉偵測模型
    source_video_path = "C:\\cvyolo\\c1029v3.mp4"
    output_video = "output2.mp4"
    
    process_video(
        source_weights_path=source_weights_path,
        face_weights_path=face_weights_path,
        source_video_path=source_video_path,
        output_video=output_video
    )