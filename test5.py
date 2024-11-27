# 追蹤人，檢測線判斷進出門 0.5 0.3
import supervision as sv
from ultralytics import YOLO 
from tqdm import tqdm
import argparse
import numpy as np
import cv2
import time
from datetime import datetime
import json

tracker = sv.ByteTrack() 
def process_video(
        source_weights_path: str, 
        source_video_path: str,
        output_video: str, 
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.3
) -> None:
    model = YOLO(source_weights_path)       # Load YOLO model 
    classes = list(model.names.values())    # Class names 
    LINE_STARTS = sv.Point(1500,700)           # Line start point for count in/out vehicle
    LINE_END = sv.Point(1500, 0)          # Line end point for count in/out vehicle
    tracker = sv.ByteTrack()                # Bytetracker instance 
    box_annotator = sv.BoundingBoxAnnotator()     # BondingBox annotator instance 
    label_annotator = sv.LabelAnnotator()         # Label annotator instance 
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path) # for generating frames from video
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    line_counter = sv.LineZone(start=LINE_STARTS, end = LINE_END)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale= 0.5)


    fourcc = cv2.VideoWriter_fourcc(*'H264') 
    fps = video_info.fps
    width, height = video_info.width, video_info.height
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for frame in tqdm(frame_generator, total= video_info.total_frames):
        # Getting result from model
        results = model(frame, verbose=False, conf= confidence_threshold, iou = iou_threshold)[0] 
        detections = sv.Detections.from_ultralytics(results)    # Getting detections
        #Filtering classes for car and truck only instead of all COCO classes.
        detections = detections[np.where((detections.class_id==0))]
        detections = tracker.update_with_detections(detections)  # Updating detection to Bytetracker
        # Annotating detection boxes
        annotated_frame = box_annotator.annotate(scene = frame.copy(), detections= detections) 
        #Prepare labels
        labels = []
        for index in range(len(detections.class_id)):
            # creating labels as per required.
            labels.append( "id"+ str(detections.tracker_id[index]) + " " + classes[detections.class_id[index]] + " "+ str(round(detections.confidence[index],2)) )
        
        # Line counter in/out trigger
        line_counter.trigger(detections=detections)
        # Annotating labels
        annotated_label_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        # Annotating line labels
        line_annotate_frame = line_annotator.annotate(frame=annotated_label_frame, line_counter=line_counter)
        out.write(line_annotate_frame)
        # sink.write_frame(frame = line_annotate_frame)

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
        type = str
    )
    parser.add_argument(
        "--output_video",
        required=True,
        help="Path to the target video file",
        type= str
    )
    # parser.add_argument(
    #     "--confidence_threshold",
    #     default = 0.5,
    #     help= "Confidence threshold for the model",
    #     type=float
    # )
    # parser.add_argument(
    #     "--iou_threshold",
    #     default=0.3,
    #     help="Iou threshold for the model",
    #     type= float
    # )
    args = parser.parse_args() 
    process_video(
        source_weights_path=args.source_weights_path, 
        source_video_path= args.source_video_path,
        output_video=args.output_video 
        # confidence_threshold=args.confidence_threshold,
        # iou_threshold=args.iou_threshold
    )