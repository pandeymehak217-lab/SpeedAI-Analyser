
import argparse
import cv2
import time
import os
import uuid 
import numpy as np


from detector import Detector
from tracker import Tracker
from analyser import Analyser
from utils import draw_boxes, save_reports



def run_analysis(video_path: str, output_dir: str, file_id: str) -> dict:
    """Core function to run the full analysis pipeline on a video file."""
    print(f"Processing... Video: {video_path}")
    
    cap = None
    out = None
    try:
       
        detector = Detector(model_path='yolov8n.pt') 
        tracker = Tracker() 
        analyser = Analyser(detector.class_names)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file at {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_video_name = f"{file_id}_processed_video.mp4"
        output_video_path = os.path.join(output_dir, output_video_name)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
    except Exception as e:
        print(f"Error during analysis initialization: {e}")
        if cap and cap.isOpened(): cap.release()
        if out and out.isOpened(): out.release()
        raise e 

  
    frame_number = 0
    start_time = time.time()
    print("Starting frame processing...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_number += 1
        
       
        detections = detector.detect(frame)
        
        
        tracked_objects = tracker.update(detections, frame) 

        
        analyser.analyse_frame(tracked_objects, frame_number)
        
        
        processed_frame = draw_boxes(frame, tracked_objects, detector.class_names, frame_number)
        out.write(processed_frame)

        if frame_number % 100 == 0:
            print(f"  > Processed {frame_number} frames.")


    cap.release()
    out.release()
    
    end_time = time.time()
    
    print("\n" + "="*40)
    print("✔ Detection complete")
    print("✔ Tracking complete")
    print(f"Total time taken: {round(end_time - start_time, 2)} seconds")
    print("="*40)
    
  
    final_data = analyser.get_final_report_data()
    save_reports(final_data, fps, output_dir, file_id) 

    return final_data


def main_cli():
    """Handles Command Line Argument Parsing and launches analysis."""
    parser = argparse.ArgumentParser(description="Realtime Counting Analyser (Terminal Version)")
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file (.mp4, .mov, .avi).')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save results.')
    
    args = parser.parse_args()
    
    video_path = args.video
    
    cli_file_id = "report-" + str(uuid.uuid4())[:8] 
    cli_output_dir = args.output_dir
    os.makedirs(cli_output_dir, exist_ok=True) 

    try:
        run_analysis(video_path, cli_output_dir, cli_file_id)
        
        print(f"\nResults successfully saved in the '{cli_output_dir}' directory:")
        print(f" - Processed Video: {cli_output_dir}/{cli_file_id}_processed_video.mp4")
        print(f" - Data JSON: {cli_output_dir}/{cli_file_id}_results.json")
        print(f" - Summary CSV: {cli_output_dir}/{cli_file_id}_report.csv")
        print("\nProcessing complete!")

    except Exception as e:
        print(f"\nFatal Error during CLI run: {e}")


if __name__ == "__main__":
    main_cli()