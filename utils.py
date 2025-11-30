
import cv2
import json
import pandas as pd 
import os
import numpy as np 

COLOR_MAP = {
    'Person': (255, 0, 0),    
    'Car': (0, 255, 0),       
    'Truck': (0, 0, 255),     
    'Motorbike': (255, 255, 0),
    'Bicycle': (0, 255, 255),
    'Potato': (255, 0, 255)
}

def draw_boxes(frame, tracked_objects, class_names, frame_number):
    """Draws bounding boxes, IDs, class names, and frame number on the frame."""
    
    if isinstance(tracked_objects, np.ndarray):
        tracked_objects = tracked_objects.tolist()

    for obj in tracked_objects:
       
        x1, y1, x2, y2, track_id, class_id = map(int, obj)
        
        class_name = class_names.get(class_id, f"Class {class_id}")
        color = COLOR_MAP.get(class_name, (200, 200, 200)) 

       
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
      
        label = f"ID:{track_id} {class_name}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    cv2.putText(frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


def save_reports(analysis_data, video_fps, output_dir="output", file_id="results"):
    """Saves the final results to JSON and CSV using a unique file_id."""
    
    
    json_path = os.path.join(output_dir, f"{file_id}_results.json")
    analysis_data['metadata'] = {'video_fps': video_fps}
    with open(json_path, 'w') as f:
        json.dump(analysis_data, f, indent=4)
    
    
    summary_list = []
    for obj_id, data in analysis_data['tracked_objects'].items():
        entry_time = round(data['entry_frame'] / video_fps, 2)
        exit_time = round(data['exit_frame'] / video_fps, 2)
        duration = round(data['duration_frames'] / video_fps, 2)
        
        summary_list.append({
            'ID': obj_id,
            'Class': data['class'],
            'Entry Time (s)': entry_time,
            'Exit Time (s)': exit_time,
            'Duration (s)': duration,
            'Path Points': len(data['path'])
        })

    df = pd.DataFrame(summary_list)
    csv_path = os.path.join(output_dir, f"{file_id}_report.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"✔ Reports saved with file ID prefix: {file_id}")