from ultralytics import YOLO
import numpy as np

class Detector:
    def __init__(self, model_path='yolov8n.pt'):
        """Initializes the YOLO model."""
        self.model = YOLO(model_path)
        self.class_names = self.model.names 
        print(f"Detector initialized with {len(self.class_names)} classes.")

    def detect(self, frame):
        """
        Runs detection on a frame.
        Returns: numpy array of detections in format: [x1, y1, x2, y2, conf, cls]
        """
        
        results = self.model(frame, verbose=False)[0] 
        
        

        if results.boxes is not None and results.boxes.data is not None:
            boxes = results.boxes.data.cpu().numpy()
            if boxes.size > 0:
                 return boxes
        
        
        return np.empty((0, 6), dtype=np.float32)
