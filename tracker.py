
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self, max_age=30, n_init=3, max_cosine_distance=0.2):
        """
        Initializes the DeepSORT tracker, disabling the ReID embedder model 
        for immediate stability (relying on IOU tracking only).
        """
        self.tracker = DeepSort(
            max_age=max_age, 
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
           
            embedder_model_name=None 
        )
        print("✔ DeepSORT Tracker initialized (ReID disabled for stability).")
        
    def update(self, detections: np.ndarray, frame): 
        """
        Updates the tracker with new detections and the current frame.
        """
        
        if len(detections) == 0:
            
            self.tracker.update_tracks([], frame=frame)
            return []

       
        bboxes = detections[:, :4].astype(int) 
        confidences = detections[:, 4]
        class_ids = detections[:, 5].astype(int) 
        
        formatted_detections = [
            (bbox, confidence, str(cls_id)) 
            for bbox, confidence, cls_id in zip(bboxes, confidences, class_ids)
        ]

        
        tracks = self.tracker.update_tracks(formatted_detections, frame=frame) 

       
        tracked_results = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            latest_class_id = int(track.get_det_class()) 

            tracked_results.append([
                int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]), 
                int(track_id), latest_class_id
            ])
            
        return np.array(tracked_results)