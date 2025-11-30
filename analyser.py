
class ObjectData:
    """A simple class to hold state for a single tracked object."""
    def __init__(self, track_id, class_name, frame_number):
        self.id = track_id
        self.class_name = class_name
        self.entry_frame = frame_number
        self.exit_frame = frame_number
        self.duration_frames = 0
        self.path = [] 

class Analyser:
    def __init__(self, detector_class_names):
        """Initializes the analysis state."""
        self.tracked_objects_data = {} 
        self.class_names = detector_class_names
        self.total_counts = {name: 0 for name in self.class_names.values()}
        print("Analyser initialized.")

    def analyse_frame(self, tracked_objects, frame_number):
        """Processes the tracked objects for the current frame."""
        current_frame_ids = set()
        
        if hasattr(tracked_objects, 'tolist'):
            tracked_objects = tracked_objects.tolist()

        for obj in tracked_objects:
            if len(obj) < 6: continue 
                
            x1, y1, x2, y2, track_id, class_id = map(int, obj)
            
            class_name = self.class_names.get(class_id, "Unknown")
            
            current_frame_ids.add(track_id)
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2

            if track_id not in self.tracked_objects_data:
               
                self.tracked_objects_data[track_id] = ObjectData(track_id, class_name, frame_number)
                self.total_counts[class_name] = self.total_counts.get(class_name, 0) + 1 
            
           
            obj_data = self.tracked_objects_data[track_id]
            obj_data.exit_frame = frame_number
            obj_data.path.append((x_center, y_center))
            obj_data.duration_frames += 1
            
    def get_final_report_data(self):
        """Formats the final data structure for saving."""
        report = {
            "total_objects_per_class": self.total_counts,
            "tracked_objects": {}
        }
        
        for id, data in self.tracked_objects_data.items():
            report["tracked_objects"][id] = {
                "class": data.class_name,
                "entry_frame": data.entry_frame,
                "exit_frame": data.exit_frame,
                "duration_frames": data.duration_frames,
                "path": data.path
            }
        
        return report