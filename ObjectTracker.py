from collections import defaultdict
import cv2
import json
from ultralytics import YOLO
import torch

ALLOWED_OBJECTS = ['person', 'car']

class ObjectTracker:
    def __init__(self, model_name='yolov8n.pt'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(0)
        print(f"Using device: {device}")
        self.model = YOLO(model_name)
        # According to this Github iessue, model must run once before changing to use GPU, so bizarre
        # https://github.com/ultralytics/ultralytics/issues/3084#issuecomment-1732433168
        self.model("IMG_6400.JPG")  # Do not care
        print(self.model.device.type)
        self.track_history = defaultdict(list)
        self.counts = {'parked car': 0, 'moving car': 0, 'person': 0}
        self.disappeared = defaultdict(int)
        self.max_disappeared = 5
        self.positions = defaultdict(list)
        self.moving_cars_certainty = defaultdict(int)
        
    def process_frame(self, frame):
        results = self.model.track(frame, persist=True, verbose=False)
        detections = json.loads(results[0].tojson())

        current_detected_ids = set()
        for det in detections:
            track_id = det['track_id']
            class_name = det['name']
            if class_name not in ALLOWED_OBJECTS:
                continue

            current_detected_ids.add(track_id)
            box = det['box']
            x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])

            bonus = calculate_center_moving_certainty_bonus(frame.shape[:2][1], x1, x2,)
            self.moving_cars_certainty[track_id] += bonus
            
            # Update the positions and reset or initialize disappeared count
            self.positions[track_id].append((y1 + y2) // 2)  # Store center y position
            self.disappeared[track_id] = 0  # Reset because the object is still in frame

            if class_name == 'car':
                if self.moving_cars_certainty[track_id] <= 0:
                    class_name = 'parked car'
                else:
                    class_name = 'moving car'

            # Determine movement and classification if enough data points have been collected
            if len(self.positions[track_id]) > 20:
                positions = self.positions[track_id][-20:]  # Consider the last 20 positions
                avg_newest = sum(positions[-7:]) / 7  # Average of the 7 newest points
                avg_oldest = sum(positions[:7]) / 7  # Average of the 7 oldest points
                movement = avg_newest - avg_oldest
                if movement < 2:
                    self.moving_cars_certainty[track_id] += 3
                else:
                    self.moving_cars_certainty[track_id] -= 1

            self.track_history[track_id].append(class_name)

            # Draw the box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"{class_name},ID {track_id}", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(frame, f"Moving certainty {self.moving_cars_certainty[track_id]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Handle disappeared objects, only mark as disappearred and increment counter after missing for more than X frames. (X = self.max_disappeared)
        for track_id in list(self.disappeared.keys()):
            if track_id not in current_detected_ids:
                self.disappeared[track_id] += 1
            if self.disappeared[track_id] > self.max_disappeared:
                if track_id in self.track_history:
                    last_known_class = self.track_history[track_id][-1]
                    self.counts[last_known_class] += 1
                    self.track_history.pop(track_id)
                self.disappeared.pop(track_id)

        # Display counts on the frame
        for idx, (class_name, count) in enumerate(self.counts.items()):
            cv2.putText(frame, f"{class_name.capitalize()}: {count}", (5, 20 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return frame
    
def calculate_center_moving_certainty_bonus(width, x1, x2):
        x_center = (x1 + x2) // 2
    
        if width*0.5 < x_center < width*0.55:
            return 9
        if width*0.45 < x_center < width*0.55:
            return 6
        if width*0.4 < x_center < width*0.55:
            return 3
        return 0