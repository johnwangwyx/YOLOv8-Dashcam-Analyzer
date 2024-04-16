import cv2
from ObjectTracker import ObjectTracker
import time

class DashcamAnalyzer:
    def __init__(self, video_path, output_path, model_name='yolov8n.pt', show=True, produce_output=True):
        self.video_path = video_path
        self.output_path = output_path
        self.cap = cv2.VideoCapture(video_path)
        self.object_tracker = ObjectTracker(model_name=model_name)
        self.show = show
        self.produce_output = produce_output

        if produce_output:
            # Video Writer setup, I think H264 will be faster here but not avaliable in all systems
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    def analyze(self):

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Use ObjectTracker to process the frame and obtain tracking annotations
            processed_frame = self.object_tracker.process_frame(frame)
            if self.produce_output:
                self.out.write(processed_frame)  # Write the annotated frame out

            # Display the frame
            if self.show:
                cv2.imshow('Frame', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        if self.produce_output:
            self.out.release()
        cv2.destroyAllWindows()

# Processing the videos
video_paths = ["sample1.mp4", "sample2.mp4"]
output_paths = ["sample1_output.mp4", "sample2_output.mp4"]
for video_path, output_path in zip(video_paths, output_paths):
    analyzer = DashcamAnalyzer(video_path, output_path, calibration_factor=0.03)
    start = time.time()
    analyzer.analyze()
    print(f"processing {video_path} took {time.time() - start} seconds")
