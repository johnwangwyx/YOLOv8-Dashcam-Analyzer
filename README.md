## Dashcam Analyzer
Dashcam Analyzer leverages advanced computer vision technologies to dynamically track and analyze vehicular and pedestrian traffic through dashcam footage. Utilizing the powerful YOLOv8 model, this project provides insights into urban mobility patterns by distinguishing between parked and moving vehicles as well as monitoring pedestrian flow.

![image](https://github.com/johnwangwyx/YOLOv8-Dashcam-Analyzer/assets/78456315/9654047a-dcad-47f4-ad54-57b400a31471)

## Features
* Real-time Tracking: Identify and track cars and pedestrians and show the processing frames in real-time while writing the annotated video to the desired output.
* Traffic Analysis: Analyze traffic patterns to determine the popularity of different regions and the availability of parking spaces.
* Flexible Output Options: Customize the output to display real-time analytics or save the processed video for further analysis.

## How It Works
The system captures video data from dashcams, and processes the video frame-by-frame to detect and classify objects using the ObjectTracker module, which relies on the YOLOv8 object detection model. Each frame is annotated with tracking information, and the analysis can be viewed in real-time or output to a file for subsequent review.

## Technology Stack
* Python
* OpenCV for video processing
* PyTorch and YOLOv8 for object detection and tracking

## Usage
Set up the analyzer with the desired video input and output paths, model parameters, and view options. Start the analysis and watch as the system processes the video, providing insightful metrics and visualizations.
```
video_paths = ["sample1.mp4", "sample2.mp4"]
output_paths = ["sample1_output.mp4", "sample2_output.mp4"]
for video_path, output_path in zip(video_paths, output_paths):
    analyzer = DashcamAnalyzer(video_path, output_path)
    analyzer.analyze()
```

(The next step for this project is to enable cli options with `click` module to allow for cli usage to specify input/output and on/off for real-time annotation display, etc)

## Example output snippets

![image](https://github.com/johnwangwyx/YOLOv8-Dashcam-Analyzer/assets/78456315/9b49651d-802d-406f-9cce-82f32fc88c42)
![image](https://github.com/johnwangwyx/YOLOv8-Dashcam-Analyzer/assets/78456315/1fcc314c-07a0-4d10-ad43-0a0bd74bcd02)
![image](https://github.com/johnwangwyx/YOLOv8-Dashcam-Analyzer/assets/78456315/efbd53e1-c17b-489d-83ae-6e2641c446c0)
