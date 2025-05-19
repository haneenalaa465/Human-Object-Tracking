# Person Route Tracking

A computer vision project that tracks people in videos and visualizes their routes with colored lines. Perfect for analyzing movement patterns, crowd flow analysis, and surveillance applications.

## Features
- Track multiple people simultaneously with different colored routes
- Generate a video file containing the original video content plus visualized routes
- Support for different tracking algorithms (ByteTrack, BoTSORT, StrongSORT)
- Export tracking data for further analysis

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- OpenCV
- A CUDA-capable GPU (recommended but optional)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/haneenalaa465/Human-Object-Tracking
   cd person-route-tracking
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
python main.py --input your_video.mp4 --output result.mp4 --show --save_data
```

Parameters:
- `--input`: Path to input video file (required)
- `--output`: Path to output video file (default: output.mp4)
- `--conf`: Confidence threshold for detection (default: 0.5)
- `--tracker`: Tracking algorithm to use (choices: bytetrack, botsort) (default: bytetrack)
- `--thickness`: Thickness of route lines (default: 3)
- `--show`: Show visualization while processing
- `--save_data`: Save tracking data to CSV


## Project Structure

```
person-route-tracking/
├── main.py    # Multi-person tracking implementation
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## How It Works

1. **Object Detection**: The system uses YOLO to detect people in each frame
2. **Object Tracking**: Detected people are tracked across frames using tracking algorithms
3. **Route Visualization**: The path of each tracked person is drawn with colored lines
4. **Video Output**: A new video is created with the original content plus visualized routes

## Requirements

See `requirements.txt` for a full list of dependencies. Main requirements:
- ultralytics
- opencv-python
- numpy
- torch
- torchvision

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- ByteTrack, BoTSORT and StrongSORT tracking algorithms
- OpenCV community

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
