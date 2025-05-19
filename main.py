import cv2  
import numpy as np  
import argparse  
from ultralytics import YOLO  
import torch  
from collections import defaultdict  
import os  
from pathlib import Path  
from scipy.spatial.distance import cosine  

class MultiPersonTracker:
    def __init__(self, video_path, output_path, conf_threshold=0.5, 
                 track_algorithm='botsort.yaml', line_thickness=3, show_video=False):
        """
        Initialize the Multi-Person Tracker with configuration parameters
        
        Args:
            video_path (str): Path to input video file
            output_path (str): Path to save output video
            conf_threshold (float): Confidence threshold for detection (0-1)
            track_algorithm (str): Tracking algorithm to use ('botsort.yaml' or 'bytetrack.yaml')
            line_thickness (int): Thickness of tracking path lines
            show_video (bool): Whether to display video during processing
        """
        # Store configuration parameters
        self.video_path = video_path
        self.output_path = output_path
        self.conf_threshold = conf_threshold
        self.track_algorithm = track_algorithm  # Tracking algorithm selection
        self.line_thickness = line_thickness
        self.show_video = show_video
        
        # Load YOLOv8 model (nano version for person detection)
        self.model = YOLO("yolov8n.pt")
        
        # Data structures for tracking
        self.tracks = defaultdict(list)  # Stores position history for each track ID
        self.appearance_features = defaultdict(list)  # Stores appearance features for re-identification
        self.last_seen = defaultdict(int)  # Records last frame each track was seen
        
        # Generate distinct colors for visualization (one per track)
        self.colors = self._generate_colors(30)
        
        self.cap = None  
        self.width = None  
        self.height = None
        self.fps = None  
        self.frame_count = 0  
        self.iou_threshold = 0.7  

    def _generate_colors(self, num_colors):
        """
        Generate distinct colors for visualizing different tracks
        
        Args:
            num_colors (int): Number of distinct colors needed
            
        Returns:
            list: List of RGB color tuples
        """
        # Base set of distinguishable colors
        base_colors = [
            (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),  # Red, Green, Blue, Yellow
            (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),  # Magenta, Cyan, etc.
            (255, 128, 0), (128, 255, 0)
        ]
        colors = base_colors.copy()
        
        # Generate additional colors in HSV space if needed
        if num_colors > len(base_colors):
            for i in range(len(base_colors), num_colors):
                hue = i / (num_colors - len(base_colors))  # Vary hue
                sat = 0.8  # Fixed saturation
                val = 0.9  # Fixed value
                # Convert HSV to RGB
                rgb = cv2.cvtColor(np.array([[[hue, sat, val]]], dtype=np.float32), cv2.COLOR_HSV2BGR)[0][0]
                color = tuple(int(c * 255) for c in rgb)
                colors.append(color)
        return colors

    def _extract_appearance_feature(self, image, box):
        """
        Extract appearance features for re-identification using color histogram and HOG
        
        Args:
            image (numpy.ndarray): Input frame
            box (list): Bounding box coordinates [x1,y1,x2,y2]
            
        Returns:
            numpy.ndarray: Combined feature vector or None if invalid box
        """
        x1, y1, x2, y2 = map(int, box)
        # Crop the person region from image
        person_img = image[y1:y2, x1:x2]
        if person_img.size == 0:  # Skip empty boxes
            return None
            
        # Resize to standard size for consistent feature extraction
        person_img = cv2.resize(person_img, (64, 128))
        
        # Extract color histogram features (for each RGB channel)
        hist_features = []
        for i in range(3):  # For each color channel (B,G,R)
            hist = cv2.calcHist([person_img], [i], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()  # Normalize histogram
            hist_features.extend(hist)
            
        # Extract HOG (Histogram of Oriented Gradients) features
        hog = cv2.HOGDescriptor()  # Default parameters
        hog_features = hog.compute(person_img)
        hog_features = hog_features.flatten()
        
        # Combine both feature types into single vector
        features = np.concatenate([hist_features, hog_features])
        return features

    def _compute_iou(self, box1, box2):
        """
        Compute Intersection over Union (IoU) between two bounding boxes
        
        Args:
            box1 (list): First bounding box [x1,y1,x2,y2]
            box2 (list): Second bounding box [x1,y1,x2,y2]
            
        Returns:
            float: IoU score (0-1)
        """
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area coordinates
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        # Compute intersection and union areas
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    def initialize_video(self):
        """Initialize video capture and writer objects"""
        # Open input video
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Create output directory if needed
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize video writer (MP4 format)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.width, self.height)
        )
        return True

    def process_frame(self, frame):
        """
        Process a single video frame with detection, tracking, and re-identification
        
        Args:
            frame (numpy.ndarray): Input video frame
            
        Returns:
            numpy.ndarray: Annotated output frame
        """
        # Run YOLOv8 tracking on the frame
        results = self.model.track(
            frame, 
            persist=True,  # Maintain track between frames
            classes=0,  # Only track person class (ID 0)
            conf=self.conf_threshold,  # Confidence threshold
            tracker=self.track_algorithm  # Tracking algorithm
        )
        
        # Prepare output frame and tracking data
        output_frame = frame.copy()
        current_boxes = []  # Current frame bounding boxes
        current_ids = []  # Current frame track IDs
        current_features = []  # Current frame appearance features
        
        # Process detections if any exist
        if results[0].boxes is not None and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
            # Get bounding boxes and track IDs
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            
            # Process each detected person
            for box, track_id in zip(boxes, track_ids):
                # Extract appearance features
                feature = self._extract_appearance_feature(frame, box)
                if feature is None:
                    continue  # Skip invalid detections
                    
                # Store current detection info
                current_boxes.append(box)
                current_ids.append(track_id)
                current_features.append(feature)
                
                # Check for occlusions with other detections
                is_occluded = False
                for i, other_box in enumerate(current_boxes[:-1]):
                    # Calculate IoU with other boxes
                    iou = self._compute_iou(box, other_box)
                    if iou > self.iou_threshold:  # Significant overlap
                        is_occluded = True
                        # Compare with previous appearance of this track
                        prev_feature = self.appearance_features[track_id][-1] if self.appearance_features[track_id] else None
                        if prev_feature is not None:
                            # Calculate cosine similarity (1 = identical, 0 = completely different)
                            similarity = 1 - cosine(feature, prev_feature)
                            if similarity < 0.8:  # Significant appearance change
                                # Search for best matching existing track
                                best_id = track_id
                                best_sim = similarity
                                for old_id, old_features in self.appearance_features.items():
                                    # Only consider recently seen tracks (last 30 frames)
                                    if old_features and self.frame_count - self.last_seen[old_id] < 30:
                                        sim = 1 - cosine(feature, old_features[-1])
                                        if sim > best_sim:  # Found better match
                                            best_sim = sim
                                            best_id = old_id
                                if best_id != track_id:  # Reassign ID if better match found
                                    current_ids[-1] = best_id
                                    print(f"Reassigned ID {track_id} to {best_id} due to occlusion")
                
                # If no occlusion or successfully handled, update track
                if not is_occluded:
                    # Calculate foot position (center bottom of box)
                    x1, y1, x2, y2 = map(int, box)
                    center_x, center_y = int((x1 + x2) / 2), int(y2)
                    # Update track history
                    self.tracks[current_ids[-1]].append((center_x, center_y))
                    # Update appearance features
                    self.appearance_features[current_ids[-1]].append(feature)
                    # Update last seen frame
                    self.last_seen[current_ids[-1]] = self.frame_count
                
                # Visualization: Draw bounding box and ID
                color = self.colors[current_ids[-1] % len(self.colors)]
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    output_frame, f"ID: {current_ids[-1]}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
                
                # Draw movement path if history exists
                if len(self.tracks[current_ids[-1]]) > 1:
                    for j in range(1, len(self.tracks[current_ids[-1]])):
                        pt1 = self.tracks[current_ids[-1]][j-1]
                        pt2 = self.tracks[current_ids[-1]][j]
                        cv2.line(output_frame, pt1, pt2, color, self.line_thickness)
        
        # Add frame counter to output
        cv2.putText(
            output_frame, f"Frame: {self.frame_count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        
        return output_frame

    def run(self):
        """Main processing loop to track people through entire video"""
        # Initialize video capture and writer
        if not self.initialize_video():
            return False
        
        # Get total frames for progress reporting
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing video with {total_frames} frames...")
        
        try:
            # Process each frame
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:  # End of video
                    break
                
                # Print progress every 50 frames
                if self.frame_count % 50 == 0:
                    print(f"Processing frame {self.frame_count}/{total_frames} ({self.frame_count/total_frames*100:.1f}%)")
                
                # Process frame and write output
                output_frame = self.process_frame(frame)
                self.writer.write(output_frame)
                
                # Display video if enabled
                if self.show_video:
                    cv2.imshow('Multi-Person Tracking', output_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q' key
                        break
                
                self.frame_count += 1  # Increment frame counter
                
            print(f"Processed {self.frame_count} frames. Output saved to: {self.output_path}")
            return True
            
        finally:
            # Clean up resources
            if self.cap is not None:
                self.cap.release()
            if hasattr(self, 'writer'):
                self.writer.release()
            cv2.destroyAllWindows()

    def save_track_data(self, output_csv=None):
        """
        Save tracking data to CSV file
        
        Args:
            output_csv (str, optional): Path to output CSV. Defaults to video path with .csv extension.
        """
        if output_csv is None:
            output_csv = Path(self.output_path).with_suffix('.csv')
        
        # Write track ID and position history
        with open(output_csv, 'w') as f:
            f.write("track_id,x,y\n")  # CSV header
            for track_id, points in self.tracks.items():
                for x, y in points:
                    f.write(f"{track_id},{x},{y}\n")  # One point per line
        
        print(f"Track data saved to: {output_csv}")

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Multi-Person Route Tracker")
    # Required/optional arguments
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to output video")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (0-1)")
    parser.add_argument("--tracker", type=str, default="botsort.yaml", 
                        choices=["bytetrack.yaml", "botsort.yaml"], help="Tracking algorithm")
    parser.add_argument("--thickness", type=int, default=3, help="Path line thickness")
    parser.add_argument("--show", action="store_true", help="Show video during processing")
    parser.add_argument("--save_data", action="store_true", help="Save tracking data to CSV")
    return parser.parse_args()

def main():
    """Main function to execute tracking"""
    # Parse command line arguments
    args = parse_args()
    
    # Initialize tracker with parameters
    tracker = MultiPersonTracker(
        video_path=args.input,
        output_path=args.output,
        conf_threshold=args.conf,
        track_algorithm=args.tracker,
        line_thickness=args.thickness,
        show_video=args.show
    )
    
    # Run tracking and save data if requested
    success = tracker.run()
    if success and args.save_data:
        tracker.save_track_data()
    print("Processing complete!")

if __name__ == "__main__":
    main()  # Entry point when run as script