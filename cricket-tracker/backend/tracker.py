"""
Video Tracker Module - YOLOv8 Segmentation + Simple Tracking

This module handles:
1. Person detection using YOLOv8 segmentation model
2. Simple centroid-based tracking
3. Extraction and storage of segmentation masks
4. Binary subdivision frame processing order for progressive preview

Author: Cricket Tracker Team
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Callable, Optional, Any
import traceback
import torch
import threading
from collections import OrderedDict
import bisect

# Fix for PyTorch 2.6+ weights_only default change
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Import YOLOv8 from ultralytics
from ultralytics import YOLO


def generate_binary_subdivision_order(n: int) -> List[int]:
    """
    Generate frame indices in binary subdivision order.
    
    Order: 0, n-1, n/2, n/4, 3n/4, n/8, 3n/8, 5n/8, 7n/8, ...
    
    This allows for progressive preview where the video can be
    previewed at increasingly finer granularity as processing continues.
    
    Args:
        n: Total number of frames
        
    Returns:
        List of frame indices in binary subdivision order
    """
    if n <= 0:
        return []
    if n == 1:
        return [0]
    if n == 2:
        return [0, 1]
    
    result = []
    added = set()
    
    # Add first and last frames
    result.append(0)
    added.add(0)
    
    result.append(n - 1)
    added.add(n - 1)
    
    # Binary subdivision
    queue = [(0, n - 1)]
    
    while queue:
        new_queue = []
        for left, right in queue:
            if right - left <= 1:
                continue
            mid = (left + right) // 2
            if mid not in added:
                result.append(mid)
                added.add(mid)
            new_queue.append((left, mid))
            new_queue.append((mid, right))
        queue = new_queue
    
    # Add any remaining frames that weren't included
    for i in range(n):
        if i not in added:
            result.append(i)
            added.add(i)
    
    return result


class ProcessedFramesBuffer:
    """
    Thread-safe buffer for storing processed frames data.
    Supports nearest-frame preview lookup.
    """
    
    def __init__(self, total_frames: int):
        self.total_frames = total_frames
        self.frames_data = {}  # frame_number -> frame_data
        self.processed_frames = []  # sorted list of processed frame numbers
        self.processing_order = []  # frames in binary subdivision processing order
        self.lock = threading.RLock()
        self.unique_ids = set()
        self.is_complete = False
        
    def add_frame(self, frame_number: int, frame_data: Dict, preview_frame: np.ndarray = None):
        """Add a processed frame to the buffer."""
        with self.lock:
            self.frames_data[frame_number] = {
                "data": frame_data,
                "preview_frame": preview_frame
            }
            bisect.insort(self.processed_frames, frame_number)
            self.processing_order.append(frame_number)  # Track in processing order
            
            # Track unique player IDs
            for player in frame_data.get("players", []):
                self.unique_ids.add(player.get("id"))
    
    def get_frame(self, frame_number: int) -> Optional[Dict]:
        """Get data for a specific frame if processed."""
        with self.lock:
            if frame_number in self.frames_data:
                return self.frames_data[frame_number]["data"]
            return None
    
    def get_nearest_frame(self, frame_number: int) -> tuple:
        """
        Get the nearest processed frame to the requested frame.
        
        Returns:
            (nearest_frame_number, frame_data, preview_frame)
        """
        with self.lock:
            if not self.processed_frames:
                return None, None, None
            
            if frame_number in self.frames_data:
                data = self.frames_data[frame_number]
                return frame_number, data["data"], data.get("preview_frame")
            
            # Find nearest using binary search
            pos = bisect.bisect_left(self.processed_frames, frame_number)
            
            if pos == 0:
                nearest = self.processed_frames[0]
            elif pos == len(self.processed_frames):
                nearest = self.processed_frames[-1]
            else:
                before = self.processed_frames[pos - 1]
                after = self.processed_frames[pos]
                nearest = before if (frame_number - before) <= (after - frame_number) else after
            
            data = self.frames_data[nearest]
            return nearest, data["data"], data.get("preview_frame")
    
    def get_processed_count(self) -> int:
        """Get the number of processed frames."""
        with self.lock:
            return len(self.processed_frames)
    
    def get_progress(self) -> float:
        """Get processing progress as percentage."""
        with self.lock:
            if self.total_frames == 0:
                return 100.0
            return (len(self.processed_frames) / self.total_frames) * 100
    
    def get_tracking_data(self) -> Dict[str, Any]:
        """Get all tracking data in the standard format."""
        with self.lock:
            frames = {}
            for frame_num, data in self.frames_data.items():
                frames[str(frame_num)] = data["data"]
            return {
                "frames": frames,
                "unique_ids": list(self.unique_ids)
            }
    
    def mark_complete(self):
        """Mark processing as complete."""
        with self.lock:
            self.is_complete = True
    
    def is_processing_complete(self) -> bool:
        """Check if processing is complete."""
        with self.lock:
            return self.is_complete
    
    def get_processing_order(self) -> List[int]:
        """Get frames in binary subdivision processing order."""
        with self.lock:
            return self.processing_order.copy()


class SimpleTracker:
    """
    Simple centroid-based tracker for maintaining object IDs across frames.
    """
    
    def __init__(self, max_disappeared: int = 30, max_distance: int = 100):
        self.next_id = 1
        self.objects = {}  # id -> (cx, cy, x1, y1, x2, y2)
        self.disappeared = {}  # id -> frames disappeared
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def update(self, detections: List[tuple]) -> Dict[int, tuple]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of (x1, y1, x2, y2) bounding boxes
            
        Returns:
            Dictionary mapping track_id to (cx, cy, x1, y1, x2, y2)
        """
        # If no detections, mark all as disappeared
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]
            return {}
        
        # Calculate centroids for new detections
        input_centroids = []
        for det in detections:
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            input_centroids.append((cx, cy, x1, y1, x2, y2))
        
        # If no existing objects, register all
        if len(self.objects) == 0:
            for centroid_data in input_centroids:
                self._register(centroid_data)
        else:
            # Match existing objects to new detections
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Calculate distances
            n_objects = len(object_centroids)
            n_inputs = len(input_centroids)
            distances = np.zeros((n_objects, n_inputs))
            
            for i in range(n_objects):
                ox, oy = object_centroids[i][0], object_centroids[i][1]
                for j in range(n_inputs):
                    cx, cy = input_centroids[j][0], input_centroids[j][1]
                    distances[i, j] = np.sqrt(float((ox - cx) ** 2 + (oy - cy) ** 2))
            
            # Simple greedy matching
            used_rows = set()
            used_cols = set()
            matches = []
            
            # Sort by distance
            flat_indices = np.argsort(distances.flatten())
            
            for idx in flat_indices:
                row = idx // n_inputs
                col = idx % n_inputs
                if row in used_rows or col in used_cols:
                    continue
                if distances[row, col] > self.max_distance:
                    continue
                used_rows.add(row)
                used_cols.add(col)
                matches.append((row, col))
            
            # Update matched objects
            for row, col in matches:
                obj_id = object_ids[row]
                self.objects[obj_id] = input_centroids[col]
                self.disappeared[obj_id] = 0
            
            # Handle unmatched existing objects
            for row in range(n_objects):
                if row not in used_rows:
                    obj_id = object_ids[row]
                    self.disappeared[obj_id] += 1
                    if self.disappeared[obj_id] > self.max_disappeared:
                        del self.objects[obj_id]
                        del self.disappeared[obj_id]
            
            # Register new detections
            for col in range(n_inputs):
                if col not in used_cols:
                    self._register(input_centroids[col])
        
        return self.objects.copy()
    
    def _register(self, centroid_data: tuple):
        """Register a new object."""
        self.objects[self.next_id] = centroid_data
        self.disappeared[self.next_id] = 0
        self.next_id += 1


class VideoTracker:
    """
    Video tracking class that combines YOLOv8 segmentation with simple tracking.
    Supports binary subdivision frame processing order for progressive preview.
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n-seg.pt",
        confidence_threshold: float = 0.5,
        max_age: int = 30
    ):
        """Initialize the video tracker."""
        self.confidence_threshold = confidence_threshold
        
        # Initialize YOLOv8 segmentation model
        print(f"Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)
        
        # Initialize simple tracker
        self.tracker = SimpleTracker(max_disappeared=max_age, max_distance=150)
        
        # Store unique player IDs seen across video
        self.unique_ids = set()
        
        # Progressive processing buffer
        self.frames_buffer = None
        self.video_properties = {}
        
    def process_video(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Process a video file with detection and tracking.
        Uses binary subdivision order for progressive preview capability.
        """
        try:
            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {input_path}")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            
            self.video_properties = {
                "video_width": width,
                "video_height": height,
                "fps": fps,
                "total_frames": total_frames
            }
            
            # Initialize frames buffer for progressive processing
            self.frames_buffer = ProcessedFramesBuffer(total_frames)
            
            # Generate binary subdivision frame order
            frame_order = generate_binary_subdivision_order(total_frames)
            
            if progress_callback:
                progress_callback(0, "Starting video analysis (binary subdivision)...")
            
            # Pre-load all frames into memory for random access
            all_frames = []
            print("Loading video frames into memory...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(frame)
            cap.release()
            
            print(f"Loaded {len(all_frames)} frames. Processing in binary subdivision order...")
            
            # Process frames in binary subdivision order
            for processed_count, frame_number in enumerate(frame_order):
                if frame_number < len(all_frames):
                    frame = all_frames[frame_number]
                    
                    # Process frame
                    frame_data, preview_frame = self._process_frame(frame, frame_number)
                    
                    # Add to buffer (thread-safe)
                    self.frames_buffer.add_frame(frame_number, frame_data, preview_frame)
                    
                    # Update progress
                    if progress_callback and (processed_count + 1) % 10 == 0:
                        progress = int(((processed_count + 1) / total_frames) * 100)
                        progress_callback(
                            progress,
                            f"Processing frame {processed_count + 1}/{total_frames} (frame #{frame_number})"
                        )
            
            # Mark processing as complete
            self.frames_buffer.mark_complete()
            
            # Build final tracking data
            tracking_data = {
                "video_width": width,
                "video_height": height,
                "fps": fps,
                "total_frames": total_frames,
                "frames": {},
                "unique_ids": list(self.unique_ids)
            }
            
            # Get all frame data from buffer
            buffer_data = self.frames_buffer.get_tracking_data()
            tracking_data["frames"] = buffer_data["frames"]
            tracking_data["unique_ids"] = buffer_data["unique_ids"]
            
            # Write output video in sequential order
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for i in range(len(all_frames)):
                out.write(all_frames[i])
            
            out.release()
            
            if progress_callback:
                progress_callback(100, "Video processing complete!")
            
            # Convert output to h264 for browser compatibility
            self._convert_to_h264(output_path)
            
            return tracking_data
            
        except Exception as e:
            print(f"Error in process_video: {str(e)}")
            traceback.print_exc()
            raise
    
    def get_preview_frame(self, frame_number: int) -> tuple:
        """
        Get preview for a specific frame during processing.
        Returns nearest processed frame if requested frame not yet processed.
        
        Returns:
            (actual_frame_number, frame_data, preview_image)
        """
        if self.frames_buffer is None:
            return None, None, None
        return self.frames_buffer.get_nearest_frame(frame_number)
    
    def get_processing_progress(self) -> Dict[str, Any]:
        """Get current processing progress and stats."""
        if self.frames_buffer is None:
            return {
                "progress": 0,
                "processed_frames": 0,
                "total_frames": 0,
                "is_complete": False,
                "processing_order": []
            }
        return {
            "progress": self.frames_buffer.get_progress(),
            "processed_frames": self.frames_buffer.get_processed_count(),
            "total_frames": self.frames_buffer.total_frames,
            "is_complete": self.frames_buffer.is_processing_complete(),
            "processing_order": self.frames_buffer.get_processing_order()
        }
    
    def get_partial_tracking_data(self) -> Optional[Dict[str, Any]]:
        """Get partial tracking data during processing for preview."""
        if self.frames_buffer is None:
            return None
        
        data = self.frames_buffer.get_tracking_data()
        return {
            **self.video_properties,
            "frames": data["frames"],
            "unique_ids": data["unique_ids"],
            "is_partial": not self.frames_buffer.is_processing_complete()
        }
    
    def _process_frame(self, frame: np.ndarray, frame_number: int) -> tuple:
        """Process a single frame with detection and tracking."""
        frame_data = {
            "frame_number": frame_number,
            "players": []
        }
        
        preview_frame = frame.copy()
        
        try:
            # Run YOLOv8 segmentation
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                classes=[0],  # Class 0 is 'person' in COCO
                verbose=False
            )
            
            # Extract detections
            detections = []
            masks_data = []
            
            if results and len(results) > 0:
                result = results[0]
                
                # Check if we have boxes
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    
                    # Get masks if available
                    masks = None
                    if result.masks is not None:
                        masks = result.masks.data.cpu().numpy()
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        detections.append((x1, y1, x2, y2))
                        
                        # Store mask data
                        if masks is not None and i < len(masks):
                            mask = masks[i]
                            mask_resized = cv2.resize(
                                mask.astype(np.uint8),
                                (frame.shape[1], frame.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            )
                            
                            contours, _ = cv2.findContours(
                                mask_resized,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE
                            )
                            
                            if contours:
                                largest_contour = max(contours, key=cv2.contourArea)
                                epsilon = 0.01 * cv2.arcLength(largest_contour, True)
                                simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
                                masks_data.append(simplified.flatten().tolist())
                            else:
                                masks_data.append([])
                        else:
                            masks_data.append([])
            
            # Update tracker
            tracked_objects = self.tracker.update(detections)
            
            # Process tracked objects
            for track_id, centroid_data in tracked_objects.items():
                cx, cy, x1, y1, x2, y2 = centroid_data
                self.unique_ids.add(track_id)
                
                # Find corresponding mask
                mask_polygon = []
                for i, det in enumerate(detections):
                    det_cx = (det[0] + det[2]) // 2
                    det_cy = (det[1] + det[3]) // 2
                    if abs(det_cx - cx) < 50 and abs(det_cy - cy) < 50:
                        if i < len(masks_data):
                            mask_polygon = masks_data[i]
                        break
                
                # Store player data
                player_data = {
                    "id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "mask_polygon": mask_polygon,
                    "confidence": 0.9
                }
                frame_data["players"].append(player_data)
                
                # Tracking data is stored but no boxes are drawn on the preview frame
                # The original video is preserved without visual overlays
                
        except Exception as e:
            print(f"Error processing frame {frame_number}: {str(e)}")
            traceback.print_exc()
        
        return frame_data, preview_frame
    
    def _get_track_color(self, track_id: int) -> tuple:
        """Generate a unique color for each track ID."""
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 255, 0),  # Lime
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (0, 128, 255),  # Light Blue
        ]
        return colors[track_id % len(colors)]
    
    def _convert_to_h264(self, video_path: str):
        """Convert video to H.264 codec for browser compatibility."""
        import subprocess
        import shutil
        
        if shutil.which('ffmpeg') is None:
            print("Warning: ffmpeg not found. Video may not play in all browsers.")
            return
        
        temp_path = video_path.replace('.mp4', '_temp.mp4')
        
        try:
            shutil.move(video_path, temp_path)
            
            cmd = [
                'ffmpeg',
                '-i', temp_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-y',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                Path(temp_path).unlink()
            else:
                shutil.move(temp_path, video_path)
                print(f"FFmpeg conversion failed: {result.stderr}")
                
        except Exception as e:
            print(f"Error during video conversion: {e}")
            if Path(temp_path).exists():
                shutil.move(temp_path, video_path)
