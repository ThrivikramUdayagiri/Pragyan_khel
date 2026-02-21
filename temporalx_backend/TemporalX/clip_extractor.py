"""
Error Clip Extractor
====================
Extracts video segments containing temporal errors for detailed review.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ErrorClipExtractor:
    """Extract video clips containing temporal errors"""
    
    def __init__(self, padding_seconds: float = 2.0):
        """
        Initialize clip extractor.
        
        Args:
            padding_seconds: Seconds to include before/after error
        """
        self.padding_seconds = padding_seconds
    
    def extract_error_clips(self,
                           video_path: str,
                           results: List[Dict],
                           output_dir: str,
                           fps: float,
                           merge_nearby: bool = True,
                           nearby_threshold: float = 3.0) -> List[str]:
        """
        Extract video clips containing errors.
        
        Args:
            video_path: Input video path
            results: Detection results from TemporalErrorDetector
            output_dir: Directory to save clips
            fps: Video frame rate
            merge_nearby: Merge errors within nearby_threshold seconds
            nearby_threshold: Seconds threshold for merging clips
            
        Returns:
            List of extracted clip paths
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Find error frames
        error_frames = []
        for result in results:
            if result['classification'] in ['Frame Drop', 'Frame Merge']:
                error_frames.append({
                    'frame_num': result['frame_num'],
                    'timestamp': result['timestamp'],
                    'classification': result['classification'],
                    'confidence': result['confidence']
                })
        
        if not error_frames:
            logger.info("No errors found to extract")
            return []
        
        # Calculate time ranges for clips
        clip_ranges = self._calculate_clip_ranges(
            error_frames, fps, merge_nearby, nearby_threshold
        )
        
        logger.info(f"Extracting {len(clip_ranges)} clips from {len(error_frames)} errors")
        
        # Extract clips
        extracted_clips = []
        for i, (start_time, end_time, error_types) in enumerate(clip_ranges):
            clip_path = self._extract_clip(
                video_path, start_time, end_time, output_dir, i, error_types
            )
            if clip_path:
                extracted_clips.append(clip_path)
        
        return extracted_clips
    
    def _calculate_clip_ranges(self,
                               error_frames: List[Dict],
                               fps: float,
                               merge_nearby: bool,
                               nearby_threshold: float) -> List[Tuple]:
        """Calculate time ranges for clips with optional merging"""
        if not error_frames:
            return []
        
        # Sort by timestamp
        error_frames.sort(key=lambda x: x['timestamp'])
        
        ranges = []
        current_start = error_frames[0]['timestamp'] - self.padding_seconds
        current_end = error_frames[0]['timestamp'] + self.padding_seconds
        current_errors = [error_frames[0]['classification']]
        
        for error in error_frames[1:]:
            error_time = error['timestamp']
            
            if merge_nearby and (error_time - current_end) < nearby_threshold:
                # Extend current clip
                current_end = error_time + self.padding_seconds
                current_errors.append(error['classification'])
            else:
                # Save current clip and start new one
                ranges.append((
                    max(0, current_start),
                    current_end,
                    current_errors.copy()
                ))
                current_start = error_time - self.padding_seconds
                current_end = error_time + self.padding_seconds
                current_errors = [error['classification']]
        
        # Add last clip
        ranges.append((max(0, current_start), current_end, current_errors))
        
        return ranges
    
    def _extract_clip(self,
                      video_path: str,
                      start_time: float,
                      end_time: float,
                      output_dir: str,
                      clip_index: int,
                      error_types: List[str]) -> str:
        """Extract a single clip from video"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Generate filename
        error_type_str = "_".join(set(error_types)).replace(" ", "_")
        output_path = Path(output_dir) / f"error_clip_{clip_index:03d}_{error_type_str}.mp4"
        
        # Setup writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Seek to start
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        
        frames_written = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if current_time > end_time:
                break
            
            # Add timestamp overlay
            cv2.putText(frame, f"t={current_time:.2f}s", (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
            frames_written += 1
        
        cap.release()
        out.release()
        
        if frames_written > 0:
            logger.info(f"Extracted clip: {output_path.name} ({frames_written} frames)")
            return str(output_path)
        else:
            output_path.unlink(missing_ok=True)
            return None
    
    def create_highlights_reel(self,
                               clip_paths: List[str],
                               output_path: str,
                               add_transitions: bool = True) -> str:
        """
        Combine multiple error clips into a single highlights video.
        
        Args:
            clip_paths: List of clip file paths
            output_path: Output highlights video path
            add_transitions: Add fade transitions between clips
            
        Returns:
            Path to highlights video
        """
        if not clip_paths:
            logger.warning("No clips to combine")
            return None
        
        # Read first clip to get properties
        first_cap = cv2.VideoCapture(clip_paths[0])
        fps = first_cap.get(cv2.CAP_PROP_FPS)
        width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        first_cap.release()
        
        # Setup writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        transition_frames = int(fps * 0.5) if add_transitions else 0
        
        for i, clip_path in enumerate(clip_paths):
            cap = cv2.VideoCapture(clip_path)
            
            # Add title frame
            title_frame = np.zeros((height, width, 3), dtype=np.uint8)
            title_text = f"Error Clip {i+1}/{len(clip_paths)}"
            cv2.putText(title_frame, title_text, (width//2 - 200, height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            for _ in range(int(fps)):  # 1 second title
                out.write(title_frame)
            
            # Add clip frames
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            
            cap.release()
            
            # Add transition
            if add_transitions and i < len(clip_paths) - 1:
                for j in range(transition_frames):
                    alpha = j / transition_frames
                    fade_frame = (frame * (1 - alpha)).astype(np.uint8)
                    out.write(fade_frame)
        
        out.release()
        logger.info(f"Created highlights reel: {output_path}")
        return output_path
