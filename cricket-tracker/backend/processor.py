"""
Video Processor Module - Highlight Effect Generation

This module creates highlight videos where:
- Selected player: Remains CLEAR and SHARP (original quality)
- Everything else (other players + background): GAUSSIAN BLURRED

Author: Cricket Tracker Team
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List, Tuple
import subprocess
import shutil


class VideoProcessor:
    """
    Video processor class for generating highlight effects.
    
    Creates a focus effect where the selected player is clearly visible
    while the rest of the frame is blurred.
    """
    
    def __init__(
        self,
        blur_strength: int = 35,  # Gaussian blur kernel size
        edge_feather: int = 15,   # Feathering for smooth edges
        blur_intensity: float = 0.15  # Blur blend intensity (0-1)
    ):
        """
        Initialize the video processor with effect parameters.
        
        Args:
            blur_strength: Gaussian blur kernel size (higher = more blur)
            edge_feather: Feathering amount for smooth mask edges
            blur_intensity: How much blur to apply (0 = no blur, 1 = full blur)
        """
        self.blur_strength = blur_strength
        self.edge_feather = edge_feather
        self.blur_intensity = blur_intensity
    
    def generate_highlight_video(
        self,
        input_path: str,
        output_path: str,
        tracking_data: Dict[str, Any],
        selected_player_id: int,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ):
        """
        Generate a highlight video with the selected player in focus.
        
        The selected player remains clear while everything else is blurred.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            tracking_data: Dictionary containing tracking data for all frames
            selected_player_id: ID of the player to highlight
            progress_callback: Callback for progress updates
        """
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_number = 0
        
        if progress_callback:
            progress_callback(0, "Starting highlight generation...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get tracking data for this frame
            frame_key = str(frame_number)
            frame_data = tracking_data.get("frames", {}).get(frame_key, {})
            players = frame_data.get("players", [])
            
            # Process frame with effects
            processed_frame = self._process_frame(
                frame,
                players,
                selected_player_id,
                width,
                height
            )
            
            # Write processed frame
            out.write(processed_frame)
            
            frame_number += 1
            
            # Update progress
            if progress_callback and frame_number % 5 == 0:
                progress = int((frame_number / total_frames) * 100)
                progress_callback(
                    progress,
                    f"Applying effects: frame {frame_number}/{total_frames}"
                )
        
        # Cleanup
        cap.release()
        out.release()
        
        # Convert to H.264 for browser compatibility
        self._convert_to_h264(output_path)
        
        if progress_callback:
            progress_callback(100, "Highlight video complete!")
    
    def _process_frame(
        self,
        frame: np.ndarray,
        players: List[Dict[str, Any]],
        selected_player_id: int,
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Apply focus effect to a single frame.
        
        The selected player stays clear/sharp, everything else is blurred.
        
        Args:
            frame: Input frame (BGR format)
            players: List of player data for this frame
            selected_player_id: ID of the selected player
            width: Frame width
            height: Frame height
            
        Returns:
            Processed frame with blur effect applied
        """
        # Create mask for the selected player
        selected_player_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Find the selected player and create their mask
        for player in players:
            player_id = player.get("id")
            if player_id == selected_player_id:
                bbox = player.get("bbox", [])
                mask_polygon = player.get("mask_polygon", [])
                
                # Create mask from polygon or bbox
                selected_player_mask = self._create_player_mask(
                    width, height, bbox, mask_polygon
                )
                break
        
        # If no selected player found in this frame, return blurred frame
        if not np.any(selected_player_mask > 0):
            blur_kernel = self.blur_strength * 2 + 1
            return cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
        
        # Step 1: Create fully blurred version of entire frame
        blur_kernel = self.blur_strength * 2 + 1
        blurred_frame = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
        
        # Step 2: Create smooth mask with feathered edges
        # This creates a gradual transition from sharp to blurred
        feather_kernel = self.edge_feather * 2 + 1
        smooth_mask = cv2.GaussianBlur(
            selected_player_mask.astype(np.float32),
            (feather_kernel, feather_kernel),
            0
        )
        
        # Normalize mask to 0-1 range
        if smooth_mask.max() > 0:
            smooth_mask = smooth_mask / smooth_mask.max()
        
        # Expand to 3 channels
        smooth_mask_3ch = np.stack([smooth_mask] * 3, axis=-1)
        
        # Step 3: Composite - blend original (sharp) with blurred
        # Where mask is 1 (selected player): show original
        # Where mask is 0 (background): show blurred (scaled by blur_intensity)
        # Apply blur_intensity to control how much blur is visible in background
        background_blend = (
            frame.astype(np.float32) * (1 - self.blur_intensity) +
            blurred_frame.astype(np.float32) * self.blur_intensity
        )
        
        result = (
            frame.astype(np.float32) * smooth_mask_3ch +
            background_blend * (1 - smooth_mask_3ch)
        ).astype(np.uint8)
        
        return result
    
    def _create_player_mask(
        self,
        width: int,
        height: int,
        bbox: List[int],
        mask_polygon: List[int]
    ) -> np.ndarray:
        """
        Create a binary mask for a player from polygon or bounding box.
        
        Args:
            width: Frame width
            height: Frame height
            bbox: Bounding box [x1, y1, x2, y2]
            mask_polygon: Flattened polygon points
            
        Returns:
            Binary mask image
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Try to use polygon mask first (more accurate segmentation)
        if mask_polygon and len(mask_polygon) >= 6:
            try:
                points = np.array(mask_polygon).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [points], 255)
                return mask
            except Exception:
                pass
        
        # Fall back to bounding box with elliptical shape
        if len(bbox) >= 4:
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            
            # Clamp to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width - 1, x2)
            y2 = min(height - 1, y2)
            
            # Create ellipse for more natural player shape
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = ((x2 - x1) // 2, (y2 - y1) // 2)
            
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        return mask
    
    def _convert_to_h264(self, video_path: str):
        """Convert video to H.264 codec for browser compatibility."""
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
