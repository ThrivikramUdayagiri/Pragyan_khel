"""
Frame interpolation using optical flow for generating intermediate frames.
Lightweight alternative to RIFE/DAIN that doesn't require ML models.
"""

import cv2
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)


class FrameInterpolator:
    """
    Interpolates frames using optical flow to generate smooth intermediate frames.
    Optimized for speed using Farneback optical flow.
    """
    
    def __init__(self, use_fast_mode: bool = True):
        self.use_fast_mode = use_fast_mode
        # Farneback is 2-3x faster than DIS with good quality
        logger.info(f"FrameInterpolator initialized with fast_mode={use_fast_mode}")
    
    def interpolate_between_frames(self, frame1: np.ndarray, frame2: np.ndarray, num_intermediate: int = 1) -> List[np.ndarray]:
        """
        Generate intermediate frames between two consecutive frames using optical flow.
        
        Args:
            frame1: First frame (BGR)
            frame2: Second frame (BGR)
            num_intermediate: Number of intermediate frames to generate
            
        Returns:
            List of interpolated frames (does not include frame1 and frame2)
        """
        # Scale down for faster processing if enabled
        h, w = frame1.shape[:2]
        if self.use_fast_mode and max(h, w) > 720:
            scale = 720.0 / max(h, w)
            small_h, small_w = int(h * scale), int(w * scale)
            frame1_small = cv2.resize(frame1, (small_w, small_h), interpolation=cv2.INTER_AREA)
            frame2_small = cv2.resize(frame2, (small_w, small_h), interpolation=cv2.INTER_AREA)
        else:
            frame1_small = frame1
            frame2_small = frame2
            scale = 1.0
        
        gray1 = cv2.cvtColor(frame1_small, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2_small, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow using Farneback (faster than DIS)
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )
        
        interpolated_frames = []
        
        h_small, w_small = gray1.shape
        x_small, y_small = np.meshgrid(np.arange(w_small), np.arange(h_small))
        
        for i in range(1, num_intermediate + 1):
            # Calculate interpolation weight (0 < alpha < 1)
            alpha = i / (num_intermediate + 1)
            
            # Apply scaled flow
            map_x = (x_small - alpha * flow[..., 0]).astype(np.float32)
            map_y = (y_small - alpha * flow[..., 1]).astype(np.float32)
            
            # Warp frame2 backwards
            warped_frame2 = cv2.remap(frame2_small, map_x, map_y, cv2.INTER_LINEAR)
            
            # Also warp frame1 forward for better blending
            map_x_fwd = (x_small + (1 - alpha) * flow[..., 0]).astype(np.float32)
            map_y_fwd = (y_small + (1 - alpha) * flow[..., 1]).astype(np.float32)
            warped_frame1 = cv2.remap(frame1_small, map_x_fwd, map_y_fwd, cv2.INTER_LINEAR)
            
            # Blend the two warped frames
            interpolated = cv2.addWeighted(warped_frame1, 1 - alpha, warped_frame2, alpha, 0)
            
            # Scale back to original resolution if needed
            if scale != 1.0:
                interpolated = cv2.resize(interpolated, (w, h), interpolation=cv2.INTER_LINEAR)
            
            interpolated_frames.append(interpolated)
        
        return interpolated_frames
    
    def interpolate_video(self, input_path: str, output_path: str, target_fps: int, source_fps: float = None) -> dict:
        """
        Interpolate an entire video to achieve target FPS.
        
        Args:
            input_path: Path to input video
            output_path: Path to save interpolated video
            target_fps: Desired output FPS (e.g., 240)
            source_fps: Source FPS (auto-detected if None)
            
        Returns:
            Dictionary with interpolation statistics
        """
        cap = cv2.VideoCapture(input_path)
        
        if source_fps is None:
            source_fps = cap.get(cv2.CAP_PROP_FPS)
        
        if source_fps >= target_fps:
            logger.warning(f"Source FPS ({source_fps}) >= target FPS ({target_fps}), no interpolation needed")
            cap.release()
            return {
                "source_fps": source_fps,
                "target_fps": target_fps,
                "interpolation_factor": 1,
                "interpolated": False,
                "reason": "Source FPS already sufficient"
            }
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate how many intermediate frames needed between each source frame
        interpolation_factor = int(target_fps / source_fps)
        num_intermediate = interpolation_factor - 1
        
        logger.info(f"Interpolating video: {source_fps} FPS → {target_fps} FPS")
        logger.info(f"Generating {num_intermediate} intermediate frame(s) between each source frame")
        logger.info(f"Total source frames: {total_frames}, Expected output: ~{total_frames * interpolation_factor}")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
        
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError("Could not read first frame")
        
        out.write(prev_frame)
        frames_written = 1
        frames_processed = 1
        
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            # Generate intermediate frames
            if num_intermediate > 0:
                intermediate_frames = self.interpolate_between_frames(prev_frame, curr_frame, num_intermediate)
                
                for interp_frame in intermediate_frames:
                    out.write(interp_frame)
                    frames_written += 1
            
            # Write current frame
            out.write(curr_frame)
            frames_written += 1
            frames_processed += 1
            
            if frames_processed % 5 == 0:
                progress_pct = (frames_processed / total_frames) * 100
                logger.info(f"Interpolation: {frames_processed}/{total_frames} frames ({progress_pct:.1f}%), generated {frames_written} output frames")
            
            prev_frame = curr_frame
        
        cap.release()
        out.release()
        
        result_fps = frames_written / (total_frames / source_fps)
        
        logger.info(f"✓ Interpolation complete: {frames_written} frames written")
        logger.info(f"✓ Achieved FPS: {result_fps:.2f}")
        
        return {
            "source_fps": source_fps,
            "target_fps": target_fps,
            "interpolation_factor": interpolation_factor,
            "num_intermediate_per_pair": num_intermediate,
            "source_frames": total_frames,
            "output_frames": frames_written,
            "achieved_fps": result_fps,
            "interpolated": True
        }
