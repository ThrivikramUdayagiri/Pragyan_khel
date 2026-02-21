"""
Video Repair Module
===================
Automatically fix temporal errors in videos.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class VideoRepairer:
    """Repair temporal errors in videos"""
    
    def __init__(self):
        self.interpolation_method = cv2.INTER_LINEAR
    
    def repair_video(self,
                     input_path: str,
                     results: List[Dict],
                     output_path: str,
                     fix_drops: bool = True,
                     fix_merges: bool = True,
                     fix_reversals: bool = True,
                     interpolate_drops: bool = True,
                     use_optical_flow: bool = False) -> Dict:
        """
        Repair temporal errors in video.
        
        ✅ Frame Drop Correction:
        - Inserts synthetic frames using interpolation
        - Uses motion estimation (optical flow) for smooth reconstruction
        - Restores missing temporal information
        
        ✅ Frame Merge Correction:
        - Replaces ghosted/blended frames with clean interpolated frames
        - Detects and reconstructs corrupted frames
        - Prevents secondary frame drops
        
        ✅ Frame Reversal Correction (Boomerang Fix):
        - Removes reversed/repeated frames that create ping-pong effect
        - Ensures smooth forward motion only
        - Eliminates stuttering caused by frame reuse
        
        Args:
            input_path: Input video path
            results: Detection results from TemporalErrorDetector
            output_path: Output repaired video path
            fix_drops: Fix frame drops by inserting synthetic frames
            fix_merges: Fix frame merges by replacing ghosted frames
            fix_reversals: Fix frame reversals/boomerang effects by removing repeated frames
            interpolate_drops: Interpolate missing frames (vs duplicate previous)
            use_optical_flow: Use optical flow for better motion estimation
            
        Returns:
            Dictionary with repair statistics
        """
        logger.info(f"Starting video repair: {input_path}")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create DataFrame for easier manipulation
        df = pd.DataFrame(results)
        
        # Identify errors to fix
        drops_to_fix = []
        merges_to_fix = []
        reversals_to_fix = []
        
        if fix_drops:
            drops_to_fix = df[df['classification'] == 'Frame Drop']['frame_num'].tolist()
        
        if fix_merges:
            merges_to_fix = df[df['classification'] == 'Frame Merge']['frame_num'].tolist()
        
        if fix_reversals:
            reversals_to_fix = df[df['classification'] == 'Frame Reversal']['frame_num'].tolist()
        
        logger.info(f"Fixing {len(drops_to_fix)} drops, {len(merges_to_fix)} merges, {len(reversals_to_fix)} reversals")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process video
        frame_num = 0
        prev_frame = None
        next_frame_cache = None
        frames_added = 0
        frames_replaced = 0
        frames_removed = 0
        
        # Read all frames first for better interpolation
        all_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        
        cap.release()
        
        # Now process with look-ahead capability
        for idx in range(len(all_frames)):
            frame_num = idx + 1
            current_frame = all_frames[idx]
            prev_frame = all_frames[idx - 1] if idx > 0 else None
            next_frame = all_frames[idx + 1] if idx < len(all_frames) - 1 else None
            
            # ===== HANDLE FRAME REVERSALS (Boomerang Effect) =====
            # REMOVE reversed/repeated frames to ensure smooth forward motion
            if frame_num in reversals_to_fix:
                frames_removed += 1
                logger.debug(f"🔄 Removed reversed frame {frame_num} (boomerang effect)")
                continue  # Skip this frame entirely
            
            # ===== HANDLE FRAME DROPS =====
            # Insert synthetic frame BEFORE this position
            if frame_num in drops_to_fix and prev_frame is not None:
                if interpolate_drops:
                    if use_optical_flow and next_frame is not None:
                        # Use advanced optical flow interpolation
                        try:
                            interpolated = self.advanced_interpolate_frame(prev_frame, current_frame)
                            out.write(interpolated)
                            frames_added += 1
                            logger.debug(f"✅ Optical flow interpolated frame at {frame_num}")
                        except Exception as e:
                            logger.warning(f"Optical flow failed at {frame_num}, using simple blend: {e}")
                            interpolated = cv2.addWeighted(prev_frame, 0.5, current_frame, 0.5, 0)
                            out.write(interpolated)
                            frames_added += 1
                    else:
                        # Simple interpolation between previous and current
                        interpolated = cv2.addWeighted(prev_frame, 0.5, current_frame, 0.5, 0)
                        out.write(interpolated)
                        frames_added += 1
                        logger.debug(f"✅ Interpolated synthetic frame at {frame_num}")
                else:
                    # Duplicate previous frame
                    out.write(prev_frame)
                    frames_added += 1
                    logger.debug(f"✅ Duplicated frame at {frame_num}")
            
            # ===== HANDLE FRAME MERGES =====
            # REPLACE corrupted ghosted frame with clean interpolation
            if frame_num in merges_to_fix:
                if prev_frame is not None and next_frame is not None:
                    # Detect ghosting and replace with clean interpolated frame
                    if use_optical_flow:
                        try:
                            # Use optical flow to reconstruct clean frame
                            reconstructed = self.advanced_interpolate_frame(prev_frame, next_frame)
                            out.write(reconstructed)
                            frames_replaced += 1
                            logger.debug(f"✅ Replaced ghosted frame {frame_num} with optical flow reconstruction")
                        except Exception as e:
                            logger.warning(f"Optical flow failed at {frame_num}, using blend: {e}")
                            reconstructed = cv2.addWeighted(prev_frame, 0.5, next_frame, 0.5, 0)
                            out.write(reconstructed)
                            frames_replaced += 1
                    else:
                        # Simple interpolation replacement
                        reconstructed = cv2.addWeighted(prev_frame, 0.5, next_frame, 0.5, 0)
                        out.write(reconstructed)
                        frames_replaced += 1
                        logger.debug(f"✅ Replaced ghosted frame {frame_num} with interpolated clean frame")
                elif prev_frame is not None:
                    # Use previous clean frame
                    out.write(prev_frame)
                    frames_replaced += 1
                    logger.debug(f"✅ Replaced ghosted frame {frame_num} with previous clean frame")
                else:
                    # No replacement possible, write as-is
                    out.write(current_frame)
                    logger.warning(f"⚠️ Cannot replace frame {frame_num}, no clean frames available")
                continue  # Skip to next frame
            
            # ===== WRITE NORMAL CLEAN FRAME =====
            out.write(current_frame)
        
        cap.release()
        out.release()
        
        stats = {
            'input_path': input_path,
            'output_path': output_path,
            'total_frames': len(all_frames),
            'frames_added': frames_added,
            'frames_replaced': frames_replaced,
            'frames_removed': frames_removed,
            'drops_fixed': len(drops_to_fix),
            'merges_fixed': len(merges_to_fix),
            'reversals_fixed': len(reversals_to_fix),
            'interpolation_method': 'Optical Flow' if use_optical_flow else 'Simple Blend',
            'errors_corrected': len(drops_to_fix) + len(merges_to_fix) + len(reversals_to_fix)
        }
        
        logger.info(f"✅ Video repair complete: {frames_added} inserted, {frames_replaced} replaced, {frames_removed} removed")
        logger.info(f"✅ Total errors corrected: {stats['errors_corrected']} ({len(drops_to_fix)} drops + {len(merges_to_fix)} merges + {len(reversals_to_fix)} reversals)")
        return stats
    
    def detect_ghosting(self, frame: np.ndarray, prev_frame: np.ndarray, 
                       next_frame: np.ndarray, threshold: float = 0.15) -> bool:
        """
        Detect ghosting/blending artifacts in a frame.
        
        Ghosting occurs when two frames are merged/blended together,
        creating a semi-transparent double-image effect.
        
        Args:
            frame: Current frame to check
            prev_frame: Previous frame
            next_frame: Next frame
            threshold: Blending detection threshold (lower = more sensitive)
            
        Returns:
            True if ghosting is detected
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        
        # Check if current frame looks like blend of prev and next
        # Ghosted frame ≈ 0.5 * prev + 0.5 * next
        expected_blend = cv2.addWeighted(prev_gray, 0.5, next_gray, 0.5, 0)
        
        # Compute difference
        diff = cv2.absdiff(gray, expected_blend)
        mean_diff = np.mean(diff) / 255.0
        
        # If mean difference is low, frame is likely a blend
        is_ghosted = mean_diff < threshold
        
        if is_ghosted:
            logger.debug(f"🔍 Ghosting detected: diff={mean_diff:.3f}")
        
        return is_ghosted
    
    def interpolate_frame(self,
                         prev_frame: np.ndarray,
                         next_frame: np.ndarray,
                         alpha: float = 0.5) -> np.ndarray:
        """
        Interpolate between two frames.
        
        Args:
            prev_frame: Previous frame
            next_frame: Next frame
            alpha: Blend factor (0.5 = equal blend)
            
        Returns:
            Interpolated frame
        """
        return cv2.addWeighted(prev_frame, 1 - alpha, next_frame, alpha, 0)
    
    def advanced_interpolate_frame(self,
                                   prev_frame: np.ndarray,
                                   next_frame: np.ndarray) -> np.ndarray:
        """
        Advanced frame interpolation using optical flow.
        
        Args:
            prev_frame: Previous frame
            next_frame: Next frame
            
        Returns:
            Interpolated frame
        """
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Create coordinate grids
        h, w = prev_frame.shape[:2]
        flow_map = np.array(np.meshgrid(np.arange(w), np.arange(h)), dtype=np.float32)
        
        # Apply half flow to get middle frame
        flow_map[0] += flow[..., 0] * 0.5
        flow_map[1] += flow[..., 1] * 0.5
        
        # Remap to create interpolated frame
        interpolated = cv2.remap(prev_frame, flow_map[0], flow_map[1],
                                cv2.INTER_LINEAR)
        
        return interpolated
    
    def remove_duplicate_frames(self,
                               input_path: str,
                               output_path: str,
                               threshold: float = 0.99) -> Dict:
        """
        Remove duplicate/very similar consecutive frames.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            threshold: SSIM threshold for duplicate detection
            
        Returns:
            Statistics dictionary
        """
        from skimage.metrics import structural_similarity as ssim
        
        logger.info(f"Removing duplicate frames (threshold={threshold})")
        
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        prev_gray = None
        frames_kept = 0
        frames_removed = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                # Calculate SSIM
                similarity = ssim(prev_gray, gray)
                
                if similarity >= threshold:
                    # Skip duplicate frame
                    frames_removed += 1
                    continue
            
            # Keep frame
            out.write(frame)
            frames_kept += 1
            prev_gray = gray
        
        cap.release()
        out.release()
        
        stats = {
            'input_path': input_path,
            'output_path': output_path,
            'total_frames': total_frames,
            'frames_kept': frames_kept,
            'frames_removed': frames_removed,
            'reduction_percentage': (frames_removed / total_frames * 100) if total_frames > 0 else 0
        }
        
        logger.info(f"Removed {frames_removed} duplicate frames ({stats['reduction_percentage']:.1f}%)")
        return stats
    
    def fix_variable_framerate(self,
                               input_path: str,
                               output_path: str,
                               target_fps: float = None) -> Dict:
        """
        Convert variable frame rate video to constant frame rate.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            target_fps: Target frame rate (None = use input fps)
            
        Returns:
            Statistics dictionary
        """
        logger.info("Converting to constant frame rate")
        
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if target_fps is None:
            target_fps = fps
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
        
        expected_interval = 1.0 / target_fps
        current_time = 0.0
        frames_written = 0
        
        all_frames = []
        all_timestamps = []
        
        # Read all frames first
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            all_frames.append(frame)
            all_timestamps.append(timestamp)
        
        cap.release()
        
        # Write frames at constant intervals
        frame_idx = 0
        while frame_idx < len(all_frames):
            # Find closest frame to current target time
            target_time = current_time
            
            # Find frame closest to target time
            closest_idx = min(range(len(all_timestamps)),
                            key=lambda i: abs(all_timestamps[i] - target_time))
            
            out.write(all_frames[closest_idx])
            frames_written += 1
            current_time += expected_interval
            
            frame_idx = closest_idx + 1
        
        out.release()
        
        stats = {
            'input_path': input_path,
            'output_path': output_path,
            'input_frames': total_frames,
            'output_frames': frames_written,
            'input_fps': fps,
            'output_fps': target_fps
        }
        
        logger.info(f"CFR conversion complete: {frames_written} frames at {target_fps} fps")
        return stats
