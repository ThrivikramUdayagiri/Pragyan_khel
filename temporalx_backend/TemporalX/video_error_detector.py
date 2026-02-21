"""
Video Temporal Error Detection System
======================================
A research-level implementation for detecting frame drops and frame merges
using hybrid computer vision techniques.

Author: TemporalX Team
Date: February 2026
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import Tuple, Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TemporalErrorDetector:
    """
    Advanced Video Temporal Error Detection System using hybrid approaches.
    
    Detects:
        - Frame Drops: Missing frames in video sequence
        - Frame Merges: Blended/ghosted frames caused by encoding errors
    
    Methods:
        - Timestamp-based irregularity detection
        - Optical Flow analysis (Farneback)
        - SSIM for structural similarity
        - Histogram difference analysis
        - Edge detection for ghosting artifacts
    """
    
    def __init__(self, 
                 flow_threshold: float = 30.0,
                 ssim_threshold: float = 0.85,
                 hist_threshold: float = 0.3,
                 timestamp_tolerance: float = 1.5,
                 resize_width: int = 640,
                 auto_tune: bool = True,
                 reversal_detection: bool = True):
        """
        Initialize the detector with configurable thresholds.
        
        Args:
            flow_threshold: Optical flow magnitude threshold for frame drops
            ssim_threshold: SSIM threshold for frame merges (lower = more similar)
            hist_threshold: Histogram difference threshold
            timestamp_tolerance: Multiplier for expected frame interval
            resize_width: Width to resize frames for faster processing
            auto_tune: Enable automatic threshold tuning based on video statistics
            reversal_detection: Enable boomerang/reversal effect detection
        """
        self.flow_threshold = flow_threshold
        self.ssim_threshold = ssim_threshold
        self.hist_threshold = hist_threshold
        self.timestamp_tolerance = timestamp_tolerance
        self.resize_width = resize_width
        self.auto_tune = auto_tune
        self.reversal_detection = reversal_detection
        
        # Statistics for auto-tuning
        self.flow_history = []
        self.ssim_history = []
        self.hist_history = []
        self.laplacian_history = []  # ← NEW: Track laplacian for adaptive blur threshold
        
        # Frame history for reversal detection (keep last 5 frames)
        self.frame_history = []
        self.max_history = 5
        
        logger.info(f"Initialized TemporalErrorDetector with thresholds: "
                   f"flow={flow_threshold}, ssim={ssim_threshold}, hist={hist_threshold}, "
                   f"reversal_detection={reversal_detection}")
    
    def load_video(self, video_path: str) -> Tuple[cv2.VideoCapture, Dict]:
        """
        Load video and extract metadata.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Tuple of (VideoCapture object, metadata dictionary)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
        metadata = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'expected_interval': 1.0 / cap.get(cv2.CAP_PROP_FPS)
        }
        
        logger.info(f"Loaded video: {metadata['frame_count']} frames at {metadata['fps']} FPS")
        return cap, metadata
    
    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess frame for efficient computation.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (grayscale frame, resized grayscale frame)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate resize height maintaining aspect ratio
        h, w = gray.shape
        resize_height = int(h * (self.resize_width / w))
        gray_resized = cv2.resize(gray, (self.resize_width, resize_height))
        
        return gray, gray_resized
    
    
    def compute_motion_direction_consistency(self, flow: np.ndarray) -> float:
        """
        🧠 ELITE: Measure motion direction smoothness/consistency
        
        Drops cause chaotic/discontinuous flow vectors.
        Normal motion shows smooth, continuous flow direction.
        
        Args:
            flow: Optical flow field (2-channel: flow_x, flow_y)
            
        Returns:
            Consistency score [0, 1] where:
            - 1.0 = perfectly consistent direction (smooth pan/camera movement)
            - 0.5 = mixed directions (normal object motion)
            - 0.0 = chaotic directions (likely frame drop with jump discontinuity)
        """
        # Extract flow components
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        
        # Compute flow magnitude and angle
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        
        # Only consider pixels with significant motion
        motion_threshold = np.mean(magnitude) * 0.5
        mask = magnitude > motion_threshold
        
        if np.sum(mask) < 10:  # Not enough motion to analyze
            return 1.0  # No clear motion signature (benign)
        
        # Get angles of moving pixels
        angles = np.arctan2(flow_y[mask], flow_x[mask])
        
        # Compute consistency metrics
        # 1. Angular variance - how spread out are the flow directions?
        angle_sin = np.sin(angles)
        angle_cos = np.cos(angles)
        
        mean_sin = np.mean(angle_sin)
        mean_cos = np.mean(angle_cos)
        
        # Resultant vector length (0 = chaotic, 1 = consistent)
        # This is a standard measure of directional concentration
        resultant_length = np.sqrt(mean_sin**2 + mean_cos**2)
        
        # Convert to consistency score [0, 1]
        # resultant_length ≥ 0.7 = smooth/consistent direction
        # resultant_length < 0.3 = chaotic/discontinuous
        consistency_score = max(resultant_length, 0.0)
        
        return consistency_score

    def compute_optical_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute dense optical flow using Farneback algorithm.
        
        Args:
            prev_frame: Previous grayscale frame
            curr_frame: Current grayscale frame
            
        Returns:
            Tuple of (mean flow magnitude, flow field)
        """
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, curr_frame,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Calculate flow magnitude
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_magnitude = np.mean(magnitude)
        
        return mean_magnitude, flow
    
    def compute_ssim(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute Structural Similarity Index between two frames.
        
        Args:
            frame1: First grayscale frame
            frame2: Second grayscale frame
            
        Returns:
            SSIM score (0 to 1, higher = more similar)
        """
        score = ssim(frame1, frame2, data_range=frame2.max() - frame2.min())
        return score
    
    def compute_histogram_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute histogram difference for scene change detection.
        
        Args:
            frame1: First grayscale frame
            frame2: Second grayscale frame
            
        Returns:
            Normalized histogram difference (0 to 1)
        """
        hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
        
        # Normalize histograms and convert to float32
        hist1 = cv2.normalize(hist1, hist1).astype(np.float32)
        hist2 = cv2.normalize(hist2, hist2).astype(np.float32)
        
        # Compute chi-square distance
        diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR_ALT)
        
        return diff
    
    def detect_ghosting_artifacts(self, frame: np.ndarray) -> float:
        """
        Detect ghosting/double-edge artifacts characteristic of frame merges.
        
        Args:
            frame: Grayscale frame
            
        Returns:
            Ghosting score (higher = more artifacts)
        """
        # Apply Laplacian for edge detection
        laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        
        # Compute variance of Laplacian (blur detection)
        laplacian_var = laplacian.var()
        
        # Detect double edges using multi-scale edge detection
        edges1 = cv2.Canny(frame, 50, 150)
        edges2 = cv2.Canny(frame, 100, 200)
        
        # Calculate edge density difference (ghosting causes multiple edge responses)
        edge_diff = np.abs(np.sum(edges1) - np.sum(edges2)) / (frame.shape[0] * frame.shape[1])
        
        return laplacian_var, edge_diff
    
    def detect_frame_drop(self, 
                         timestamp_diff: float,
                         expected_interval: float,
                         flow_magnitude: float,
                         hist_diff: float,
                         flow: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        """
        🧠 ELITE FRAME DROP DETECTION
        
        Advanced multi-signal detection with motion direction consistency check.
        
        Detection Rule:
        ├─ If timestamp gap > threshold → LIKELY DROP (strong signal)
        └─ OR if (motion spike AND histogram discontinuity) → LIKELY DROP (combined signal)
           └─ BUT if motion direction is smooth/continuous → NOT A DROP (override)
        
        This prevents normal high-motion content (camera pan, fast action) from being
        mistaken for missing frames by checking if flow vectors are consistent.
        
        Args:
            timestamp_diff: Actual time difference between frames
            expected_interval: Expected time interval (1/FPS)
            flow_magnitude: Optical flow magnitude
            hist_diff: Histogram difference
            flow: Optical flow field (optional, for elite motion consistency check)
            
        Returns:
            Tuple of (is_drop: bool, confidence: float [0-1])
        """
        # ===== SIGNAL 1: Timestamp Regularity Score =====
        timestamp_ratio = timestamp_diff / expected_interval
        # Strong signal: if timestamp gap exceeds tolerance significantly
        timestamp_score = min((timestamp_ratio - 1.0) / (self.timestamp_tolerance - 1.0), 1.0)
        timestamp_score = max(timestamp_score, 0.0)
        
        # STRONG SIGNAL: Extreme timestamp gap almost certainly indicates drop
        extreme_timestamp = timestamp_ratio > self.timestamp_tolerance
        
        # ===== SIGNAL 2: Optical Flow Magnitude Score =====
        if len(self.flow_history) > 20:
            flow_mean = np.mean(self.flow_history[-20:])
            flow_std = np.std(self.flow_history[-20:])
            flow_dynamic_threshold = flow_mean + (2.5 * flow_std)
        else:
            flow_dynamic_threshold = self.flow_threshold
        
        if flow_dynamic_threshold > 0:
            flow_score = min(flow_magnitude / flow_dynamic_threshold, 1.0)
        else:
            flow_score = 0.0
        
        motion_spike = flow_score > 0.6
        
        # ===== SIGNAL 3: Histogram Discontinuity Score =====
        hist_score = min(hist_diff / max(self.hist_threshold * 2, 1.0), 1.0)
        hist_discontinuity = hist_score > 0.5
        
        # ===== 🧠 ELITE SIGNAL 4: Motion Direction Consistency =====
        # Check if flow direction is smooth/continuous (not chaotic)
        # Chaotic flow = frame drop, Smooth flow = normal motion
        motion_consistency = 1.0  # Default: assume smooth motion
        
        if flow is not None and flow.shape[0] > 0 and flow.shape[1] > 0:
            motion_consistency = self.compute_motion_direction_consistency(flow)
        
        smooth_motion = motion_consistency > 0.4  # > 0.4 = reasonably consistent
        
        # ===== ADVANCED DETECTION LOGIC =====
        # Apply the elite rules:
        # 1. Extreme timestamp gap → almost certainly a drop (strong signal)
        if extreme_timestamp:
            drop_confidence = 0.85 + (0.15 * flow_score)  # High confidence, modulated by flow
            is_drop = True
        
        # 2. High motion spike + histogram discontinuity → likely drop
        elif motion_spike and hist_discontinuity:
            # BUT: if motion direction is smooth/consistent, override to NOT a drop
            if smooth_motion:
                # This is normal camera motion (pan, zoom, etc.), not a frame drop
                is_drop = False
                drop_confidence = 0.15  # Low confidence
            else:
                # Chaotic flow + motion spike + content change = frame drop
                drop_confidence = 0.70
                is_drop = True
        
        # 3. Everything else - use weighted scoring
        else:
            weights = [0.30, 0.40, 0.30]
            scores = [timestamp_score, flow_score, hist_score]
            drop_confidence = sum(w * s for w, s in zip(weights, scores))
            
            # Apply motion consistency override for low-confidence cases
            if drop_confidence < 0.50 and smooth_motion:
                # If motion is smooth, trust it (not a drop)
                drop_confidence *= 0.5  # Reduce confidence for drop
            
            is_drop = drop_confidence > 0.55
        
        # Logging with elite details
        if is_drop:
            logger.debug(f"🔴 DROP DETECTED: conf={drop_confidence:.3f}, "
                        f"timestamp={timestamp_score:.2f}, flow={flow_score:.2f}, "
                        f"hist={hist_score:.2f}, consistency={motion_consistency:.2f}")
        elif motion_spike and hist_discontinuity and smooth_motion:
            logger.debug(f"✅ HIGH MOTION (not drop): consistency={motion_consistency:.2f}, "
                        f"flow={flow_score:.2f}, smooth_motion detected")
        
        return is_drop, drop_confidence
    
    def detect_frame_merge(self,
                          ssim_score: float,
                          laplacian_var: float,
                          edge_diff: float) -> Tuple[bool, float]:
        """
        🎯 MULTI-SIGNAL SCORING DETECTION
        
        Detect frame merge/ghosting using weighted confidence scoring algorithm.
        
        Merge Score = w1 * Low SSIM + w2 * Edge Ghosting + w3 * Laplacian Blur + w4 * Frame Blending
        
        This combines four independent signals for detecting merged/ghosted frames:
        - SSIM Similarity (40% weight): Very similar frames indicate blending
        - Edge Ghosting Artifacts (25% weight): Double edges from merging
        - Laplacian Blur (20% weight): Low variance indicates merged frames
        - Frame Blending Pattern (15% weight): Characteristic of encoder errors
        
        Args:
            ssim_score: SSIM between current and previous frame
            laplacian_var: Laplacian variance (blur metric)
            edge_diff: Edge density difference
            
        Returns:
            Tuple of (is_merge: bool, confidence: float [0-1])
        """
        
        # ===== SIGNAL 1: SSIM (Frame Similarity) Score =====
        # Inverted: High SSIM = likely merge (score closer to 1)
        ssim_score_normalized = 1.0 - ssim_score  # Invert: very similar → high score
        ssim_score_normalized = max(ssim_score_normalized, 0.0)
        
        # ===== SIGNAL 2: Edge Ghosting Artifacts Score =====
        # High edge difference indicates ghosting
        edge_score = min(edge_diff / max(0.15, edge_diff), 1.0)
        
        # ===== SIGNAL 3: Laplacian Blur Score =====
        # Compute dynamic blur threshold from laplacian variance history
        # Low laplacian variance = blurry/merged frames
        if len(self.laplacian_history) > 20:
            # Use historical laplacian variance to set adaptive threshold
            laplacian_mean = np.mean(self.laplacian_history[-20:])
            laplacian_std = max(np.std(self.laplacian_history[-20:]), 1.0)
            # Threshold = mean - 1.5σ (deviations below this = likely blur/merge)
            blur_threshold = laplacian_mean - (1.5 * laplacian_std)
        else:
            # Default threshold until we have enough history
            blur_threshold = 50
        
        # Low Laplacian variance indicates blur/merging
        # Score = 0 when laplacian_var is normal, 1 when severely blurred
        blur_score = 1.0 - min(laplacian_var / max(blur_threshold, 1.0), 1.0)
        blur_score = max(blur_score, 0.0)
        
        # ===== SIGNAL 4: Frame Blending Pattern Score =====
        # Check if current frame looks like blend of previous frames
        # This is detected through SSIM history oscillation
        if len(self.ssim_history) > 3:
            recent_ssim = self.ssim_history[-3:]
            ssim_variance = np.var(recent_ssim)
            # High variance in recent SSIM suggests blending/ghosting pattern
            blending_score = min(ssim_variance / 0.05, 1.0)
        else:
            blending_score = 0.0
        
        # ===== WEIGHTED CONFIDENCE SCORING =====
        # Weights tuned for merge detection
        weights = [0.40, 0.25, 0.20, 0.15]  # [ssim, edge, blur, blending]
        scores = [ssim_score_normalized, edge_score, blur_score, blending_score]
        
        # Calculate weighted confidence score [0, 1]
        merge_confidence = sum(w * s for w, s in zip(weights, scores))
        
        # Classification threshold (empirically tuned)
        CONFIDENCE_THRESHOLD = 0.40  # Lower threshold for better merge detection
        is_merge = merge_confidence > CONFIDENCE_THRESHOLD
        
        if is_merge and ssim_score > 0.75:  # Lowered primary signal threshold
            logger.debug(f"🟡 Merge detected: merge_conf={merge_confidence:.3f}, "
                        f"ssim={ssim_score_normalized:.2f}, edge={edge_score:.2f}, "
                        f"blur={blur_score:.2f}, blend={blending_score:.2f}")
        
        return is_merge, merge_confidence
    
    def detect_frame_reversal(self,
                             curr_frame: np.ndarray,
                             prev_frame: np.ndarray,
                             flow: np.ndarray) -> Tuple[bool, float]:
        """
        Detect boomerang/reversal effect where frames play backwards or repeat in reverse order.
        
        This creates a stuttering "ping-pong" effect instead of smooth forward motion.
        Examples:
        - Normal: Frame 1 → 2 → 3 → 4 → 5
        - Reversal: Frame 1 → 2 → 3 → 2 → 1 (boomerang)
        - Repeat: Frame 1 → 2 → 1 → 2 → 1 (ping-pong)
        
        Detection methods:
        1. Check if current frame matches older frames (2-4 steps back) better than immediate previous
        2. Detect reversed flow patterns (negative temporal motion)
        3. Identify ping-pong similarity patterns
        
        Args:
            curr_frame: Current grayscale frame
            prev_frame: Previous grayscale frame
            flow: Optical flow field
            
        Returns:
            Tuple of (is_reversal: bool, confidence: float)
        """
        if not self.reversal_detection or len(self.frame_history) < 3:
            return False, 0.0
        
        signals = []
        
        # Signal 1: Check if current frame matches an older frame better than immediate previous
        # This indicates frame reuse/reversal (boomerang pattern)
        curr_vs_prev_ssim = self.compute_ssim(curr_frame, prev_frame)
        
        max_old_similarity = 0.0
        for old_frame in self.frame_history[-4:-1]:  # Check 2-4 frames back
            old_similarity = self.compute_ssim(curr_frame, old_frame)
            max_old_similarity = max(max_old_similarity, old_similarity)
        
        # If current matches old frame MUCH better than immediate previous -> reversal
        similarity_signal = (max_old_similarity > curr_vs_prev_ssim + 0.15) and (max_old_similarity > 0.90)
        signals.append(similarity_signal)
        
        # Signal 2: Detect reversed/negative motion patterns in optical flow
        # Check if flow vectors are predominantly pointing backwards
        flow_x = flow[..., 0]  # Horizontal component
        flow_y = flow[..., 1]  # Vertical component
        
        # Count negative vs positive flow (backward vs forward motion)
        negative_flow_ratio = np.sum(np.abs(flow_x[flow_x < 0])) / (np.sum(np.abs(flow_x)) + 1e-6)
        
        # High negative flow indicates backward motion (reversal)
        reversed_flow_signal = negative_flow_ratio > 0.6
        signals.append(reversed_flow_signal)
        
        # Signal 3: Detect ping-pong pattern (alternating high/low similarity)
        # Check if similarity oscillates between high and low
        if len(self.frame_history) >= 4:
            recent_ssim = []
            for i in range(len(self.frame_history) - 3, len(self.frame_history)):
                ssim_val = self.compute_ssim(self.frame_history[i], self.frame_history[i-1])
                recent_ssim.append(ssim_val)
            
            # Ping-pong creates alternating high/low pattern
            ssim_variance = np.var(recent_ssim) if len(recent_ssim) > 1 else 0
            pingpong_signal = ssim_variance > 0.05 and max_old_similarity > 0.88
            signals.append(pingpong_signal)
        
        # Weighted confidence
        weights = [0.5, 0.3, 0.2]
        confidence = sum(w for w, s in zip(weights, signals[:len(signals)]) if s)
        
        # Require at least 1 strong signal
        is_reversal = signals[0] or (sum(signals) >= 2)
        
        if is_reversal:
            logger.debug(f"🔄 Reversal detected: old_sim={max_old_similarity:.3f}, "
                        f"neg_flow={negative_flow_ratio:.3f}, signals={signals}")
        
        return is_reversal, confidence
    
    def classify_frame(self,
                      frame_data: Dict,
                      curr_frame: np.ndarray = None,
                      prev_frame: np.ndarray = None,
                      flow: np.ndarray = None) -> Tuple[str, float]:
        """
        Classify frame as Normal, Frame Drop, Frame Merge, or Frame Reversal.
        
        Args:
            frame_data: Dictionary containing all computed metrics
            curr_frame: Current grayscale frame (for reversal detection)
            prev_frame: Previous grayscale frame (for reversal detection)
            flow: Optical flow field (for reversal detection)
            
        Returns:
            Tuple of (classification: str, confidence: float)
        """
        # Check for frame reversal first (boomerang effect)
        if curr_frame is not None and prev_frame is not None and flow is not None:
            is_reversal, reversal_conf = self.detect_frame_reversal(curr_frame, prev_frame, flow)
            if is_reversal:
                return 'Frame Reversal', reversal_conf
        
        # Check for frame drop (higher priority than merge)
        is_drop, drop_conf = self.detect_frame_drop(
            frame_data['timestamp_diff'],
            frame_data['expected_interval'],
            frame_data['flow_magnitude'],
            frame_data['hist_diff'],
            flow=flow  # Pass optical flow for elite motion consistency check
        )
        
        if is_drop:
            return 'Frame Drop', drop_conf
        
        # Check for frame merge
        is_merge, merge_conf = self.detect_frame_merge(
            frame_data['ssim_score'],
            frame_data['laplacian_var'],
            frame_data['edge_diff']
        )
        
        if is_merge:
            return 'Frame Merge', merge_conf
        
        # Normal frame
        return 'Normal', 1.0 - max(drop_conf, merge_conf, reversal_conf if curr_frame is not None else 0)
    
    def annotate_frame(self,
                      frame: np.ndarray,
                      frame_num: int,
                      classification: str,
                      confidence: float,
                      metrics: Dict) -> np.ndarray:
        """
        Annotate frame with detection results.
        
        Args:
            frame: Input BGR frame
            frame_num: Frame number
            classification: Classification label
            confidence: Confidence score
            metrics: Dictionary of computed metrics
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Color coding
        colors = {
            'Normal': (0, 255, 0),          # Green
            'Frame Drop': (0, 0, 255),       # Red
            'Frame Merge': (0, 255, 255),    # Yellow
            'Frame Reversal': (255, 0, 255)  # Magenta (boomerang effect)
        }
        color = colors.get(classification, (128, 128, 128))  # Gray for unknown
        
        # Draw status bar
        cv2.rectangle(annotated, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.rectangle(annotated, (0, 0), (w, 120), color, 3)
        
        # Add text annotations
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(annotated, f"Frame: {frame_num}", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"Status: {classification}", (10, 60), font, 0.8, color, 2)
        cv2.putText(annotated, f"Confidence: {confidence:.2f}", (10, 90), font, 0.7, (255, 255, 255), 2)
        
        # Add metrics
        y_offset = h - 120
        cv2.rectangle(annotated, (0, y_offset), (w, h), (0, 0, 0), -1)
        cv2.putText(annotated, f"Flow: {metrics.get('flow_magnitude', 0):.2f}", 
                   (10, y_offset + 25), font, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, f"SSIM: {metrics.get('ssim_score', 0):.3f}", 
                   (10, y_offset + 50), font, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, f"Hist: {metrics.get('hist_diff', 0):.3f}", 
                   (10, y_offset + 75), font, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, f"Blur: {metrics.get('laplacian_var', 0):.1f}", 
                   (10, y_offset + 100), font, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def auto_tune_thresholds(self):
        """
        Automatically tune detection thresholds based on video statistics.
        Uses mean and standard deviation of collected metrics.
        """
        if len(self.flow_history) < 30:
            return
        
        # Calculate statistics
        flow_mean = np.mean(self.flow_history)
        flow_std = np.std(self.flow_history)
        
        ssim_mean = np.mean(self.ssim_history)
        ssim_std = np.std(self.ssim_history)
        
        hist_mean = np.mean(self.hist_history)
        hist_std = np.std(self.hist_history)
        
        # Update thresholds (2 std devs above mean for anomaly detection)
        self.flow_threshold = flow_mean + 2 * flow_std
        self.ssim_threshold = min(0.95, ssim_mean + 1.5 * ssim_std)
        self.hist_threshold = hist_mean + 2 * hist_std
        
        logger.info(f"Auto-tuned thresholds: flow={self.flow_threshold:.2f}, "
                   f"ssim={self.ssim_threshold:.3f}, hist={self.hist_threshold:.3f}")
    
    def process_video(self,
                     input_path: str,
                     output_path: str,
                     csv_path: str,
                     show_progress: bool = True) -> List[Dict]:
        """
        Process entire video and detect temporal errors.
        
        Args:
            input_path: Input video path
            output_path: Output annotated video path
            csv_path: Output CSV results path
            show_progress: Show real-time progress window
            
        Returns:
            List of detection results for each frame
        """
        # Load video
        cap, metadata = self.load_video(input_path)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, metadata['fps'],
                             (metadata['width'], metadata['height']))
        
        results = []
        frame_num = 0
        prev_gray = None
        prev_gray_resized = None
        prev_timestamp = 0
        flow_field = None
        
        logger.info("Starting video processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert to seconds
            
            # Preprocess
            gray, gray_resized = self.preprocess_frame(frame)
            
            # Maintain frame history for reversal detection
            if self.reversal_detection:
                self.frame_history.append(gray.copy())
                if len(self.frame_history) > self.max_history:
                    self.frame_history.pop(0)
            
            # Initialize metrics dictionary
            metrics = {
                'frame_num': frame_num,
                'timestamp': timestamp,
                'timestamp_diff': 0,
                'expected_interval': metadata['expected_interval'],
                'flow_magnitude': 0,
                'ssim_score': 0,
                'hist_diff': 0,
                'laplacian_var': 0,
                'edge_diff': 0
            }
            
            # Skip first frame (no previous frame to compare)
            if prev_gray is not None:
                # Compute all metrics
                metrics['timestamp_diff'] = timestamp - prev_timestamp
                metrics['flow_magnitude'], flow_field = self.compute_optical_flow(prev_gray_resized, gray_resized)
                metrics['ssim_score'] = self.compute_ssim(prev_gray, gray)
                metrics['hist_diff'] = self.compute_histogram_difference(prev_gray, gray)
                metrics['laplacian_var'], metrics['edge_diff'] = self.detect_ghosting_artifacts(gray)
                
                # Store for auto-tuning
                self.flow_history.append(metrics['flow_magnitude'])
                self.ssim_history.append(metrics['ssim_score'])
                self.hist_history.append(metrics['hist_diff'])
                self.laplacian_history.append(metrics['laplacian_var'])  # ← Track for adaptive blur threshold
                
                # Auto-tune thresholds periodically
                if self.auto_tune and frame_num % 100 == 0:
                    self.auto_tune_thresholds()
                
                # Classify frame (pass frames and flow for reversal detection)
                classification, confidence = self.classify_frame(
                    metrics, 
                    curr_frame=gray, 
                    prev_frame=prev_gray,
                    flow=flow_field
                )
            else:
                classification, confidence = 'Normal', 1.0
            
            # Store results
            result = {
                **metrics,
                'classification': classification,
                'confidence': confidence
            }
            results.append(result)
            
            # Annotate frame
            annotated = self.annotate_frame(frame, frame_num, classification, confidence, metrics)
            out.write(annotated)
            
            # Show progress
            if show_progress:
                cv2.imshow('Temporal Error Detection', cv2.resize(annotated, (960, 540)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Processing interrupted by user")
                    break
            
            # Update previous frame
            prev_gray = gray
            prev_gray_resized = gray_resized
            prev_timestamp = timestamp
            
            # Log progress
            if frame_num % 100 == 0:
                logger.info(f"Processed {frame_num}/{metadata['frame_count']} frames")
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Save results to CSV
        self.save_results_csv(results, csv_path)
        
        logger.info(f"Processing complete! Output saved to {output_path}")
        logger.info(f"Results saved to {csv_path}")
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def save_results_csv(self, results: List[Dict], csv_path: str):
        """
        Save detection results to CSV file.
        
        Args:
            results: List of result dictionaries
            csv_path: Output CSV path
        """
        import csv
        
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['frame_num', 'timestamp', 'flow_magnitude', 'ssim_score', 
                         'hist_diff', 'laplacian_var', 'edge_diff', 
                         'classification', 'confidence']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                row = {k: result[k] for k in fieldnames}
                writer.writerow(row)
        
        logger.info(f"CSV results saved to {csv_path}")
    
    def print_summary(self, results: List[Dict]):
        """
        Print detection summary statistics.
        
        Args:
            results: List of result dictionaries
        """
        total_frames = len(results)
        normal_frames = sum(1 for r in results if r['classification'] == 'Normal')
        drop_frames = sum(1 for r in results if r['classification'] == 'Frame Drop')
        merge_frames = sum(1 for r in results if r['classification'] == 'Frame Merge')
        reversal_frames = sum(1 for r in results if r['classification'] == 'Frame Reversal')
        
        logger.info("\n" + "="*60)
        logger.info("DETECTION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Frames:       {total_frames}")
        logger.info(f"Normal Frames:      {normal_frames} ({100*normal_frames/total_frames:.1f}%)")
        logger.info(f"Frame Drops:        {drop_frames} ({100*drop_frames/total_frames:.1f}%)")
        logger.info(f"Frame Merges:       {merge_frames} ({100*merge_frames/total_frames:.1f}%)")
        logger.info(f"Frame Reversals:    {reversal_frames} ({100*reversal_frames/total_frames:.1f}%) 🔄 Boomerang")
        logger.info("="*60 + "\n")


if __name__ == "__main__":
    # Example usage
    detector = TemporalErrorDetector(auto_tune=True)
    
    input_video = "input_video.mp4"
    output_video = "output_annotated.mp4"
    output_csv = "detection_results.csv"
    
    results = detector.process_video(input_video, output_video, output_csv, show_progress=True)
