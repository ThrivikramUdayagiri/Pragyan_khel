"""
Real-Time Webcam Temporal Error Detection
==========================================
Live detection system for webcam/screen capture streams.
"""

import cv2
import numpy as np
from video_error_detector import TemporalErrorDetector
import time
from collections import deque
import argparse


class RealtimeDetector:
    """
    Real-time video stream analyzer for temporal errors.
    Optimized for low-latency webcam detection.
    """
    
    def __init__(self, 
                 detector: TemporalErrorDetector,
                 buffer_size: int = 30,
                 target_fps: float = 30.0):
        """
        Initialize real-time detector.
        
        Args:
            detector: TemporalErrorDetector instance
            buffer_size: Size of rolling statistics buffer
            target_fps: Expected FPS for timestamp analysis
        """
        self.detector = detector
        self.buffer_size = buffer_size
        self.target_fps = target_fps
        self.expected_interval = 1.0 / target_fps
        
        # Rolling buffers for statistics
        self.flow_buffer = deque(maxlen=buffer_size)
        self.ssim_buffer = deque(maxlen=buffer_size)
        self.hist_buffer = deque(maxlen=buffer_size)
        
        # Tracking variables
        self.frame_count = 0
        self.prev_gray = None
        self.prev_gray_resized = None
        self.prev_timestamp = None
        
        # Statistics
        self.total_drops = 0
        self.total_merges = 0
        self.total_frames = 0
        
        print("Real-time detector initialized")
        print(f"Target FPS: {target_fps}")
        print(f"Buffer size: {buffer_size}")
    
    def process_frame(self, frame: np.ndarray, timestamp: float) -> dict:
        """
        Process a single frame in real-time.
        
        Args:
            frame: Input BGR frame
            timestamp: Current timestamp in seconds
            
        Returns:
            Dictionary with detection results
        """
        self.frame_count += 1
        self.total_frames += 1
        
        # Preprocess
        gray, gray_resized = self.detector.preprocess_frame(frame)
        
        # Initialize result
        result = {
            'frame_num': self.frame_count,
            'timestamp': timestamp,
            'classification': 'Normal',
            'confidence': 1.0,
            'flow_magnitude': 0,
            'ssim_score': 0,
            'hist_diff': 0,
            'laplacian_var': 0,
            'edge_diff': 0
        }
        
        # Skip first frame
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_gray_resized = gray_resized
            self.prev_timestamp = timestamp
            return result
        
        # Compute metrics
        timestamp_diff = timestamp - self.prev_timestamp
        flow_magnitude, _ = self.detector.compute_optical_flow(self.prev_gray_resized, gray_resized)
        ssim_score = self.detector.compute_ssim(self.prev_gray, gray)
        hist_diff = self.detector.compute_histogram_difference(self.prev_gray, gray)
        laplacian_var, edge_diff = self.detector.detect_ghosting_artifacts(gray)
        
        # Update buffers
        self.flow_buffer.append(flow_magnitude)
        self.ssim_buffer.append(ssim_score)
        self.hist_buffer.append(hist_diff)
        
        # Update result
        result.update({
            'flow_magnitude': flow_magnitude,
            'ssim_score': ssim_score,
            'hist_diff': hist_diff,
            'laplacian_var': laplacian_var,
            'edge_diff': edge_diff,
            'timestamp_diff': timestamp_diff,
            'expected_interval': self.expected_interval
        })
        
        # Classify
        classification, confidence = self.detector.classify_frame(result)
        result['classification'] = classification
        result['confidence'] = confidence
        
        # Update statistics
        if classification == 'Frame Drop':
            self.total_drops += 1
        elif classification == 'Frame Merge':
            self.total_merges += 1
        
        # Update previous frame
        self.prev_gray = gray
        self.prev_gray_resized = gray_resized
        self.prev_timestamp = timestamp
        
        return result
    
    def draw_realtime_overlay(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """
        Draw real-time overlay with detection info.
        
        Args:
            frame: Input frame
            result: Detection result dictionary
            
        Returns:
            Frame with overlay
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Color mapping
        colors = {
            'Normal': (0, 255, 0),
            'Frame Drop': (0, 0, 255),
            'Frame Merge': (0, 255, 255)
        }
        color = colors[result['classification']]
        
        # Status banner
        banner_height = 100
        cv2.rectangle(overlay, (0, 0), (w, banner_height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, 0), (w, banner_height), color, 4)
        
        # Main status
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, f"STATUS: {result['classification']}", 
                   (10, 35), font, 1.0, color, 2)
        cv2.putText(overlay, f"Confidence: {result['confidence']:.2f}", 
                   (10, 70), font, 0.7, (255, 255, 255), 2)
        
        # Frame counter
        cv2.putText(overlay, f"Frame: {result['frame_num']}", 
                   (w - 200, 35), font, 0.7, (255, 255, 255), 2)
        
        # Metrics sidebar
        sidebar_x = 10
        sidebar_y = h - 220
        sidebar_w = 300
        sidebar_h = 210
        
        cv2.rectangle(overlay, (sidebar_x, sidebar_y), 
                     (sidebar_x + sidebar_w, sidebar_y + sidebar_h), (0, 0, 0), -1)
        cv2.rectangle(overlay, (sidebar_x, sidebar_y), 
                     (sidebar_x + sidebar_w, sidebar_y + sidebar_h), (100, 100, 100), 2)
        
        # Metrics text
        y = sidebar_y + 25
        cv2.putText(overlay, "METRICS", (sidebar_x + 10, y), 
                   font, 0.6, (255, 255, 255), 2)
        y += 30
        cv2.putText(overlay, f"Flow: {result['flow_magnitude']:.2f}", 
                   (sidebar_x + 10, y), font, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(overlay, f"SSIM: {result['ssim_score']:.3f}", 
                   (sidebar_x + 10, y), font, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(overlay, f"Hist: {result['hist_diff']:.3f}", 
                   (sidebar_x + 10, y), font, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(overlay, f"Blur: {result['laplacian_var']:.1f}", 
                   (sidebar_x + 10, y), font, 0.5, (255, 255, 255), 1)
        y += 30
        cv2.putText(overlay, f"Drops: {self.total_drops}  Merges: {self.total_merges}", 
                   (sidebar_x + 10, y), font, 0.5, (255, 200, 0), 1)
        
        # FPS display
        if len(self.flow_buffer) > 0:
            avg_flow = np.mean(list(self.flow_buffer))
            cv2.putText(overlay, f"Avg Flow: {avg_flow:.2f}", 
                       (w - 250, h - 20), font, 0.5, (200, 200, 200), 1)
        
        return overlay
    
    def run_webcam(self, camera_id: int = 0, display_width: int = 1280, display_height: int = 720):
        """
        Run real-time detection on webcam stream.
        
        Args:
            camera_id: Camera device ID (0 for default)
            display_width: Display window width
            display_height: Display window height
        """
        print("\n" + "="*70)
        print("REAL-TIME WEBCAM TEMPORAL ERROR DETECTION")
        print("="*70)
        print(f"Camera ID: {camera_id}")
        print(f"Display: {display_width}x{display_height}")
        print("\nControls:")
        print("  'q' - Quit")
        print("  'r' - Reset statistics")
        print("  's' - Save current frame")
        print("="*70 + "\n")
        
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        print("✓ Camera opened successfully")
        print("Starting detection...\n")
        
        # Create window
        cv2.namedWindow('Real-Time Temporal Error Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Real-Time Temporal Error Detection', display_width, display_height)
        
        fps_counter = deque(maxlen=30)
        start_time = time.time()
        
        try:
            while True:
                frame_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                # Get timestamp
                timestamp = time.time() - start_time
                
                # Process frame
                result = self.process_frame(frame, timestamp)
                
                # Draw overlay
                display_frame = self.draw_realtime_overlay(frame, result)
                
                # Calculate FPS
                frame_time = time.time() - frame_start
                fps_counter.append(1.0 / frame_time if frame_time > 0 else 0)
                current_fps = np.mean(list(fps_counter))
                
                # FPS overlay
                cv2.putText(display_frame, f"FPS: {current_fps:.1f}", 
                           (display_frame.shape[1] - 120, display_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display
                cv2.imshow('Real-Time Temporal Error Detection', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('r'):
                    print("\nResetting statistics...")
                    self.total_drops = 0
                    self.total_merges = 0
                    self.frame_count = 0
                    self.total_frames = 0
                    self.flow_buffer.clear()
                    self.ssim_buffer.clear()
                    self.hist_buffer.clear()
                    print("✓ Statistics reset")
                elif key == ord('s'):
                    filename = f"capture_frame_{self.frame_count}.png"
                    cv2.imwrite(filename, display_frame)
                    print(f"✓ Frame saved: {filename}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            print("\n" + "="*70)
            print("SESSION SUMMARY")
            print("="*70)
            print(f"Total Frames:       {self.total_frames}")
            print(f"Frame Drops:        {self.total_drops} ({100*self.total_drops/self.total_frames:.2f}%)")
            print(f"Frame Merges:       {self.total_merges} ({100*self.total_merges/self.total_frames:.2f}%)")
            print(f"Session Duration:   {time.time() - start_time:.1f} seconds")
            print("="*70 + "\n")
    
    def run_screen_capture(self, region: tuple = None):
        """
        Run real-time detection on screen capture.
        
        Args:
            region: (x, y, width, height) tuple for capture region, None for full screen
        """
        try:
            import mss
        except ImportError:
            print("Error: mss library required for screen capture")
            print("Install with: pip install mss")
            return
        
        print("\n" + "="*70)
        print("REAL-TIME SCREEN CAPTURE TEMPORAL ERROR DETECTION")
        print("="*70)
        print("Controls: 'q' to quit")
        print("="*70 + "\n")
        
        with mss.mss() as sct:
            # Define capture region
            if region is None:
                monitor = sct.monitors[1]  # Primary monitor
            else:
                monitor = {"left": region[0], "top": region[1], 
                          "width": region[2], "height": region[3]}
            
            print(f"Capture region: {monitor}")
            print("Starting detection...\n")
            
            start_time = time.time()
            
            try:
                while True:
                    # Capture screen
                    img = sct.grab(monitor)
                    frame = np.array(img)[:, :, :3]  # Remove alpha channel
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Get timestamp
                    timestamp = time.time() - start_time
                    
                    # Process
                    result = self.process_frame(frame, timestamp)
                    
                    # Draw overlay
                    display_frame = self.draw_realtime_overlay(frame, result)
                    
                    # Display
                    cv2.imshow('Screen Capture Detection', cv2.resize(display_frame, (1280, 720)))
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
            
            finally:
                cv2.destroyAllWindows()


def main():
    """CLI for real-time detection."""
    parser = argparse.ArgumentParser(
        description='Real-time temporal error detection for webcam/screen capture'
    )
    
    parser.add_argument('--mode', choices=['webcam', 'screen'], default='webcam',
                       help='Detection mode (default: webcam)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='Target FPS (default: 30.0)')
    parser.add_argument('--width', type=int, default=1280,
                       help='Display width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                       help='Display height (default: 720)')
    parser.add_argument('--flow-threshold', type=float, default=30.0,
                       help='Flow threshold (default: 30.0)')
    parser.add_argument('--ssim-threshold', type=float, default=0.85,
                       help='SSIM threshold (default: 0.85)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = TemporalErrorDetector(
        flow_threshold=args.flow_threshold,
        ssim_threshold=args.ssim_threshold,
        auto_tune=True
    )
    
    # Initialize real-time detector
    realtime = RealtimeDetector(
        detector=detector,
        target_fps=args.fps
    )
    
    # Run detection
    if args.mode == 'webcam':
        realtime.run_webcam(
            camera_id=args.camera,
            display_width=args.width,
            display_height=args.height
        )
    else:
        realtime.run_screen_capture()


if __name__ == "__main__":
    main()
