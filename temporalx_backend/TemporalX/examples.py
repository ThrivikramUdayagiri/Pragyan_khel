"""
Example Usage and Demo Script for TemporalX
============================================
Demonstrates various usage patterns and features.
"""

import cv2
import numpy as np
from video_error_detector import TemporalErrorDetector
from visualizer import DetectionVisualizer
import os


def create_test_video_with_errors(output_path: str = "test_video.mp4", 
                                  fps: int = 30, 
                                  duration: int = 10):
    """
    Create a synthetic test video with intentional frame drops and merges.
    
    Args:
        output_path: Output video path
        fps: Frames per second
        duration: Video duration in seconds
    """
    print("Creating synthetic test video with temporal errors...")
    
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = fps * duration
    
    for i in range(total_frames):
        # Create frame with moving elements
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background gradient
        for y in range(height):
            color_val = int(255 * y / height)
            frame[y, :] = [color_val // 3, color_val // 2, color_val]
        
        # Moving circle
        center_x = int(width * (i / total_frames))
        center_y = height // 2 + int(100 * np.sin(i * 0.1))
        cv2.circle(frame, (center_x, center_y), 50, (0, 255, 0), -1)
        
        # Moving rectangle
        rect_x = int(width * ((total_frames - i) / total_frames))
        rect_y = height // 3
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 100, rect_y + 60), (255, 0, 0), -1)
        
        # Text with frame number
        cv2.putText(frame, f"Frame: {i}", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Simulate frame drop (skip frames at intervals)
        if i % 120 == 0 and i > 0:
            print(f"  Simulating frame drop at frame {i}")
            continue  # Skip this frame
        
        # Simulate frame merge (blend consecutive frames)
        if i % 200 == 0 and i > 0 and i < total_frames - 1:
            print(f"  Simulating frame merge at frame {i}")
            # Create next frame
            next_frame = frame.copy()
            next_center_x = int(width * ((i + 1) / total_frames))
            next_center_y = height // 2 + int(100 * np.sin((i + 1) * 0.1))
            cv2.circle(next_frame, (next_center_x, next_center_y), 50, (0, 255, 0), -1)
            
            # Blend frames
            frame = cv2.addWeighted(frame, 0.5, next_frame, 0.5, 0)
        
        out.write(frame)
    
    out.release()
    print(f"✓ Test video created: {output_path}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration}s at {fps} FPS")


def example_1_basic_detection():
    """Example 1: Basic video detection."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Frame Drop/Merge Detection")
    print("="*70 + "\n")
    
    # Create test video
    input_video = "test_video.mp4"
    if not os.path.exists(input_video):
        create_test_video_with_errors(input_video, fps=30, duration=10)
    
    # Initialize detector with default settings
    detector = TemporalErrorDetector()
    
    # Process video
    results = detector.process_video(
        input_path=input_video,
        output_path="output_basic.mp4",
        csv_path="results_basic.csv",
        show_progress=False
    )
    
    print(f"\n✓ Example 1 complete!")
    print(f"  Output: output_basic.mp4")
    print(f"  CSV: results_basic.csv")


def example_2_custom_thresholds():
    """Example 2: Detection with custom thresholds."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Detection with Custom Thresholds")
    print("="*70 + "\n")
    
    input_video = "test_video.mp4"
    if not os.path.exists(input_video):
        create_test_video_with_errors(input_video, fps=30, duration=10)
    
    # Initialize with custom thresholds
    detector = TemporalErrorDetector(
        flow_threshold=25.0,      # More sensitive to motion changes
        ssim_threshold=0.90,      # Stricter similarity threshold
        hist_threshold=0.25,      # More sensitive to scene changes
        timestamp_tolerance=1.3,  # Tighter timestamp tolerance
        auto_tune=False           # Disable auto-tuning
    )
    
    results = detector.process_video(
        input_path=input_video,
        output_path="output_custom.mp4",
        csv_path="results_custom.csv",
        show_progress=False
    )
    
    print(f"\n✓ Example 2 complete!")
    print(f"  Output: output_custom.mp4")
    print(f"  CSV: results_custom.csv")


def example_3_with_visualization():
    """Example 3: Full pipeline with visualizations."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Complete Analysis with Visualizations")
    print("="*70 + "\n")
    
    input_video = "test_video.mp4"
    if not os.path.exists(input_video):
        create_test_video_with_errors(input_video, fps=30, duration=10)
    
    # Detection with auto-tuning
    detector = TemporalErrorDetector(auto_tune=True)
    
    results = detector.process_video(
        input_path=input_video,
        output_path="output_complete.mp4",
        csv_path="results_complete.csv",
        show_progress=False
    )
    
    # Generate visualizations
    visualizer = DetectionVisualizer(results=results)
    visualizer.generate_all_visualizations(output_dir="analysis_output", dpi=150)
    visualizer.print_detailed_report()
    
    print(f"\n✓ Example 3 complete!")
    print(f"  Output: output_complete.mp4")
    print(f"  CSV: results_complete.csv")
    print(f"  Visualizations: analysis_output/")


def example_4_programmatic_analysis():
    """Example 4: Programmatic analysis and filtering."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Programmatic Analysis")
    print("="*70 + "\n")
    
    import pandas as pd
    
    # Create test video if needed
    input_video = "test_video.mp4"
    if not os.path.exists(input_video):
        create_test_video_with_errors(input_video, fps=30, duration=10)
    
    # Run detection
    detector = TemporalErrorDetector(auto_tune=True)
    results = detector.process_video(
        input_path=input_video,
        output_path="temp_output.mp4",
        csv_path="results_analysis.csv",
        show_progress=False
    )
    
    # Load results as DataFrame
    df = pd.DataFrame(results)
    
    # Analyze results programmatically
    print("\nProgrammatic Analysis:")
    print("-" * 70)
    
    # Find all frame drops
    drops = df[df['classification'] == 'Frame Drop']
    print(f"\nFrame Drops Detected: {len(drops)}")
    if len(drops) > 0:
        print(f"  Frames: {drops['frame_num'].tolist()}")
        print(f"  Avg Confidence: {drops['confidence'].mean():.3f}")
        print(f"  Avg Flow Magnitude: {drops['flow_magnitude'].mean():.2f}")
    
    # Find all frame merges
    merges = df[df['classification'] == 'Frame Merge']
    print(f"\nFrame Merges Detected: {len(merges)}")
    if len(merges) > 0:
        print(f"  Frames: {merges['frame_num'].tolist()}")
        print(f"  Avg Confidence: {merges['confidence'].mean():.3f}")
        print(f"  Avg SSIM: {merges['ssim_score'].mean():.3f}")
    
    # Find high-confidence anomalies
    high_conf_anomalies = df[(df['classification'] != 'Normal') & 
                             (df['confidence'] > 0.8)]
    print(f"\nHigh-Confidence Anomalies (>0.8): {len(high_conf_anomalies)}")
    
    # Calculate video quality score
    normal_pct = len(df[df['classification'] == 'Normal']) / len(df) * 100
    quality_score = normal_pct
    print(f"\nVideo Quality Score: {quality_score:.1f}%")
    
    # Clean up temp file
    if os.path.exists("temp_output.mp4"):
        os.remove("temp_output.mp4")
    
    print(f"\n✓ Example 4 complete!")


def example_5_batch_processing():
    """Example 5: Batch processing multiple videos."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Batch Processing Multiple Videos")
    print("="*70 + "\n")
    
    # Create multiple test videos
    test_videos = []
    for i in range(3):
        video_name = f"test_batch_{i+1}.mp4"
        if not os.path.exists(video_name):
            create_test_video_with_errors(video_name, fps=30, duration=5)
        test_videos.append(video_name)
    
    # Process all videos
    detector = TemporalErrorDetector(auto_tune=True)
    
    batch_results = {}
    for video in test_videos:
        print(f"\nProcessing: {video}")
        print("-" * 70)
        
        output_video = video.replace(".mp4", "_processed.mp4")
        output_csv = video.replace(".mp4", "_results.csv")
        
        results = detector.process_video(
            input_path=video,
            output_path=output_video,
            csv_path=output_csv,
            show_progress=False
        )
        
        batch_results[video] = results
    
    # Summary
    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY")
    print("="*70)
    for video, results in batch_results.items():
        drops = sum(1 for r in results if r['classification'] == 'Frame Drop')
        merges = sum(1 for r in results if r['classification'] == 'Frame Merge')
        print(f"\n{video}:")
        print(f"  Total Frames: {len(results)}")
        print(f"  Frame Drops:  {drops}")
        print(f"  Frame Merges: {merges}")
    
    print(f"\n✓ Example 5 complete!")


def run_all_examples():
    """Run all example demonstrations."""
    print("\n" + "="*70)
    print("TEMPORALX - EXAMPLE DEMONSTRATIONS")
    print("="*70)
    
    examples = [
        ("Basic Detection", example_1_basic_detection),
        ("Custom Thresholds", example_2_custom_thresholds),
        ("Full Analysis with Visualizations", example_3_with_visualization),
        ("Programmatic Analysis", example_4_programmatic_analysis),
        ("Batch Processing", example_5_batch_processing)
    ]
    
    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print(f"  {len(examples) + 1}. Run All Examples")
    print(f"  0. Exit")
    
    choice = input("\nSelect example (0-6): ").strip()
    
    try:
        choice = int(choice)
        if choice == 0:
            print("Exiting.")
            return
        elif choice == len(examples) + 1:
            # Run all
            for name, func in examples:
                try:
                    func()
                except Exception as e:
                    print(f"\n✗ Error in {name}: {str(e)}")
        elif 1 <= choice <= len(examples):
            # Run selected
            name, func = examples[choice - 1]
            func()
        else:
            print("Invalid choice.")
    except ValueError:
        print("Invalid input.")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")


if __name__ == "__main__":
    run_all_examples()
