"""
Command-Line Interface for Video Temporal Error Detection System
=================================================================
Provides flexible CLI for video analysis with customizable parameters.
"""

import argparse
import sys
import os
from video_error_detector import TemporalErrorDetector
import logging

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Video Temporal Error Detection System - Detect frame drops and merges',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage
  python cli.py --input video.mp4 --output annotated.mp4

  # Custom thresholds
  python cli.py -i video.mp4 -o out.mp4 --flow-threshold 40 --ssim-threshold 0.9

  # Save CSV results without video output
  python cli.py -i video.mp4 --csv results.csv --no-video

  # Disable auto-tuning
  python cli.py -i video.mp4 -o out.mp4 --no-auto-tune

  # Silent mode (no progress window)
  python cli.py -i video.mp4 -o out.mp4 --silent
        '''
    )
    
    # Required arguments
    parser.add_argument('-i', '--input',
                       required=True,
                       help='Input video file path')
    
    # Output options
    parser.add_argument('-o', '--output',
                       default='output_annotated.mp4',
                       help='Output annotated video path (default: output_annotated.mp4)')
    
    parser.add_argument('--csv',
                       default='detection_results.csv',
                       help='Output CSV results path (default: detection_results.csv)')
    
    parser.add_argument('--no-video',
                       action='store_true',
                       help='Skip video output, only generate CSV')
    
    # Detection thresholds
    parser.add_argument('--flow-threshold',
                       type=float,
                       default=30.0,
                       help='Optical flow magnitude threshold (default: 30.0)')
    
    parser.add_argument('--ssim-threshold',
                       type=float,
                       default=0.85,
                       help='SSIM threshold for frame merge detection (default: 0.85)')
    
    parser.add_argument('--hist-threshold',
                       type=float,
                       default=0.3,
                       help='Histogram difference threshold (default: 0.3)')
    
    parser.add_argument('--timestamp-tolerance',
                       type=float,
                       default=1.5,
                       help='Timestamp irregularity tolerance multiplier (default: 1.5)')
    
    # Processing options
    parser.add_argument('--resize-width',
                       type=int,
                       default=640,
                       help='Frame width for optical flow computation (default: 640)')
    
    parser.add_argument('--no-auto-tune',
                       action='store_true',
                       help='Disable automatic threshold tuning')
    
    parser.add_argument('--silent',
                       action='store_true',
                       help='Disable progress window display')
    
    parser.add_argument('--log-level',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level (default: INFO)')
    
    return parser.parse_args()


def validate_inputs(args):
    """Validate input arguments."""
    # Check input file exists
    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Check video format
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    ext = os.path.splitext(args.input)[1].lower()
    if ext not in valid_extensions:
        print(f"Warning: Video format '{ext}' may not be supported")
    
    # Validate threshold ranges
    if not (0 < args.flow_threshold < 1000):
        print("Error: flow-threshold must be between 0 and 1000")
        sys.exit(1)
    
    if not (0 < args.ssim_threshold < 1):
        print("Error: ssim-threshold must be between 0 and 1")
        sys.exit(1)
    
    if not (0 < args.hist_threshold < 10):
        print("Error: hist-threshold must be between 0 and 10")
        sys.exit(1)
    
    if not (1.0 <= args.timestamp_tolerance <= 5.0):
        print("Error: timestamp-tolerance must be between 1.0 and 5.0")
        sys.exit(1)
    
    return True


def main():
    """Main CLI entry point."""
    args = parse_arguments()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Validate inputs
    validate_inputs(args)
    
    # Print configuration
    print("\n" + "="*70)
    print("VIDEO TEMPORAL ERROR DETECTION SYSTEM")
    print("="*70)
    print(f"Input Video:         {args.input}")
    print(f"Output Video:        {args.output if not args.no_video else 'Disabled'}")
    print(f"CSV Results:         {args.csv}")
    print(f"\nDetection Thresholds:")
    print(f"  Flow Threshold:    {args.flow_threshold}")
    print(f"  SSIM Threshold:    {args.ssim_threshold}")
    print(f"  Histogram Thresh:  {args.hist_threshold}")
    print(f"  Timestamp Tol:     {args.timestamp_tolerance}x")
    print(f"\nProcessing Options:")
    print(f"  Resize Width:      {args.resize_width}px")
    print(f"  Auto-tune:         {'Enabled' if not args.no_auto_tune else 'Disabled'}")
    print(f"  Show Progress:     {'Yes' if not args.silent else 'No'}")
    print("="*70 + "\n")
    
    # Initialize detector
    detector = TemporalErrorDetector(
        flow_threshold=args.flow_threshold,
        ssim_threshold=args.ssim_threshold,
        hist_threshold=args.hist_threshold,
        timestamp_tolerance=args.timestamp_tolerance,
        resize_width=args.resize_width,
        auto_tune=not args.no_auto_tune
    )
    
    # Process video
    try:
        output_video = None if args.no_video else args.output
        results = detector.process_video(
            input_path=args.input,
            output_path=output_video if output_video else "temp_output.mp4",
            csv_path=args.csv,
            show_progress=not args.silent
        )
        
        # Clean up temp file if video output disabled
        if args.no_video and os.path.exists("temp_output.mp4"):
            os.remove("temp_output.mp4")
        
        print("\n✓ Processing completed successfully!")
        print(f"✓ CSV results saved to: {args.csv}")
        if not args.no_video:
            print(f"✓ Annotated video saved to: {args.output}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        return 130
    
    except Exception as e:
        print(f"\n✗ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
