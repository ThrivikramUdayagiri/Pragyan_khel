"""
Batch Video Processor
====================
Process multiple videos simultaneously.
"""

import cv2
import pandas as pd
from pathlib import Path
from typing import List, Dict
import concurrent.futures
import logging
from video_error_detector import TemporalErrorDetector
import time

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Process multiple videos in batch"""
    
    def __init__(self,
                 flow_threshold: float = 30.0,
                 ssim_threshold: float = 0.85,
                 hist_threshold: float = 0.3,
                 timestamp_tolerance: float = 1.5,
                 resize_width: int = 640,
                 auto_tune: bool = True):
        """Initialize batch processor with detection parameters"""
        self.flow_threshold = flow_threshold
        self.ssim_threshold = ssim_threshold
        self.hist_threshold = hist_threshold
        self.timestamp_tolerance = timestamp_tolerance
        self.resize_width = resize_width
        self.auto_tune = auto_tune
    
    def process_videos(self,
                      video_paths: List[str],
                      output_dir: str,
                      parallel: bool = True,
                      max_workers: int = 4) -> Dict:
        """
        Process multiple videos.
        
        Args:
            video_paths: List of video file paths
            output_dir: Output directory for results
            parallel: Process in parallel (True) or sequential (False)
            max_workers: Maximum parallel workers
            
        Returns:
            Dictionary with processing results and statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing {len(video_paths)} videos")
        start_time = time.time()
        
        if parallel:
            results = self._process_parallel(video_paths, output_path, max_workers)
        else:
            results = self._process_sequential(video_paths, output_path)
        
        total_time = time.time() - start_time
        
        # Generate batch summary
        summary = self._generate_batch_summary(results, total_time)
        
        # Save batch report
        self._save_batch_report(results, summary, output_path / "batch_report.csv")
        
        logger.info(f"Batch processing complete in {total_time:.1f}s")
        return summary
    
    def _process_parallel(self,
                         video_paths: List[str],
                         output_path: Path,
                         max_workers: int) -> List[Dict]:
        """Process videos in parallel"""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_single_video, path, output_path): path
                for path in video_paths
            }
            
            for future in concurrent.futures.as_completed(futures):
                video_path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed: {Path(video_path).name}")
                except Exception as e:
                    logger.error(f"Error processing {video_path}: {e}")
                    results.append({
                        'video_path': video_path,
                        'status': 'error',
                        'error': str(e)
                    })
        
        return results
    
    def _process_sequential(self,
                           video_paths: List[str],
                           output_path: Path) -> List[Dict]:
        """Process videos sequentially"""
        results = []
        
        for video_path in video_paths:
            try:
                result = self._process_single_video(video_path, output_path)
                results.append(result)
                logger.info(f"Completed: {Path(video_path).name}")
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                results.append({
                    'video_path': video_path,
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def _process_single_video(self,
                             video_path: str,
                             output_path: Path) -> Dict:
        """Process a single video"""
        video_name = Path(video_path).stem
        
        # Create output paths
        output_video = output_path / f"{video_name}_annotated.mp4"
        output_csv = output_path / f"{video_name}_results.csv"
        
        # Initialize detector
        detector = TemporalErrorDetector(
            flow_threshold=self.flow_threshold,
            ssim_threshold=self.ssim_threshold,
            hist_threshold=self.hist_threshold,
            timestamp_tolerance=self.timestamp_tolerance,
            resize_width=self.resize_width,
            auto_tune=self.auto_tune
        )
        
        # Process video
        start_time = time.time()
        results = detector.process_video(
            input_path=video_path,
            output_path=str(output_video),
            csv_path=str(output_csv),
            show_progress=False
        )
        processing_time = time.time() - start_time
        
        # Calculate statistics
        total = len(results)
        normal = sum(1 for r in results if r['classification'] == 'Normal')
        drops = sum(1 for r in results if r['classification'] == 'Frame Drop')
        merges = sum(1 for r in results if r['classification'] == 'Frame Merge')
        quality_score = 100 * normal / total if total > 0 else 0
        
        return {
            'video_path': video_path,
            'video_name': video_name,
            'status': 'success',
            'total_frames': total,
            'normal_frames': normal,
            'frame_drops': drops,
            'frame_merges': merges,
            'quality_score': quality_score,
            'processing_time': processing_time,
            'output_video': str(output_video),
            'output_csv': str(output_csv)
        }
    
    def _generate_batch_summary(self,
                               results: List[Dict],
                               total_time: float) -> Dict:
        """Generate summary statistics for batch processing"""
        successful = [r for r in results if r.get('status') == 'success']
        failed = [r for r in results if r.get('status') == 'error']
        
        if not successful:
            return {
                'total_videos': len(results),
                'successful': 0,
                'failed': len(failed),
                'total_time': total_time
            }
        
        total_frames = sum(r['total_frames'] for r in successful)
        total_drops = sum(r['frame_drops'] for r in successful)
        total_merges = sum(r['frame_merges'] for r in successful)
        avg_quality = sum(r['quality_score'] for r in successful) / len(successful)
        
        return {
            'total_videos': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'total_frames': total_frames,
            'total_drops': total_drops,
            'total_merges': total_merges,
            'average_quality_score': avg_quality,
            'total_time': total_time,
            'videos_per_minute': len(results) / (total_time / 60) if total_time > 0 else 0,
            'results': results
        }
    
    def _save_batch_report(self,
                          results: List[Dict],
                          summary: Dict,
                          output_path: Path):
        """Save batch processing report to CSV"""
        df = pd.DataFrame(results)
        
        # Add summary as comment in CSV
        with open(output_path, 'w') as f:
            f.write(f"# Batch Processing Report\n")
            f.write(f"# Total Videos: {summary['total_videos']}\n")
            f.write(f"# Successful: {summary['successful']}\n")
            f.write(f"# Failed: {summary['failed']}\n")
            if 'average_quality_score' in summary:
                f.write(f"# Average Quality Score: {summary['average_quality_score']:.1f}%\n")
            f.write(f"# Total Processing Time: {summary['total_time']:.1f}s\n")
            f.write("#\n")
            
            # Write DataFrame
            df.to_csv(f, index=False)
        
        logger.info(f"Batch report saved: {output_path}")
    
    def compare_videos(self,
                      video_paths: List[str],
                      output_dir: str) -> pd.DataFrame:
        """
        Compare multiple videos and generate comparison report.
        
        Args:
            video_paths: List of video paths to compare
            output_dir: Output directory
            
        Returns:
            DataFrame with comparison metrics
        """
        logger.info(f"Comparing {len(video_paths)} videos")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process all videos
        results = self._process_sequential(video_paths, output_path)
        
        # Create comparison DataFrame
        comparison_data = []
        for r in results:
            if r.get('status') == 'success':
                comparison_data.append({
                    'Video Name': r['video_name'],
                    'Total Frames': r['total_frames'],
                    'Frame Drops': r['frame_drops'],
                    'Frame Merges': r['frame_merges'],
                    'Quality Score': f"{r['quality_score']:.1f}%",
                    'Processing Time': f"{r['processing_time']:.1f}s"
                })
        
        df = pd.DataFrame(comparison_data)
        
        # Save comparison report
        comparison_path = output_path / "video_comparison.csv"
        df.to_csv(comparison_path, index=False)
        logger.info(f"Comparison report saved: {comparison_path}")
        
        return df
