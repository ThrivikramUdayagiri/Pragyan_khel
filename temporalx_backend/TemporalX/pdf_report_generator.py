"""
PDF Report Generator
====================
Generate professional PDF reports for video temporal error analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class PDFReportGenerator:
    """Generate comprehensive PDF reports"""
    
    def __init__(self, title: str = "Video Temporal Error Detection Report"):
        self.title = title
        self.colors = {
            'Normal': '#28a745',
            'Frame Drop': '#dc3545',
            'Frame Merge': '#ffc107'
        }
    
    def generate_report(self,
                       results: List[Dict],
                       video_info: Dict,
                       output_path: str,
                       processing_time: float = None,
                       include_charts: bool = True,
                       include_thumbnails: bool = False) -> str:
        """
        Generate comprehensive PDF report.
        
        Args:
            results: Detection results
            video_info: Video metadata
            output_path: Output PDF path
            processing_time: Analysis processing time
            include_charts: Include visualization charts
            include_thumbnails: Include frame thumbnails (if available)
            
        Returns:
            Path to generated PDF
        """
        logger.info(f"Generating PDF report: {output_path}")
        
        with PdfPages(output_path) as pdf:
            # Page 1: Title and Executive Summary
            self._create_title_page(pdf, video_info, processing_time)
            
            # Page 2: Statistics Overview
            self._create_statistics_page(pdf, results)
            
            # Page 3: Timeline Visualization
            if include_charts:
                self._create_timeline_page(pdf, results)
            
            # Page 4: Metrics Analysis
            if include_charts:
                self._create_metrics_page(pdf, results)
            
            # Page 5: Error Details Table
            self._create_error_table_page(pdf, results)
            
            # Page 6: Recommendations
            self._create_recommendations_page(pdf, results, video_info)
            
            # Add metadata
            d = pdf.infodict()
            d['Title'] = self.title
            d['Author'] = 'TemporalX - Video Error Detection System'
            d['Subject'] = 'Temporal Error Analysis Report'
            d['Keywords'] = 'Video, Temporal Errors, Frame Drop, Frame Merge'
            d['CreationDate'] = datetime.now()
        
        logger.info(f"PDF report generated successfully: {output_path}")
        return output_path
    
    def _create_title_page(self, pdf: PdfPages, video_info: Dict, processing_time: float):
        """Create title page with executive summary"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.9, self.title, ha='center', va='top',
                fontsize=24, fontweight='bold', transform=ax.transAxes)
        
        # Subtitle
        ax.text(0.5, 0.85, 'Automated Video Quality Analysis',
                ha='center', va='top', fontsize=14, color='gray',
                transform=ax.transAxes)
        
        # Date
        date_str = datetime.now().strftime("%B %d, %Y %H:%M:%S")
        ax.text(0.5, 0.80, f"Generated: {date_str}", ha='center', va='top',
                fontsize=10, style='italic', transform=ax.transAxes)
        
        # Divider line
        ax.plot([0.1, 0.9], [0.77, 0.77], 'k-', lw=2, transform=ax.transAxes)
        
        # Video Information
        y_pos = 0.70
        ax.text(0.5, y_pos, 'Video Information', ha='center', va='top',
                fontsize=16, fontweight='bold', transform=ax.transAxes)
        
        y_pos -= 0.08
        info_lines = [
            f"Resolution: {video_info.get('width', 'N/A')}x{video_info.get('height', 'N/A')}",
            f"Frame Rate: {video_info.get('fps', 'N/A'):.2f} fps",
            f"Total Frames: {video_info.get('frame_count', 'N/A')}",
            f"Duration: {video_info.get('duration', 'N/A'):.2f} seconds"
        ]
        
        if processing_time:
            info_lines.append(f"Processing Time: {processing_time:.2f} seconds")
            info_lines.append(f"Processing Speed: {video_info.get('frame_count', 0)/processing_time:.1f} fps")
        
        for line in info_lines:
            ax.text(0.2, y_pos, line, ha='left', va='top',
                   fontsize=11, transform=ax.transAxes)
            y_pos -= 0.05
        
        # Logo/Branding
        ax.text(0.5, 0.1, '🎥 TemporalX', ha='center', va='center',
                fontsize=32, transform=ax.transAxes)
        ax.text(0.5, 0.05, 'Advanced Video Temporal Error Detection',
                ha='center', va='center', fontsize=10, color='gray',
                transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_statistics_page(self, pdf: PdfPages, results: List[Dict]):
        """Create statistics overview page"""
        df = pd.DataFrame(results)
        
        fig = plt.figure(figsize=(8.5, 11))
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
        
        # Title
        fig.suptitle('Statistical Analysis', fontsize=20, fontweight='bold', y=0.98)
        
        # Overall Statistics
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        total = len(results)
        normal = sum(1 for r in results if r['classification'] == 'Normal')
        drops = sum(1 for r in results if r['classification'] == 'Frame Drop')
        merges = sum(1 for r in results if r['classification'] == 'Frame Merge')
        quality_score = 100 * normal / total if total > 0 else 0
        
        stats_text = f"""
        Total Frames Analyzed: {total}
        
        ✓ Normal Frames: {normal} ({100*normal/total:.1f}%)
        ✗ Frame Drops: {drops} ({100*drops/total:.1f}%)
        ⚠ Frame Merges: {merges} ({100*merges/total:.1f}%)
        
        Overall Quality Score: {quality_score:.1f}%
        """
        
        ax1.text(0.5, 0.5, stats_text, ha='center', va='center',
                fontsize=12, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5),
                transform=ax1.transAxes)
        
        # Pie Chart
        ax2 = fig.add_subplot(gs[1, 0])
        counts = [normal, drops, merges]
        labels = ['Normal', 'Frame Drop', 'Frame Merge']
        colors = [self.colors[label] for label in labels]
        
        ax2.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 10})
        ax2.set_title('Classification Distribution', fontsize=14, fontweight='bold')
        
        # Bar Chart
        ax3 = fig.add_subplot(gs[1, 1])
        bars = ax3.bar(labels, counts, color=colors)
        ax3.set_ylabel('Frame Count', fontsize=11)
        ax3.set_title('Error Frequency', fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        # Confidence Distribution
        ax4 = fig.add_subplot(gs[2, :])
        error_results = [r for r in results if r['classification'] != 'Normal']
        
        if error_results:
            confidences = [r['confidence'] for r in error_results]
            classifications = [r['classification'] for r in error_results]
            
            drop_conf = [c for c, cls in zip(confidences, classifications) if cls == 'Frame Drop']
            merge_conf = [c for c, cls in zip(confidences, classifications) if cls == 'Frame Merge']
            
            bins = np.linspace(0, 100, 21)
            ax4.hist([drop_conf, merge_conf], bins=bins, label=['Frame Drops', 'Frame Merges'],
                    color=[self.colors['Frame Drop'], self.colors['Frame Merge']], alpha=0.7)
            ax4.set_xlabel('Confidence Score (%)', fontsize=11)
            ax4.set_ylabel('Frequency', fontsize=11)
            ax4.set_title('Error Detection Confidence Distribution', fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No errors detected', ha='center', va='center',
                    fontsize=14, transform=ax4.transAxes)
            ax4.axis('off')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_timeline_page(self, pdf: PdfPages, results: List[Dict]):
        """Create timeline visualization page"""
        df = pd.DataFrame(results)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11), height_ratios=[1, 2])
        fig.suptitle('Temporal Analysis Timeline', fontsize=20, fontweight='bold')
        
        # Timeline bar
        classifications = df['classification'].values
        colors_list = [self.colors.get(c, '#gray') for c in classifications]
        
        ax1.barh(0, len(results), left=0, height=0.5, color=colors_list,
                edgecolor='none', linewidth=0)
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_xlim(0, len(results))
        ax1.set_xlabel('Frame Number', fontsize=11)
        ax1.set_title('Complete Video Timeline', fontsize=14, fontweight='bold')
        ax1.set_yticks([])
        
        # Legend
        patches = [mpatches.Patch(color=self.colors[label], label=label)
                  for label in ['Normal', 'Frame Drop', 'Frame Merge']]
        ax1.legend(handles=patches, loc='upper right')
        
        # Classification over time
        frame_nums = df['frame_num'].values
        class_numeric = df['classification'].map({
            'Normal': 0, 'Frame Drop': 1, 'Frame Merge': 2
        }).values
        
        for cls_val, cls_name in [(1, 'Frame Drop'), (2, 'Frame Merge')]:
            mask = class_numeric == cls_val
            if mask.any():
                ax2.scatter(frame_nums[mask], class_numeric[mask],
                           c=self.colors[cls_name], label=cls_name,
                           s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        ax2.axhline(y=0, color=self.colors['Normal'], linestyle='--',
                   linewidth=2, alpha=0.5, label='Normal')
        ax2.set_xlabel('Frame Number', fontsize=11)
        ax2.set_ylabel('Classification', fontsize=11)
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['Normal', 'Frame Drop', 'Frame Merge'])
        ax2.set_title('Error Detection Points', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.legend()
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_metrics_page(self, pdf: PdfPages, results: List[Dict]):
        """Create metrics analysis page"""
        df = pd.DataFrame(results)
        
        fig = plt.figure(figsize=(8.5, 11))
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
        fig.suptitle('Detection Metrics Analysis', fontsize=20, fontweight='bold', y=0.98)
        
        # Optical Flow
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df['frame_num'], df['flow_magnitude'], linewidth=0.5, alpha=0.7)
        ax1.set_ylabel('Flow Magnitude', fontsize=10)
        ax1.set_title('Optical Flow', fontsize=12, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # SSIM
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(df['frame_num'], df['ssim_score'], linewidth=0.5, alpha=0.7, color='green')
        ax2.set_ylabel('SSIM Score', fontsize=10)
        ax2.set_title('Structural Similarity', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # Histogram Difference
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(df['frame_num'], df['hist_diff'], linewidth=0.5, alpha=0.7, color='orange')
        ax3.set_ylabel('Histogram Diff', fontsize=10)
        ax3.set_title('Histogram Difference', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # Laplacian Variance
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(df['frame_num'], df['laplacian_var'], linewidth=0.5, alpha=0.7, color='purple')
        ax4.set_ylabel('Laplacian Variance', fontsize=10)
        ax4.set_title('Blur/Sharpness', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)
        
        # Timestamp Differences
        ax5 = fig.add_subplot(gs[2, :])
        ax5.plot(df['frame_num'], df['timestamp_diff'], linewidth=0.5, alpha=0.7, color='red')
        ax5.axhline(y=df['expected_interval'].iloc[0], color='gray',
                   linestyle='--', label='Expected Interval')
        ax5.set_xlabel('Frame Number', fontsize=11)
        ax5.set_ylabel('Time Diff (seconds)', fontsize=10)
        ax5.set_title('Frame Timing Analysis', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_error_table_page(self, pdf: PdfPages, results: List[Dict]):
        """Create detailed error table page"""
        df = pd.DataFrame(results)
        errors = df[df['classification'] != 'Normal'].copy()
        
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        fig.suptitle('Detailed Error Report', fontsize=20, fontweight='bold', y=0.98)
        
        if len(errors) == 0:
            ax.text(0.5, 0.5, '✓ No Errors Detected!\n\nThis video has excellent temporal quality.',
                   ha='center', va='center', fontsize=16,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
                   transform=ax.transAxes)
        else:
            # Show first 30 errors (to avoid overcrowding)
            display_errors = errors.head(30)
            
            table_data = []
            for _, row in display_errors.iterrows():
                table_data.append([
                    f"{int(row['frame_num'])}",
                    f"{row['timestamp']:.2f}s",
                    row['classification'],
                    f"{row['confidence']:.1f}%"
                ])
            
            table = ax.table(cellText=table_data,
                           colLabels=['Frame', 'Time', 'Error Type', 'Confidence'],
                           cellLoc='center',
                           loc='upper center',
                           bbox=[0.1, 0.1, 0.8, 0.85])
            
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            
            # Color code error types
            for i, row in enumerate(display_errors.iterrows(), start=1):
                error_type = row[1]['classification']
                color = self.colors.get(error_type, 'white')
                table[(i, 2)].set_facecolor(color)
                table[(i, 2)].set_alpha(0.3)
            
            # Style header
            for i in range(4):
                table[(0, i)].set_facecolor('#4472C4')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            if len(errors) > 30:
                ax.text(0.5, 0.02,
                       f'Showing first 30 of {len(errors)} total errors',
                       ha='center', va='bottom', fontsize=10, style='italic',
                       transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_recommendations_page(self, pdf: PdfPages, results: List[Dict],
                                    video_info: Dict):
        """Create recommendations page"""
        df = pd.DataFrame(results)
        
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        fig.suptitle('Analysis & Recommendations', fontsize=20, fontweight='bold', y=0.98)
        
        # Calculate statistics
        total = len(results)
        drops = sum(1 for r in results if r['classification'] == 'Frame Drop')
        merges = sum(1 for r in results if r['classification'] == 'Frame Merge')
        quality_score = 100 * (total - drops - merges) / total
        
        # Generate recommendations
        recommendations = []
        
        if quality_score >= 95:
            recommendations.append("✓ Excellent video quality! No significant temporal errors detected.")
        elif quality_score >= 85:
            recommendations.append("✓ Good video quality with minor temporal issues.")
        elif quality_score >= 70:
            recommendations.append("⚠ Fair video quality. Some temporal issues detected.")
        else:
            recommendations.append("✗ Poor video quality. Significant temporal issues detected.")
        
        if drops > 0:
            drop_rate = drops / total * 100
            recommendations.append(f"\n⚠ Frame Drops Detected ({drops} frames, {drop_rate:.1f}%)")
            recommendations.append("Possible causes:")
            recommendations.append("  • Recording hardware overload")
            recommendations.append("  • Insufficient storage write speed")
            recommendations.append("  • Network bandwidth issues (streaming)")
            recommendations.append("  • Codec processing limitations")
            recommendations.append("\nRecommended actions:")
            recommendations.append("  → Use faster storage (SSD recommended)")
            recommendations.append("  → Reduce recording resolution/bitrate")
            recommendations.append("  → Close background applications")
            recommendations.append("  → Check hardware temperature/throttling")
        
        if merges > 0:
            merge_rate = merges / total * 100
            recommendations.append(f"\n⚠ Frame Merges Detected ({merges} frames, {merge_rate:.1f}%)")
            recommendations.append("Possible causes:")
            recommendations.append("  • Variable frame rate (VFR) encoding")
            recommendations.append("  • Duplicate frames in source")
            recommendations.append("  • Frame blending effects")
            recommendations.append("  • Improper frame rate conversion")
            recommendations.append("\nRecommended actions:")
            recommendations.append("  → Use constant frame rate (CFR) encoding")
            recommendations.append("  → Check source video integrity")
            recommendations.append("  → Avoid frame rate conversion if possible")
            recommendations.append("  → Review encoding settings")
        
        if drops == 0 and merges == 0:
            recommendations.append("\n✓ No corrective actions needed!")
            recommendations.append("  • Video timing is consistent")
            recommendations.append("  • Frame continuity is excellent")
            recommendations.append("  • Ready for production use")
        
        # Display recommendations
        text = '\n'.join(recommendations)
        ax.text(0.1, 0.9, text, ha='left', va='top',
               fontsize=10, family='sans-serif',
               transform=ax.transAxes)
        
        # Add footer
        footer_text = (
            "\n\n" + "="*70 + "\n"
            "Report generated by TemporalX - Video Temporal Error Detection System\n"
            "For more information, visit the project documentation\n"
            "="*70
        )
        ax.text(0.5, 0.05, footer_text, ha='center', va='bottom',
               fontsize=8, family='monospace', style='italic',
               transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
