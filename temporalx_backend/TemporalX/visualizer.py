"""
Visualization and Analysis Tools for Temporal Error Detection
==============================================================
Generate graphs and visual analytics for detection results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from typing import List, Dict
import os


class DetectionVisualizer:
    """
    Create visualization graphs for temporal error detection results.
    """
    
    def __init__(self, results: List[Dict] = None, csv_path: str = None):
        """
        Initialize visualizer with detection results.
        
        Args:
            results: List of result dictionaries from detector
            csv_path: Path to CSV file with results (alternative to results)
        """
        if csv_path:
            self.df = pd.read_csv(csv_path)
        elif results:
            self.df = pd.DataFrame(results)
        else:
            raise ValueError("Must provide either results list or csv_path")
        
        # Setup style
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_all_metrics(self, output_path: str = 'analysis_dashboard.png', dpi: int = 150):
        """
        Create comprehensive dashboard with all metrics.
        
        Args:
            output_path: Output image path
            dpi: Image resolution
        """
        fig, axes = plt.subplots(4, 1, figsize=(16, 12))
        fig.suptitle('Video Temporal Error Detection - Analysis Dashboard', 
                    fontsize=16, fontweight='bold')
        
        # Color mapping for classifications
        color_map = {
            'Normal': 'green',
            'Frame Drop': 'red',
            'Frame Merge': 'yellow'
        }
        
        colors = [color_map.get(c, 'gray') for c in self.df['classification']]
        
        # Plot 1: Optical Flow Magnitude
        ax1 = axes[0]
        ax1.plot(self.df['frame_num'], self.df['flow_magnitude'], 
                linewidth=0.5, color='blue', alpha=0.7)
        ax1.scatter(self.df['frame_num'], self.df['flow_magnitude'], 
                   c=colors, s=10, alpha=0.6, edgecolors='none')
        ax1.set_ylabel('Optical Flow Magnitude', fontweight='bold')
        ax1.set_title('Motion Analysis - Optical Flow')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=self.df['flow_magnitude'].mean(), color='orange', 
                   linestyle='--', label='Mean', linewidth=2)
        ax1.legend()
        
        # Plot 2: SSIM Score
        ax2 = axes[1]
        ax2.plot(self.df['frame_num'], self.df['ssim_score'], 
                linewidth=0.5, color='purple', alpha=0.7)
        ax2.scatter(self.df['frame_num'], self.df['ssim_score'], 
                   c=colors, s=10, alpha=0.6, edgecolors='none')
        ax2.set_ylabel('SSIM Score', fontweight='bold')
        ax2.set_title('Structural Similarity Analysis')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=self.df['ssim_score'].mean(), color='orange', 
                   linestyle='--', label='Mean', linewidth=2)
        ax2.legend()
        
        # Plot 3: Histogram Difference
        ax3 = axes[2]
        ax3.plot(self.df['frame_num'], self.df['hist_diff'], 
                linewidth=0.5, color='brown', alpha=0.7)
        ax3.scatter(self.df['frame_num'], self.df['hist_diff'], 
                   c=colors, s=10, alpha=0.6, edgecolors='none')
        ax3.set_ylabel('Histogram Difference', fontweight='bold')
        ax3.set_title('Scene Discontinuity Analysis')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=self.df['hist_diff'].mean(), color='orange', 
                   linestyle='--', label='Mean', linewidth=2)
        ax3.legend()
        
        # Plot 4: Laplacian Variance (Blur Detection)
        ax4 = axes[3]
        ax4.plot(self.df['frame_num'], self.df['laplacian_var'], 
                linewidth=0.5, color='darkgreen', alpha=0.7)
        ax4.scatter(self.df['frame_num'], self.df['laplacian_var'], 
                   c=colors, s=10, alpha=0.6, edgecolors='none')
        ax4.set_ylabel('Laplacian Variance', fontweight='bold')
        ax4.set_xlabel('Frame Number', fontweight='bold')
        ax4.set_title('Blur/Sharpness Analysis')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=self.df['laplacian_var'].mean(), color='orange', 
                   linestyle='--', label='Mean', linewidth=2)
        ax4.legend()
        
        # Add legend for classifications
        legend_elements = [
            mpatches.Patch(color='green', label='Normal'),
            mpatches.Patch(color='red', label='Frame Drop'),
            mpatches.Patch(color='yellow', label='Frame Merge')
        ]
        fig.legend(handles=legend_elements, loc='upper right', 
                  bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Dashboard saved to: {output_path}")
        plt.close()
    
    def plot_detection_timeline(self, output_path: str = 'detection_timeline.png', dpi: int = 150):
        """
        Create timeline visualization of detections.
        
        Args:
            output_path: Output image path
            dpi: Image resolution
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
        fig.suptitle('Detection Timeline', fontsize=16, fontweight='bold')
        
        # Classification over time
        classification_numeric = {
            'Normal': 0,
            'Frame Drop': 1,
            'Frame Merge': 2
        }
        numeric_class = [classification_numeric[c] for c in self.df['classification']]
        
        # Plot 1: Classification timeline with confidence
        colors = []
        for c in self.df['classification']:
            if c == 'Normal':
                colors.append('green')
            elif c == 'Frame Drop':
                colors.append('red')
            else:
                colors.append('yellow')
        
        ax1.scatter(self.df['frame_num'], numeric_class, 
                   c=colors, alpha=self.df['confidence'], s=30, edgecolors='black', linewidth=0.5)
        ax1.set_yticks([0, 1, 2])
        ax1.set_yticklabels(['Normal', 'Frame Drop', 'Frame Merge'])
        ax1.set_ylabel('Classification', fontweight='bold')
        ax1.set_title('Frame Classification Over Time (opacity = confidence)')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Plot 2: Confidence score
        ax2.bar(self.df['frame_num'], self.df['confidence'], 
               color=colors, alpha=0.7, width=1, edgecolor='none')
        ax2.set_ylabel('Confidence Score', fontweight='bold')
        ax2.set_xlabel('Frame Number', fontweight='bold')
        ax2.set_title('Detection Confidence')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Timeline saved to: {output_path}")
        plt.close()
    
    def plot_anomaly_heatmap(self, output_path: str = 'anomaly_heatmap.png', dpi: int = 150):
        """
        Create heatmap showing anomaly intensity.
        
        Args:
            output_path: Output image path
            dpi: Image resolution
        """
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Create anomaly score (0 for normal, confidence for anomalies)
        anomaly_score = []
        for idx, row in self.df.iterrows():
            if row['classification'] == 'Normal':
                anomaly_score.append(0)
            else:
                anomaly_score.append(row['confidence'])
        
        # Create color array
        colors = []
        for c, score in zip(self.df['classification'], anomaly_score):
            if c == 'Normal':
                colors.append('green')
            elif c == 'Frame Drop':
                colors.append('red')
            else:
                colors.append('yellow')
        
        # Plot as bars
        ax.bar(self.df['frame_num'], anomaly_score, 
              color=colors, alpha=0.8, width=1, edgecolor='none')
        
        ax.set_xlabel('Frame Number', fontweight='bold')
        ax.set_ylabel('Anomaly Intensity', fontweight='bold')
        ax.set_title('Temporal Error Anomaly Heatmap', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='green', label='Normal'),
            mpatches.Patch(color='red', label='Frame Drop'),
            mpatches.Patch(color='yellow', label='Frame Merge')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Heatmap saved to: {output_path}")
        plt.close()
    
    def plot_statistics_summary(self, output_path: str = 'statistics_summary.png', dpi: int = 150):
        """
        Create statistical summary visualizations.
        
        Args:
            output_path: Output image path
            dpi: Image resolution
        """
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Statistical Summary', fontsize=16, fontweight='bold')
        
        # Plot 1: Classification distribution (pie chart)
        ax1 = fig.add_subplot(gs[0, 0])
        class_counts = self.df['classification'].value_counts()
        colors_pie = [{'Normal': 'green', 'Frame Drop': 'red', 'Frame Merge': 'yellow'}[c] 
                     for c in class_counts.index]
        ax1.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
               colors=colors_pie, startangle=90)
        ax1.set_title('Classification Distribution')
        
        # Plot 2: Confidence distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(self.df['confidence'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Confidence Score Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Flow magnitude distribution
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(self.df['flow_magnitude'], bins=50, color='blue', edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Optical Flow Magnitude')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Optical Flow Distribution')
        ax3.grid(True, alpha=0.3)
        ax3.axvline(self.df['flow_magnitude'].mean(), color='red', 
                   linestyle='--', linewidth=2, label='Mean')
        ax3.legend()
        
        # Plot 4: SSIM distribution
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(self.df['ssim_score'], bins=50, color='purple', edgecolor='black', alpha=0.7)
        ax4.set_xlabel('SSIM Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('SSIM Distribution')
        ax4.grid(True, alpha=0.3)
        ax4.axvline(self.df['ssim_score'].mean(), color='red', 
                   linestyle='--', linewidth=2, label='Mean')
        ax4.legend()
        
        # Plot 5: Box plot of metrics by classification
        ax5 = fig.add_subplot(gs[2, :])
        data_to_plot = []
        labels = []
        for classification in ['Normal', 'Frame Drop', 'Frame Merge']:
            subset = self.df[self.df['classification'] == classification]
            if len(subset) > 0:
                data_to_plot.append(subset['flow_magnitude'])
                labels.append(f"{classification}\n(n={len(subset)})")
        
        bp = ax5.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, label in zip(bp['boxes'], labels):
            if 'Normal' in label:
                patch.set_facecolor('lightgreen')
            elif 'Drop' in label:
                patch.set_facecolor('lightcoral')
            else:
                patch.set_facecolor('lightyellow')
        
        ax5.set_ylabel('Optical Flow Magnitude')
        ax5.set_title('Flow Magnitude by Classification')
        ax5.grid(True, alpha=0.3, axis='y')
        
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Statistics summary saved to: {output_path}")
        plt.close()
    
    def generate_all_visualizations(self, output_dir: str = 'visualizations', dpi: int = 150):
        """
        Generate all visualization reports.
        
        Args:
            output_dir: Directory to save visualizations
            dpi: Image resolution
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nGenerating visualizations...")
        print("-" * 60)
        
        self.plot_all_metrics(os.path.join(output_dir, 'analysis_dashboard.png'), dpi)
        self.plot_detection_timeline(os.path.join(output_dir, 'detection_timeline.png'), dpi)
        self.plot_anomaly_heatmap(os.path.join(output_dir, 'anomaly_heatmap.png'), dpi)
        self.plot_anomaly_timeline(os.path.join(output_dir, 'anomaly_timeline.png'), dpi)
        self.plot_statistics_summary(os.path.join(output_dir, 'statistics_summary.png'), dpi)
        
        print("-" * 60)
        print(f"✓ All visualizations saved to: {output_dir}/")
        print("\nGenerated files:")
        print("  - analysis_dashboard.png")
        print("  - detection_timeline.png")
        print("  - anomaly_heatmap.png")
        print("  - anomaly_timeline.png")
        print("  - statistics_summary.png")
    
    def print_detailed_report(self):
        """Print detailed statistical report to console."""
        print("\n" + "="*70)
        print("DETAILED STATISTICAL REPORT")
        print("="*70)
        
        # Overall statistics
        print("\n1. OVERALL STATISTICS")
        print("-" * 70)
        print(f"Total Frames Analyzed:    {len(self.df)}")
        print(f"Video Duration:           {self.df['timestamp'].max():.2f} seconds")
        print(f"Average FPS:              {len(self.df) / self.df['timestamp'].max():.2f}")
        
        # Classification breakdown
        print("\n2. CLASSIFICATION BREAKDOWN")
        print("-" * 70)
        for classification in ['Normal', 'Frame Drop', 'Frame Merge']:
            count = len(self.df[self.df['classification'] == classification])
            percentage = 100 * count / len(self.df)
            avg_conf = self.df[self.df['classification'] == classification]['confidence'].mean()
            print(f"{classification:15s} {count:6d} frames  ({percentage:5.2f}%)  "
                  f"Avg Confidence: {avg_conf:.3f}")
        
        # Metric statistics
        print("\n3. METRIC STATISTICS")
        print("-" * 70)
        metrics = ['flow_magnitude', 'ssim_score', 'hist_diff', 'laplacian_var']
        for metric in metrics:
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  Mean:   {self.df[metric].mean():.4f}")
            print(f"  Median: {self.df[metric].median():.4f}")
            print(f"  Std:    {self.df[metric].std():.4f}")
            print(f"  Min:    {self.df[metric].min():.4f}")
            print(f"  Max:    {self.df[metric].max():.4f}")
        
        # Anomaly clusters
        print("\n4. ANOMALY CLUSTERS")
        print("-" * 70)
        anomalies = self.df[self.df['classification'] != 'Normal']
        if len(anomalies) > 0:
            print(f"Total Anomalies:          {len(anomalies)}")
            print(f"First Anomaly at Frame:   {anomalies.iloc[0]['frame_num']:.0f}")
            print(f"Last Anomaly at Frame:    {anomalies.iloc[-1]['frame_num']:.0f}")
            
            # Find consecutive anomaly clusters
            anomaly_frames = anomalies['frame_num'].values
            clusters = []
            if len(anomaly_frames) > 0:
                cluster_start = anomaly_frames[0]
                cluster_end = anomaly_frames[0]
                
                for i in range(1, len(anomaly_frames)):
                    if anomaly_frames[i] == cluster_end + 1:
                        cluster_end = anomaly_frames[i]
                    else:
                        clusters.append((cluster_start, cluster_end))
                        cluster_start = anomaly_frames[i]
                        cluster_end = anomaly_frames[i]
                clusters.append((cluster_start, cluster_end))
            
            print(f"Number of Clusters:       {len(clusters)}")
            print(f"Largest Cluster:          {max([end-start+1 for start, end in clusters])} frames")
        else:
            print("No anomalies detected!")
        
        print("\n" + "="*70 + "\n")
    
    def plot_anomaly_timeline(self, output_path: str = 'anomaly_timeline.png', dpi: int = 150):
        """
        🎯 ELITE VISUALIZATION: Multi-dimensional anomaly timeline
        
        Plot optical flow magnitude, SSIM, and anomaly spikes over time.
        This visualization explains WHAT went wrong and WHEN.
        
        Args:
            output_path: Output image path
            dpi: Image resolution
        """
        fig, axes = plt.subplots(3, 1, figsize=(16, 10))
        fig.suptitle('🎯 Temporal Error Anomaly Timeline\nMulti-Signal Confidence Analysis', 
                    fontsize=14, fontweight='bold')
        
        frames = self.df['frame_num'].values
        
        # ===== PLOT 1: Optical Flow Magnitude Over Time =====
        ax1 = axes[0]
        flow_mag = self.df['flow_magnitude'].values
        
        # Plot normal flow
        ax1.plot(frames, flow_mag, 'b-', linewidth=1.5, alpha=0.7, label='Optical Flow Magnitude')
        
        # Highlight drop detection zones
        drops = self.df[self.df['classification'] == 'Frame Drop']
        if len(drops) > 0:
            ax1.scatter(drops['frame_num'], drops['flow_magnitude'], 
                       color='red', s=100, marker='X', label='Frame Drop', zorder=5)
        
        # Add mean and std bands
        flow_mean = flow_mag.mean()
        flow_std = flow_mag.std()
        ax1.axhline(flow_mean, color='blue', linestyle='--', alpha=0.3, label=f'Mean: {flow_mean:.2f}')
        ax1.fill_between(frames, flow_mean, flow_mean + 2.5*flow_std, alpha=0.2, color='red', 
                        label=f'Anomaly Zone (μ + 2.5σ)')
        
        ax1.set_ylabel('Optical Flow Magnitude', fontsize=11, fontweight='bold')
        ax1.set_title('Signal 1: Motion Discontinuity Detection', fontsize=10)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # ===== PLOT 2: SSIM (Structural Similarity) Over Time =====
        ax2 = axes[1]
        ssim_scores = self.df['ssim_score'].values
        
        # Plot SSIM
        ax2.plot(frames, ssim_scores, 'g-', linewidth=1.5, alpha=0.7, label='SSIM Score')
        
        # Highlight merge detection zones
        merges = self.df[self.df['classification'] == 'Frame Merge']
        if len(merges) > 0:
            ax2.scatter(merges['frame_num'], merges['ssim_score'], 
                       color='orange', s=100, marker='^', label='Frame Merge', zorder=5)
        
        # Add threshold bands
        ssim_mean = ssim_scores.mean()
        ssim_std = ssim_scores.std()
        ax2.axhline(0.85, color='orange', linestyle='--', alpha=0.3, label='Merge Threshold (0.85)')
        ax2.fill_between(frames, 0.85, 1.0, alpha=0.15, color='orange', 
                        label='Merge Suspicious Zone')
        
        ax2.set_ylabel('SSIM Score (0-1)', fontsize=11, fontweight='bold')
        ax2.set_title('Signal 2: Frame Similarity & Blending Detection', fontsize=10)
        ax2.legend(loc='lower right', fontsize=9)
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3)
        
        # ===== PLOT 3: Combined Anomaly Confidence Score =====
        ax3 = axes[2]
        
        # Calculate combined anomaly score for visualization
        confidence = self.df['confidence'].values
        
        # Create color gradient based on classification
        colors = []
        for cls in self.df['classification']:
            if cls == 'Frame Drop':
                colors.append('red')
            elif cls == 'Frame Merge':
                colors.append('orange')
            elif cls == 'Frame Reversal':
                colors.append('magenta')
            else:
                colors.append('green')
        
        # Plot confidence bars
        ax3.bar(frames, confidence, color=colors, alpha=0.6, width=1.0, edgecolor='black', linewidth=0.5)
        
        # Threshold lines
        ax3.axhline(0.55, color='red', linestyle='--', linewidth=2, label='Drop Threshold (0.55)')
        ax3.axhline(0.50, color='orange', linestyle='--', linewidth=2, label='Merge Threshold (0.50)')
        ax3.fill_between(frames, 0.50, 1.0, alpha=0.1, color='red', label='Critical Zone')
        
        ax3.set_xlabel('Frame Number', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Detection Confidence Score', fontsize=11, fontweight='bold')
        ax3.set_title('Signal 3: Multi-Signal Confidence Scoring (Combined)', fontsize=10)
        ax3.set_ylim([0, 1.05])
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add summary statistics
        summary_text = (
            f"Total Errors: {len(self.df[self.df['classification'] != 'Normal'])} | "
            f"Drops: {len(drops)} | "
            f"Merges: {len(merges)} | "
            f"Quality: {100*len(self.df[self.df['classification'] == 'Normal'])/len(self.df):.1f}%"
        )
        fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10, 
                style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ Anomaly timeline saved: {output_path}")
        plt.close()
        
        return output_path

def main():
    """CLI for visualization tool."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize temporal error detection results')
    parser.add_argument('-i', '--input', required=True, help='Input CSV file from detection')
    parser.add_argument('-o', '--output-dir', default='visualizations', 
                       help='Output directory for visualizations')
    parser.add_argument('--dpi', type=int, default=150, help='Image DPI')
    parser.add_argument('--report', action='store_true', help='Print detailed report')
    
    args = parser.parse_args()
    
    # Load and visualize
    visualizer = DetectionVisualizer(csv_path=args.input)
    visualizer.generate_all_visualizations(args.output_dir, args.dpi)
    
    if args.report:
        visualizer.print_detailed_report()


if __name__ == "__main__":
    main()
