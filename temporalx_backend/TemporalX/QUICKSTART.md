# TemporalX - Quick Start Guide

## 🚀 5-Minute Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Test the System
```bash
# Run example demos (creates test video automatically)
python examples.py
# Select option 1 for basic detection
```

### Step 3: Analyze Your Video
```bash
python cli.py --input your_video.mp4 --output annotated_output.mp4
```

### Step 4: Generate Visualizations
```bash
python visualizer.py -i detection_results.csv --report
```

---

## 📋 Common Usage Patterns

### Pattern 1: Quick Analysis
```bash
python cli.py -i video.mp4 -o output.mp4
```

### Pattern 2: High-Sensitivity Detection
```bash
python cli.py -i video.mp4 -o output.mp4 \
    --flow-threshold 20 \
    --ssim-threshold 0.92 \
    --hist-threshold 0.25
```

### Pattern 3: Fast Processing (Lower Resolution)
```bash
python cli.py -i video.mp4 -o output.mp4 \
    --resize-width 480 \
    --silent
```

### Pattern 4: CSV Only (No Video Output)
```bash
python cli.py -i video.mp4 --csv results.csv --no-video
```

### Pattern 5: Real-time Webcam
```bash
python webcam_detector.py --mode webcam --fps 30
```

---

## 🎯 Recommended Thresholds by Video Type

### Static Camera (Security, Surveillance)
```bash
python cli.py -i video.mp4 -o output.mp4 \
    --flow-threshold 25 \
    --ssim-threshold 0.92 \
    --hist-threshold 0.25
```

### Moving Camera (Sports, Action)
```bash
python cli.py -i video.mp4 -o output.mp4 \
    --flow-threshold 45 \
    --ssim-threshold 0.85 \
    --hist-threshold 0.50
```

### Screen Recording
```bash
python cli.py -i video.mp4 -o output.mp4 \
    --flow-threshold 20 \
    --ssim-threshold 0.95 \
    --hist-threshold 0.20
```

### Webcam/Streaming
```bash
python cli.py -i video.mp4 -o output.mp4 \
    --flow-threshold 35 \
    --ssim-threshold 0.88 \
    --hist-threshold 0.35
```

---

## 🔍 Understanding Output

### Annotated Video Colors
- **Green Border** = Normal frame (no issues)
- **Red Border** = Frame Drop detected
- **Yellow Border** = Frame Merge detected

### CSV Columns
- `frame_num`: Frame number in sequence
- `timestamp`: Time in seconds
- `flow_magnitude`: Optical flow intensity
- `ssim_score`: Structural similarity (0-1)
- `hist_diff`: Histogram difference
- `laplacian_var`: Sharpness/blur metric
- `edge_diff`: Edge artifact metric
- `classification`: Normal/Frame Drop/Frame Merge
- `confidence`: Detection confidence (0-1)

### Visualization Files
- `analysis_dashboard.png`: 4-panel metric timeline
- `detection_timeline.png`: Classification over time
- `anomaly_heatmap.png`: Visual anomaly intensity
- `statistics_summary.png`: Statistical distributions

---

## 🛠️ Troubleshooting

### Problem: Too many false positives
**Solution**: Increase thresholds or enable auto-tuning
```bash
python cli.py -i video.mp4 -o output.mp4 --flow-threshold 50
```

### Problem: Missing obvious errors
**Solution**: Decrease thresholds
```bash
python cli.py -i video.mp4 -o output.mp4 --flow-threshold 20
```

### Problem: Slow processing
**Solution**: Reduce processing resolution
```bash
python cli.py -i video.mp4 -o output.mp4 --resize-width 480
```

### Problem: Out of memory
**Solution**: Process in segments or reduce resolution

---

## 💡 Pro Tips

### Tip 1: Auto-Tuning
Let the system automatically adjust thresholds:
```bash
python cli.py -i video.mp4 -o output.mp4
# Auto-tuning is enabled by default
```

### Tip 2: Disable Auto-Tuning for Consistency
For batch processing multiple similar videos:
```bash
python cli.py -i video.mp4 -o output.mp4 --no-auto-tune
```

### Tip 3: Silent Mode for Batch Processing
```bash
python cli.py -i video.mp4 -o output.mp4 --silent
```

### Tip 4: High-Quality Visualizations
```bash
python visualizer.py -i results.csv --dpi 300
```

### Tip 5: Quick Quality Check
```bash
# CSV only (fastest)
python cli.py -i video.mp4 --csv results.csv --no-video --silent
# Then check the CSV
```

---

## 📊 Interpreting Results

### Good Video Quality
```
Normal Frames:      95-100%
Frame Drops:        0-2%
Frame Merges:       0-1%
```

### Moderate Issues
```
Normal Frames:      85-95%
Frame Drops:        2-5%
Frame Merges:       1-3%
```

### Poor Quality
```
Normal Frames:      <85%
Frame Drops:        >5%
Frame Merges:       >3%
```

---

## 🎓 Learning Path

1. **Start Here**: Run `python examples.py` (Example 1)
2. **Experiment**: Try different threshold values
3. **Visualize**: Generate analysis graphs
4. **Real-time**: Test webcam detection
5. **Advanced**: Use programmatic API

---

## 📞 Need Help?

1. Check `README.md` for detailed documentation
2. Run examples: `python examples.py`
3. Check CLI help: `python cli.py --help`
4. Review CSV output for detailed metrics

---

## 🏆 Quick Win Demo

```bash
# Complete workflow in 3 commands:

# 1. Create test video and analyze
python examples.py  # Select option 1

# 2. Generate visualizations
python visualizer.py -i results_basic.csv -o viz_output --report

# 3. View results
# - Video: output_basic.mp4
# - Graphs: viz_output/
# - Data: results_basic.csv
```

---

**You're ready to detect temporal errors like a pro! 🎥✨**
