"""
TemporalX FastAPI backend.

Provides a POST /analyze endpoint for video file analysis.
"""

import os
import shutil
import tempfile
import traceback
import logging
import csv
import json
from io import StringIO, BytesIO
from pathlib import Path
from typing import Dict, List

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse

from video_error_detector import TemporalErrorDetector

from frame_interpolator import FrameInterpolator
from ffmpeg_utils import fix_mp4_fps_metadata

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_allowed_origins() -> list[str]:
    env_value = os.getenv("ALLOWED_ORIGINS", "*").strip()
    if env_value == "*":
        return ["*"]
    return [origin.strip() for origin in env_value.split(",") if origin.strip()]


app = FastAPI(title="TemporalX API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def _validate_upload(upload: UploadFile) -> None:
    if not upload.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = Path(upload.filename).suffix.lower()
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
    if ext and ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension: {ext}"
        )


def _summarize_results(results: list[Dict], metadata: Dict = None) -> Dict:
    logger.info(f"Summarizing {len(results)} results")
    total_frames = len(results)
    normal = sum(1 for r in results if r.get("classification") == "Normal")
    drops = sum(1 for r in results if r.get("classification") == "Frame Drop")
    merges = sum(1 for r in results if r.get("classification") == "Frame Merge")
    reversals = sum(1 for r in results if r.get("classification") == "Frame Reversal")
    health_score = (100.0 * normal / total_frames) if total_frames > 0 else 0.0
    
    # Calculate FPS
    detected_fps = 0.0
    expected_fps = 0.0
    if metadata:
        expected_fps = float(metadata.get("fps", 0.0))
        logger.info(f"Expected FPS from metadata: {expected_fps}")
    
    if len(results) > 1:
        # Calculate detected FPS from actual timestamps
        first_ts = results[0].get("timestamp", 0)
        last_ts = results[-1].get("timestamp", 0)
        total_duration = last_ts - first_ts
        logger.info(f"Video duration: {total_duration}s (from {first_ts} to {last_ts})")
        if total_duration > 0:
            detected_fps = float((total_frames - 1) / total_duration)
            logger.info(f"Detected FPS: {detected_fps}")
    
    # Calculate average metrics with explicit float conversion
    ssim_scores = [float(r.get("ssim_score", 0)) for r in results if r.get("ssim_score", 0) > 0]
    flow_magnitudes = [float(r.get("flow_magnitude", 0)) for r in results if r.get("flow_magnitude", 0) > 0]
    hist_diffs = [float(r.get("hist_diff", 0)) for r in results if r.get("hist_diff", 0) > 0]
    
    avg_ssim = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0.0
    avg_flow = sum(flow_magnitudes) / len(flow_magnitudes) if flow_magnitudes else 0.0
    avg_hist_diff = sum(hist_diffs) / len(hist_diffs) if hist_diffs else 0.0
    
    # Calculate video duration
    video_duration = float(results[-1].get("timestamp", 0)) if results else 0.0
    
    logger.info(f"Calculated metrics - normal:{normal}, drops:{drops}, merges:{merges}, reversals:{reversals}")
    
    # Build frame-by-frame timeline data for visualization
    timeline_data = []
    for r in results:
        timeline_data.append({
            "frame_num": int(r.get("frame_num", 0)),
            "classification": r.get("classification", "Normal"),
            "confidence": float(r.get("confidence", 0.0))
        })

    return {
        "total_frames": int(total_frames),
        "normal": int(normal),
        "drops": int(drops),
        "merges": int(merges),
        "reversals": int(reversals),
        "health_score": float(round(health_score, 2)),
        "detected_fps": float(round(detected_fps, 2)),
        "expected_fps": float(round(expected_fps, 2)),
        "avg_ssim": float(round(avg_ssim, 4)),
        "avg_flow_magnitude": float(round(avg_flow, 2)),
        "avg_hist_diff": float(round(avg_hist_diff, 4)),
        "video_duration": float(round(video_duration, 2)),
        "drop_percentage": float(round(100.0 * drops / total_frames, 2) if total_frames > 0 else 0.0),
        "merge_percentage": float(round(100.0 * merges / total_frames, 2) if total_frames > 0 else 0.0),
        "reversal_percentage": float(round(100.0 * reversals / total_frames, 2) if total_frames > 0 else 0.0),
        "timeline": timeline_data
    }


def _process_video_fast(detector: TemporalErrorDetector, input_path: str) -> List[Dict]:
    cap, metadata = detector.load_video(input_path)

    results = []
    frame_num = 0
    prev_gray = None
    prev_gray_resized = None
    prev_timestamp = 0.0
    flow_field = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        gray, gray_resized = detector.preprocess_frame(frame)

        if detector.reversal_detection:
            detector.frame_history.append(gray.copy())
            if len(detector.frame_history) > detector.max_history:
                detector.frame_history.pop(0)

        metrics = {
            "frame_num": frame_num,
            "timestamp": timestamp,
            "timestamp_diff": 0.0,
            "expected_interval": metadata["expected_interval"],
            "flow_magnitude": 0.0,
            "ssim_score": 0.0,
            "hist_diff": 0.0,
            "laplacian_var": 0.0,
            "edge_diff": 0.0
        }

        if prev_gray is not None:
            metrics["timestamp_diff"] = timestamp - prev_timestamp
            metrics["flow_magnitude"], flow_field = detector.compute_optical_flow(
                prev_gray_resized, gray_resized
            )
            metrics["ssim_score"] = detector.compute_ssim(prev_gray, gray)
            metrics["hist_diff"] = detector.compute_histogram_difference(prev_gray, gray)
            metrics["laplacian_var"], metrics["edge_diff"] = detector.detect_ghosting_artifacts(gray)

            detector.flow_history.append(metrics["flow_magnitude"])
            detector.ssim_history.append(metrics["ssim_score"])
            detector.hist_history.append(metrics["hist_diff"])
            detector.laplacian_history.append(metrics["laplacian_var"])

            if detector.auto_tune and frame_num % 100 == 0:
                detector.auto_tune_thresholds()

            classification, confidence = detector.classify_frame(
                metrics,
                curr_frame=gray,
                prev_frame=prev_gray,
                flow=flow_field
            )
        else:
            classification, confidence = "Normal", 1.0

        results.append({
            **metrics,
            "classification": classification,
            "confidence": confidence
        })

        prev_gray = gray
        prev_gray_resized = gray_resized
        prev_timestamp = timestamp

    cap.release()
    return results


@app.post("/interpolate")
async def interpolate_video(
    file: UploadFile = File(...), 
    target_fps: int = 240,
    source_fps: float = None,
    return_video: bool = True
):
    """
    Interpolate video frames to achieve higher FPS using optical flow.
    Useful for devices that cannot record at true 240 FPS.
    
    Args:
        file: Input video file
        target_fps: Target FPS (default: 240)
        source_fps: Source FPS (auto-detected if None)
        return_video: If True, returns the interpolated video file; if False, returns stats only
    """
    _validate_upload(file)
    
    # Use a temporary directory that persists beyond the function
    tmpdir = tempfile.mkdtemp()
    tmp_path = Path(tmpdir)
    input_path = tmp_path / "input_video.mp4"
    output_path = tmp_path / f"interpolated_{target_fps}fps.mp4"
    fixed_output_path = tmp_path / f"interpolated_{target_fps}fps_fixed.mp4"
    
    try:
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        if input_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        logger.info(f"Starting interpolation to {target_fps} FPS")
        
        interpolator = FrameInterpolator(use_fast_mode=True)
        stats = interpolator.interpolate_video(
            str(input_path),
            str(output_path),
            target_fps=target_fps,
            source_fps=source_fps
        )
        logger.info(f"Interpolation complete: {stats}")

        # Fix MP4 FPS metadata using ffmpeg
        ffmpeg_success = False
        if output_path.exists():
            ffmpeg_success = fix_mp4_fps_metadata(str(output_path), str(fixed_output_path), target_fps)
            if ffmpeg_success and fixed_output_path.exists():
                final_path = fixed_output_path
            else:
                final_path = output_path
        else:
            final_path = output_path

        if return_video and final_path.exists():
            # Return the interpolated video file with correct FPS metadata
            return FileResponse(
                path=str(final_path),
                media_type='video/mp4',
                filename=f"interpolated_{target_fps}fps.mp4",
                headers={
                    "X-Interpolation-Stats": json.dumps(stats),
                    "X-Source-FPS": str(stats['source_fps']),
                    "X-Target-FPS": str(target_fps),
                    "X-Achieved-FPS": str(stats.get('achieved_fps', target_fps)),
                    "X-FFMPEG-Fixed": str(ffmpeg_success)
                }
            )
        else:
            return JSONResponse(content={
                "status": "success",
                "interpolation_stats": stats,
                "message": f"Video interpolated from {stats['source_fps']} FPS to {stats.get('achieved_fps', target_fps):.2f} FPS"
            })
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error during interpolation: {exc}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            file.file.close()
        except Exception:
            pass


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...), 
    fast: bool = True,
    interpolate: bool = False,
    target_fps: int = 240
) -> JSONResponse:
    _validate_upload(file)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        input_path = tmp_path / "input_video"
        interpolated_path = tmp_path / "interpolated_video.mp4"
        output_path = tmp_path / "annotated_output.mp4"
        csv_path = tmp_path / "results.csv"

        try:
            with open(input_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            if input_path.stat().st_size == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
            # Optional AI interpolation step
            interpolation_stats = None
            video_to_analyze = input_path
            
            if interpolate:
                logger.info(f"AI Interpolation enabled, target FPS: {target_fps}")
                interpolator = FrameInterpolator(use_fast_mode=True)
                interpolation_stats = interpolator.interpolate_video(
                    str(input_path),
                    str(interpolated_path),
                    target_fps=target_fps
                )
                logger.info(f"Interpolation complete: {interpolation_stats}")
                video_to_analyze = interpolated_path

            if fast:
                detector = TemporalErrorDetector(
                    resize_width=320,
                    auto_tune=False,
                    reversal_detection=False
                )
                results = _process_video_fast(detector, str(video_to_analyze))
                # Get metadata for FPS calculation
                cap = cv2.VideoCapture(str(video_to_analyze))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                metadata = {"fps": fps}
            else:
                detector = TemporalErrorDetector()
                results = detector.process_video(
                    input_path=str(video_to_analyze),
                    output_path=str(output_path),
                    csv_path=str(csv_path),
                    show_progress=False
                )
                cap = cv2.VideoCapture(str(video_to_analyze))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                metadata = {"fps": fps}

            summary = _summarize_results(results, metadata)
            
            # Add interpolation stats if available
            if interpolation_stats:
                summary["ai_interpolation"] = {
                    "enabled": True,
                    "source_fps": interpolation_stats.get("source_fps", 0),
                    "target_fps": interpolation_stats.get("target_fps", 0),
                    "achieved_fps": interpolation_stats.get("achieved_fps", 0),
                    "interpolation_factor": interpolation_stats.get("interpolation_factor", 1),
                    "output_frames": interpolation_stats.get("output_frames", 0)
                }
            else:
                summary["ai_interpolation"] = {"enabled": False}
            
            return JSONResponse(content=summary)

        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Error during analysis: {exc}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            try:
                file.file.close()
            except Exception:
                pass


@app.post("/export/csv")
async def export_csv(file: UploadFile = File(...), fast: bool = True) -> StreamingResponse:
    """Export analysis results as CSV file"""
    _validate_upload(file)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        input_path = tmp_path / "input_video"
        
        try:
            with open(input_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            if input_path.stat().st_size == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")

            if fast:
                detector = TemporalErrorDetector(
                    resize_width=320,
                    auto_tune=False,
                    reversal_detection=False
                )
                results = _process_video_fast(detector, str(input_path))
                cap = cv2.VideoCapture(str(input_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                metadata = {"fps": fps}
            else:
                detector = TemporalErrorDetector()
                results = detector.process_video(
                    input_path=str(input_path),
                    output_path=str(tmp_path / "output.mp4"),
                    csv_path=str(tmp_path / "results.csv"),
                    show_progress=False
                )
                cap = cv2.VideoCapture(str(input_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                metadata = {"fps": fps}

            # Generate CSV
            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=[
                'frame_num', 'timestamp', 'classification', 'confidence',
                'flow_magnitude', 'ssim_score', 'hist_diff', 'laplacian_var'
            ])
            writer.writeheader()
            
            for r in results:
                writer.writerow({
                    'frame_num': int(r.get('frame_num', 0)),
                    'timestamp': float(r.get('timestamp', 0)),
                    'classification': r.get('classification', 'Normal'),
                    'confidence': float(r.get('confidence', 0.0)),
                    'flow_magnitude': float(r.get('flow_magnitude', 0.0)),
                    'ssim_score': float(r.get('ssim_score', 0.0)),
                    'hist_diff': float(r.get('hist_diff', 0.0)),
                    'laplacian_var': float(r.get('laplacian_var', 0.0))
                })
            
            csv_content = output.getvalue()
            logger.info(f"Generated CSV with {len(results)} frames")
            
            return StreamingResponse(
                iter([csv_content]),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=chronovision_analysis.csv"}
            )

        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Error during CSV export: {exc}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            try:
                file.file.close()
            except Exception:
                pass


@app.post("/export/json")
async def export_json(file: UploadFile = File(...), fast: bool = True):
    """Export complete analysis results as JSON file"""
    _validate_upload(file)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        input_path = tmp_path / "input_video"
        
        try:
            with open(input_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            if input_path.stat().st_size == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")

            if fast:
                detector = TemporalErrorDetector(
                    resize_width=320,
                    auto_tune=False,
                    reversal_detection=False
                )
                results = _process_video_fast(detector, str(input_path))
                cap = cv2.VideoCapture(str(input_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                metadata = {"fps": fps}
            else:
                detector = TemporalErrorDetector()
                results = detector.process_video(
                    input_path=str(input_path),
                    output_path=str(tmp_path / "output.mp4"),
                    csv_path=str(tmp_path / "results.csv"),
                    show_progress=False
                )
                cap = cv2.VideoCapture(str(input_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                metadata = {"fps": fps}

            summary = _summarize_results(results, metadata)
            
            # Generate JSON content
            json_content = json.dumps(summary, indent=2)
            json_bytes = json_content.encode('utf-8')
            
            logger.info(f"Generated JSON export with analysis data")
            
            return StreamingResponse(
                iter([json_bytes]),
                media_type="application/json",
                headers={"Content-Disposition": "attachment; filename=chronovision_analysis.json"}
            )

        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Error during JSON export: {exc}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            try:
                file.file.close()
            except Exception:
                pass
