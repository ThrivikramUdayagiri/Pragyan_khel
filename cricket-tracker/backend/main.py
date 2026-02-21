"""
Cricket Video Player Tracking and Highlight System - Backend API

This FastAPI application provides endpoints for:
1. Video upload and initial processing with YOLOv8 + DeepSORT
2. Player selection and highlight effect generation
3. Video streaming and download
4. Progressive preview during processing

Author: Cricket Tracker Team
"""

import os
import uuid
import json
import asyncio
import base64
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil
import cv2
import numpy as np

from tracker import VideoTracker
from processor import VideoProcessor

# Initialize FastAPI app
app = FastAPI(
    title="Cricket Video Player Tracking API",
    description="API for tracking and highlighting players in cricket videos",
    version="1.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for storing videos and processed data
UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
TRACKING_DIR = Path("tracking_data")

for directory in [UPLOAD_DIR, PROCESSED_DIR, TRACKING_DIR]:
    directory.mkdir(exist_ok=True)

# Mount static files directory for video serving
app.mount("/videos", StaticFiles(directory="processed"), name="videos")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Store processing status in memory (use Redis in production)
processing_status = {}

# Store active tracker instances for preview during processing
active_trackers = {}

# Store loaded video frames for preview
video_frames_cache = {}


class PlayerSelectionRequest(BaseModel):
    """Request model for player selection"""
    video_id: str
    player_id: int
    frame_number: Optional[int] = None


class ClickPositionRequest(BaseModel):
    """Request model for click position to identify player"""
    video_id: str
    x: float  # Normalized x coordinate (0-1)
    y: float  # Normalized y coordinate (0-1)
    frame_number: int


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "running", "message": "Cricket Video Tracker API"}


@app.post("/api/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a cricket video for processing.
    
    The video will be:
    1. Saved to the uploads directory
    2. Processed with YOLOv8 segmentation
    3. Tracked with DeepSORT
    4. Tracking data saved for later use
    
    Returns a video_id for tracking progress and results.
    """
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a video file (mp4, avi, mov, mkv)"
        )
    
    # Generate unique video ID
    video_id = str(uuid.uuid4())
    
    # Save uploaded file
    file_extension = Path(file.filename).suffix
    input_path = UPLOAD_DIR / f"{video_id}{file_extension}"
    
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Initialize processing status
    processing_status[video_id] = {
        "status": "uploaded",
        "progress": 0,
        "message": "Video uploaded, starting processing...",
        "input_path": str(input_path),
        "tracking_data_path": None,
        "processed_video_path": None
    }
    
    # Start background processing
    background_tasks.add_task(process_video_task, video_id, str(input_path))
    
    return {
        "video_id": video_id,
        "message": "Video uploaded successfully. Processing started.",
        "status": "processing"
    }


async def process_video_task(video_id: str, input_path: str):
    """
    Background task for processing video with YOLOv8 and DeepSORT.
    Uses binary subdivision frame order for progressive preview.
    
    This function:
    1. Loads the video
    2. Runs YOLOv8 segmentation on each frame in binary order
    3. Applies DeepSORT tracking
    4. Saves tracking data (bounding boxes, masks, IDs)
    5. Generates a preview video with tracking overlays
    """
    try:
        processing_status[video_id]["status"] = "processing"
        processing_status[video_id]["message"] = "Initializing tracker..."
        
        # Initialize tracker and store for preview access
        tracker = VideoTracker()
        active_trackers[video_id] = tracker
        
        # Load video frames for preview
        cap = cv2.VideoCapture(input_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        video_frames_cache[video_id] = frames
        
        # Process video and get tracking data
        tracking_data_path = TRACKING_DIR / f"{video_id}_tracking.json"
        preview_video_path = PROCESSED_DIR / f"{video_id}_preview.mp4"
        
        def progress_callback(progress, message):
            processing_status[video_id]["progress"] = progress
            processing_status[video_id]["message"] = message
            # Add preview availability info
            if tracker.frames_buffer:
                processing_status[video_id]["preview_available"] = True
                processing_status[video_id]["processed_frames"] = tracker.frames_buffer.get_processed_count()
        
        # Run tracking on the video
        tracking_data = await asyncio.to_thread(
            tracker.process_video,
            input_path,
            str(preview_video_path),
            progress_callback
        )
        
        # Save tracking data to JSON
        with open(tracking_data_path, 'w') as f:
            json.dump(tracking_data, f)
        
        # Cleanup cached data
        if video_id in active_trackers:
            del active_trackers[video_id]
        if video_id in video_frames_cache:
            del video_frames_cache[video_id]
        
        # Update status
        processing_status[video_id].update({
            "status": "completed",
            "progress": 100,
            "message": "Processing complete!",
            "tracking_data_path": str(tracking_data_path),
            "processed_video_path": str(preview_video_path),
            "player_ids": list(tracking_data.get("unique_ids", [])),
            "total_frames": tracking_data.get("total_frames", 0)
        })
        
    except Exception as e:
        processing_status[video_id].update({
            "status": "error",
            "message": f"Processing failed: {str(e)}",
            "progress": 0
        })
        # Cleanup on error
        if video_id in active_trackers:
            del active_trackers[video_id]
        if video_id in video_frames_cache:
            del video_frames_cache[video_id]
        print(f"Error processing video {video_id}: {str(e)}")


@app.get("/api/status/{video_id}")
async def get_processing_status(video_id: str):
    """
    Get the current processing status of a video.
    
    Returns:
    - status: uploaded, processing, completed, error
    - progress: 0-100
    - message: Human-readable status message
    - player_ids: List of detected player IDs (when completed)
    """
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return processing_status[video_id]


@app.get("/api/preview/{video_id}/frame/{frame_number}")
async def get_preview_frame(video_id: str, frame_number: int):
    """
    Get a preview frame during processing.
    Uses nearest-frame preview method: if the requested frame is not yet
    processed, returns the nearest processed frame.
    
    Returns:
    - actual_frame: The frame number actually returned
    - requested_frame: The originally requested frame
    - is_exact_match: Whether the returned frame is the requested one
    - frame_data: Tracking data for the frame
    - image: Base64 encoded JPEG image
    """
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video not found")
    
    status = processing_status[video_id]
    
    # If processing is complete, serve from tracking data
    if status["status"] == "completed":
        tracking_data_path = status.get("tracking_data_path")
        if tracking_data_path and Path(tracking_data_path).exists():
            with open(tracking_data_path, 'r') as f:
                tracking_data = json.load(f)
            
            frame_key = str(frame_number)
            frame_data = tracking_data.get("frames", {}).get(frame_key, {})
            
            # Get the actual video frame
            input_path = status.get("input_path")
            if input_path:
                cap = cv2.VideoCapture(input_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    image_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    return {
                        "actual_frame": frame_number,
                        "requested_frame": frame_number,
                        "is_exact_match": True,
                        "frame_data": frame_data,
                        "image": image_base64
                    }
        
        raise HTTPException(status_code=404, detail="Frame not found")
    
    # During processing, use the tracker's preview capability
    if video_id not in active_trackers:
        raise HTTPException(
            status_code=400,
            detail="Video is not currently being processed or preview not available"
        )
    
    tracker = active_trackers[video_id]
    actual_frame_num, frame_data, _ = tracker.get_preview_frame(frame_number)
    
    if actual_frame_num is None:
        raise HTTPException(status_code=404, detail="No processed frames available yet")
    
    # Get the video frame from cache
    if video_id in video_frames_cache and actual_frame_num < len(video_frames_cache[video_id]):
        frame = video_frames_cache[video_id][actual_frame_num]
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "actual_frame": actual_frame_num,
            "requested_frame": frame_number,
            "is_exact_match": actual_frame_num == frame_number,
            "frame_data": frame_data,
            "image": image_base64
        }
    
    raise HTTPException(status_code=404, detail="Frame not found in cache")


@app.get("/api/preview/{video_id}/tracking-data")
async def get_partial_tracking_data(video_id: str):
    """
    Get partial tracking data during processing for preview purposes.
    Returns currently processed frames' tracking data.
    """
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video not found")
    
    status = processing_status[video_id]
    
    # If completed, return full tracking data
    if status["status"] == "completed":
        tracking_data_path = status.get("tracking_data_path")
        if tracking_data_path and Path(tracking_data_path).exists():
            with open(tracking_data_path, 'r') as f:
                tracking_data = json.load(f)
            tracking_data["is_partial"] = False
            return tracking_data
    
    # During processing, get partial data from tracker
    if video_id not in active_trackers:
        return {
            "frames": {},
            "unique_ids": [],
            "is_partial": True,
            "message": "Processing not started yet"
        }
    
    tracker = active_trackers[video_id]
    partial_data = tracker.get_partial_tracking_data()
    
    if partial_data:
        return partial_data
    
    return {
        "frames": {},
        "unique_ids": [],
        "is_partial": True,
        "message": "No frames processed yet"
    }


@app.get("/api/preview/{video_id}/info")
async def get_preview_info(video_id: str):
    """
    Get preview availability info during processing.
    Returns processing_order with frames in binary subdivision order.
    """
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video not found")
    
    status = processing_status[video_id]
    
    if status["status"] == "completed":
        # For completed videos, generate the full binary subdivision order
        total_frames = status.get("total_frames", 0)
        from tracker import generate_binary_subdivision_order
        processing_order = generate_binary_subdivision_order(total_frames)
        return {
            "preview_available": True,
            "is_complete": True,
            "processed_frames": total_frames,
            "total_frames": total_frames,
            "progress": 100,
            "processing_order": processing_order
        }
    
    if video_id not in active_trackers:
        return {
            "preview_available": False,
            "is_complete": False,
            "processed_frames": 0,
            "total_frames": 0,
            "progress": 0,
            "processing_order": []
        }
    
    tracker = active_trackers[video_id]
    progress_info = tracker.get_processing_progress()
    
    return {
        "preview_available": progress_info["processed_frames"] > 0,
        "is_complete": progress_info["is_complete"],
        "processed_frames": progress_info["processed_frames"],
        "total_frames": progress_info["total_frames"],
        "progress": progress_info["progress"],
        "processing_order": progress_info.get("processing_order", [])
    }


@app.get("/api/tracking-data/{video_id}")
async def get_tracking_data(video_id: str):
    """
    Get the tracking data for a processed video.
    
    Returns frame-by-frame tracking information including:
    - Bounding boxes
    - Segmentation mask coordinates
    - Player IDs
    """
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video not found")
    
    status = processing_status[video_id]
    if status["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail="Video processing not completed yet"
        )
    
    tracking_data_path = status.get("tracking_data_path")
    if not tracking_data_path or not Path(tracking_data_path).exists():
        raise HTTPException(status_code=404, detail="Tracking data not found")
    
    with open(tracking_data_path, 'r') as f:
        tracking_data = json.load(f)
    
    return tracking_data


@app.post("/api/identify-player")
async def identify_player(request: ClickPositionRequest):
    """
    Identify which player was clicked based on position.
    
    Takes normalized x, y coordinates and frame number,
    returns the player ID at that position.
    """
    if request.video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video not found")
    
    status = processing_status[request.video_id]
    if status["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail="Video processing not completed yet"
        )
    
    # Load tracking data
    tracking_data_path = status.get("tracking_data_path")
    with open(tracking_data_path, 'r') as f:
        tracking_data = json.load(f)
    
    # Find player at clicked position
    frame_key = str(request.frame_number)
    if frame_key not in tracking_data.get("frames", {}):
        # Try to find closest frame
        frames = tracking_data.get("frames", {})
        if not frames:
            raise HTTPException(status_code=404, detail="No tracking data available")
        frame_key = min(frames.keys(), key=lambda k: abs(int(k) - request.frame_number))
    
    frame_data = tracking_data["frames"].get(frame_key, {})
    players = frame_data.get("players", [])
    
    # Check which player's bounding box contains the click
    video_width = tracking_data.get("video_width", 1920)
    video_height = tracking_data.get("video_height", 1080)
    
    click_x = request.x * video_width
    click_y = request.y * video_height
    
    selected_player_id = None
    min_distance = float('inf')
    
    for player in players:
        bbox = player.get("bbox", [])
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
            
            # Check if click is inside bounding box
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                # Calculate distance to center for tie-breaking
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                distance = ((click_x - center_x) ** 2 + (click_y - center_y) ** 2) ** 0.5
                
                if distance < min_distance:
                    min_distance = distance
                    selected_player_id = player.get("id")
    
    if selected_player_id is None:
        return {"found": False, "message": "No player found at clicked position"}
    
    return {
        "found": True,
        "player_id": selected_player_id,
        "message": f"Player {selected_player_id} selected"
    }


@app.post("/api/highlight-player")
async def highlight_player(
    background_tasks: BackgroundTasks,
    request: PlayerSelectionRequest
):
    """
    Generate a highlighted video for the selected player.
    
    The selected player will have:
    - Neon glow outline
    - Increased brightness
    - Sharp edges
    
    Other players will have:
    - Heavy Gaussian blur
    - Reduced opacity (60%)
    - White haze effect
    
    Background will have:
    - Medium blur
    """
    if request.video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video not found")
    
    status = processing_status[request.video_id]
    if status["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail="Video processing not completed yet"
        )
    
    # Create a new processing ID for the highlight video
    highlight_id = f"{request.video_id}_highlight_{request.player_id}"
    
    # Initialize highlight processing status
    processing_status[highlight_id] = {
        "status": "processing",
        "progress": 0,
        "message": "Starting highlight generation...",
        "player_id": request.player_id
    }
    
    # Start background processing
    background_tasks.add_task(
        generate_highlight_video_task,
        request.video_id,
        request.player_id,
        highlight_id
    )
    
    return {
        "highlight_id": highlight_id,
        "message": f"Generating highlight video for player {request.player_id}",
        "status": "processing"
    }


async def generate_highlight_video_task(
    video_id: str,
    player_id: int,
    highlight_id: str
):
    """
    Background task for generating the highlighted video.
    
    Applies visual effects:
    - Selected player: glow outline + brightness increase
    - Other players: blur + opacity reduction + haze
    - Background: medium blur
    """
    try:
        status = processing_status[video_id]
        input_path = status["input_path"]
        tracking_data_path = status["tracking_data_path"]
        
        # Load tracking data
        with open(tracking_data_path, 'r') as f:
            tracking_data = json.load(f)
        
        # Output path for highlighted video
        output_path = PROCESSED_DIR / f"{highlight_id}.mp4"
        
        def progress_callback(progress, message):
            processing_status[highlight_id]["progress"] = progress
            processing_status[highlight_id]["message"] = message
        
        # Initialize processor and generate highlight video
        processor = VideoProcessor()
        
        await asyncio.to_thread(
            processor.generate_highlight_video,
            input_path,
            str(output_path),
            tracking_data,
            player_id,
            progress_callback
        )
        
        # Update status
        processing_status[highlight_id].update({
            "status": "completed",
            "progress": 100,
            "message": "Highlight video ready!",
            "video_path": str(output_path),
            "video_url": f"/videos/{highlight_id}.mp4"
        })
        
    except Exception as e:
        processing_status[highlight_id].update({
            "status": "error",
            "message": f"Highlight generation failed: {str(e)}",
            "progress": 0
        })
        print(f"Error generating highlight video: {str(e)}")


@app.get("/api/highlight-status/{highlight_id}")
async def get_highlight_status(highlight_id: str):
    """Get the status of highlight video generation"""
    if highlight_id not in processing_status:
        raise HTTPException(status_code=404, detail="Highlight job not found")
    
    return processing_status[highlight_id]


@app.get("/api/video/{video_id}")
async def get_video(video_id: str):
    """
    Get the processed preview video with tracking overlays.
    """
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video not found")
    
    status = processing_status[video_id]
    video_path = status.get("processed_video_path")
    
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{video_id}_preview.mp4"
    )


@app.get("/api/highlight-video/{highlight_id}")
async def get_highlight_video(highlight_id: str):
    """
    Get the generated highlight video.
    """
    if highlight_id not in processing_status:
        raise HTTPException(status_code=404, detail="Highlight not found")
    
    status = processing_status[highlight_id]
    if status["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail="Highlight video not ready yet"
        )
    
    video_path = status.get("video_path")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"highlight_{highlight_id}.mp4"
    )


@app.delete("/api/video/{video_id}")
async def delete_video(video_id: str):
    """
    Delete a video and all associated data.
    """
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video not found")
    
    status = processing_status[video_id]
    
    # Delete files
    files_to_delete = [
        status.get("input_path"),
        status.get("tracking_data_path"),
        status.get("processed_video_path")
    ]
    
    for file_path in files_to_delete:
        if file_path and Path(file_path).exists():
            Path(file_path).unlink()
    
    # Remove from status
    del processing_status[video_id]
    
    return {"message": "Video deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
