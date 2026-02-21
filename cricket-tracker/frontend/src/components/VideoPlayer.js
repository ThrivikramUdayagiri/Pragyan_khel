/**
 * VideoPlayer Component
 * 
 * HTML5 video player with REAL-TIME canvas effects:
 * - Selected player: Clear (no blur)
 * - Transparent ring zone (0-25px from border): adjustable opacity, no blur
 * - Beyond ring: Fully blurred (blur = 0.15)
 * 
 * All effects are applied ON-THE-SPOT as the video plays.
 */

import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Play, Pause, RotateCcw, Volume2, VolumeX } from 'lucide-react';
import '../styles/VideoPlayer.css';

const VideoPlayer = ({ 
  videoUrl, 
  trackingData, 
  onPlayerClick, 
  selectedPlayerId 
}) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const animationFrameRef = useRef(null);
  const offscreenCanvasRef = useRef(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [videoLoaded, setVideoLoaded] = useState(false);
  
  // Real-time effect controls
  const [ringWidth, setRingWidth] = useState(10); // 0-25px transparent ring width
  const [ringOpacity, setRingOpacity] = useState(0.5); // Opacity of the ring area (0-1)
  const [effectEnabled, setEffectEnabled] = useState(true); // Toggle effects on/off
  const [blurIntensity] = useState(0.15); // Fixed blur intensity as requested

  /**
   * Get the frame number from current time
   */
  const getFrameNumber = useCallback((time) => {
    const fps = trackingData?.fps || 30;
    return Math.floor(time * fps);
  }, [trackingData]);

  /**
   * Get player mask polygon points scaled to canvas
   */
  const getPlayerMaskPath = useCallback((player, scaleX, scaleY) => {
    const maskPolygon = player.mask_polygon;
    if (maskPolygon && maskPolygon.length >= 6) {
      const points = [];
      for (let i = 0; i < maskPolygon.length; i += 2) {
        points.push({
          x: maskPolygon[i] * scaleX,
          y: maskPolygon[i + 1] * scaleY
        });
      }
      return points;
    }
    
    // Fallback to bounding box
    const bbox = player.bbox;
    if (bbox && bbox.length >= 4) {
      const x1 = bbox[0] * scaleX;
      const y1 = bbox[1] * scaleY;
      const x2 = bbox[2] * scaleX;
      const y2 = bbox[3] * scaleY;
      return [
        { x: x1, y: y1 },
        { x: x2, y: y1 },
        { x: x2, y: y2 },
        { x: x1, y: y2 }
      ];
    }
    
    return null;
  }, []);

  /**
   * Draw path on context
   */
  const drawPath = useCallback((ctx, points) => {
    if (!points || points.length < 3) return;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.closePath();
  }, []);

  /**
   * Draw real-time effects on canvas
   * - Selected player: Clear (no blur)
   * - Transparent ring (ringWidth px from border): adjustable opacity, no blur
   * - Everything beyond: Fully blurred
   */
  const drawRealTimeEffects = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !trackingData) return;
    if (video.paused && !video.seeking) return;

    const ctx = canvas.getContext('2d');
    const frameNumber = getFrameNumber(video.currentTime);
    setCurrentFrame(frameNumber);

    const width = canvas.width;
    const height = canvas.height;

    // Draw current video frame
    ctx.drawImage(video, 0, 0, width, height);

    // If no player selected or effects disabled, just show video with overlays
    if (!selectedPlayerId || !effectEnabled) {
      drawPlayerOverlays(ctx, frameNumber);
      if (isPlaying) {
        animationFrameRef.current = requestAnimationFrame(drawRealTimeEffects);
      }
      return;
    }

    // Get frame data
    const frameKey = String(frameNumber);
    const frameData = trackingData.frames?.[frameKey];
    
    if (!frameData || !frameData.players) {
      if (isPlaying) {
        animationFrameRef.current = requestAnimationFrame(drawRealTimeEffects);
      }
      return;
    }

    // Calculate scale factor
    const scaleX = width / trackingData.video_width;
    const scaleY = height / trackingData.video_height;

    // Find selected player
    const selectedPlayer = frameData.players.find(p => p.id === selectedPlayerId);
    
    if (!selectedPlayer) {
      // Player not in this frame, apply full blur
      ctx.filter = `blur(${blurIntensity * 50}px)`;
      ctx.drawImage(video, 0, 0, width, height);
      ctx.filter = 'none';
      drawPlayerOverlays(ctx, frameNumber);
      if (isPlaying) {
        animationFrameRef.current = requestAnimationFrame(drawRealTimeEffects);
      }
      return;
    }

    // Get player mask path
    const playerPath = getPlayerMaskPath(selectedPlayer, scaleX, scaleY);
    if (!playerPath) {
      if (isPlaying) {
        animationFrameRef.current = requestAnimationFrame(drawRealTimeEffects);
      }
      return;
    }

    // Create offscreen canvas for blurred version
    if (!offscreenCanvasRef.current) {
      offscreenCanvasRef.current = document.createElement('canvas');
    }
    const offscreen = offscreenCanvasRef.current;
    offscreen.width = width;
    offscreen.height = height;
    const offscreenCtx = offscreen.getContext('2d');

    // Draw blurred version to offscreen canvas using CSS filter
    offscreenCtx.filter = `blur(${blurIntensity * 50}px)`;
    offscreenCtx.drawImage(video, 0, 0, width, height);
    offscreenCtx.filter = 'none';

    // Strategy: Layer from back to front
    // 1. Draw full blurred background
    // 2. Draw clear ring area on top (with opacity control)
    // 3. Draw clear player on top

    // Step 1: Start with blurred background
    ctx.drawImage(offscreen, 0, 0);

    // Step 2: Draw the ring area (clear video between player boundary and ringWidth px out)
    if (ringWidth > 0) {
      ctx.save();
      
      // Create a clipping region for just the ring (expanded area minus player area)
      // We use a path with the expanded boundary, then subtract the player path
      
      // Draw expanded boundary path
      ctx.beginPath();
      
      // First, draw the outer ring boundary (player path expanded by ringWidth)
      // We do this by stroking the path with thick lineWidth
      playerPath.forEach((point, i) => {
        if (i === 0) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
      });
      ctx.closePath();
      ctx.lineWidth = ringWidth * 2;
      ctx.strokeStyle = 'transparent';
      
      // Use clip with stroke to create ring region
      // Save the path for the expanded area
      const expandedPath = new Path2D();
      playerPath.forEach((point, i) => {
        if (i === 0) expandedPath.moveTo(point.x, point.y);
        else expandedPath.lineTo(point.x, point.y);
      });
      expandedPath.closePath();
      
      // Create a mask canvas for the ring
      const ringCanvas = document.createElement('canvas');
      ringCanvas.width = width;
      ringCanvas.height = height;
      const ringCtx = ringCanvas.getContext('2d');
      
      // Draw expanded player area (player + ring) filled
      ringCtx.fillStyle = 'white';
      ringCtx.lineWidth = ringWidth * 2;
      ringCtx.lineJoin = 'round';
      ringCtx.lineCap = 'round';
      
      // Draw and stroke the path to get expanded area
      ringCtx.beginPath();
      playerPath.forEach((point, i) => {
        if (i === 0) ringCtx.moveTo(point.x, point.y);
        else ringCtx.lineTo(point.x, point.y);
      });
      ringCtx.closePath();
      ringCtx.stroke();
      ringCtx.fill();
      
      // Cut out the player area from the ring (leaving just the ring)
      ringCtx.globalCompositeOperation = 'destination-out';
      ringCtx.beginPath();
      playerPath.forEach((point, i) => {
        if (i === 0) ringCtx.moveTo(point.x, point.y);
        else ringCtx.lineTo(point.x, point.y);
      });
      ringCtx.closePath();
      ringCtx.fill();
      
      // Now use ringCanvas as a mask to draw clear video in the ring area
      // The ring area is where ringCanvas is white
      ctx.save();
      ctx.globalAlpha = 1 - ringOpacity; // 0 = fully clear, 1 = fully transparent (showing blur)
      
      // Create clipping from the ring mask
      // Draw clear video only where the ring mask exists
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = width;
      tempCanvas.height = height;
      const tempCtx = tempCanvas.getContext('2d');
      
      // Draw clear video
      tempCtx.drawImage(video, 0, 0, width, height);
      
      // Mask it with the ring
      tempCtx.globalCompositeOperation = 'destination-in';
      tempCtx.drawImage(ringCanvas, 0, 0);
      
      // Draw the masked clear video onto main canvas
      ctx.drawImage(tempCanvas, 0, 0);
      ctx.restore();
      
      ctx.restore();
    }

    // Step 3: Draw the selected player (completely clear)
    ctx.save();
    drawPath(ctx, playerPath);
    ctx.clip();
    ctx.drawImage(video, 0, 0, width, height);
    ctx.restore();

    // Draw player overlays (bounding boxes, labels)
    drawPlayerOverlays(ctx, frameNumber);

    // Continue animation loop
    if (isPlaying) {
      animationFrameRef.current = requestAnimationFrame(drawRealTimeEffects);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trackingData, selectedPlayerId, isPlaying, getFrameNumber, effectEnabled, 
      ringWidth, ringOpacity, blurIntensity, getPlayerMaskPath, drawPath]);

  /**
   * Draw player overlays (bounding boxes, labels) without blur effects
   * NOTE: Box drawing disabled - only tracking data is maintained for player selection
   */
  const drawPlayerOverlays = useCallback((ctx, frameNumber) => {
    // Boxes and labels are not drawn, but the function is kept
    // for player click detection which uses the tracking data
    return;
  }, [trackingData, selectedPlayerId]);

  /**
   * Draw tracking overlays on canvas (legacy - now uses drawRealTimeEffects)
   */
  const drawOverlays = useCallback(() => {
    drawRealTimeEffects();
  }, [drawRealTimeEffects]);

  /**
   * Handle video time update
   */
  const handleTimeUpdate = useCallback(() => {
    const video = videoRef.current;
    if (video) {
      setCurrentTime(video.currentTime);
      if (!isPlaying) {
        drawOverlays();
      }
    }
  }, [isPlaying, drawOverlays]);

  /**
   * Handle video loaded
   */
  const handleLoadedMetadata = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video && canvas) {
      setDuration(video.duration);
      setVideoLoaded(true);

      // Set canvas size to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      drawOverlays();
    }
  }, [drawOverlays]);

  /**
   * Handle canvas click to select player
   * Click on any player to select/switch to them
   */
  const handleCanvasClick = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas || !trackingData) return;
    
    const rect = canvas.getBoundingClientRect();
    
    // Calculate click position relative to canvas display size
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    
    // Scale from display size to canvas internal size
    const displayScaleX = canvas.width / rect.width;
    const displayScaleY = canvas.height / rect.height;
    
    const canvasX = clickX * displayScaleX;
    const canvasY = clickY * displayScaleY;
    
    // Scale from canvas size to original video coordinates
    const videoScaleX = trackingData.video_width / canvas.width;
    const videoScaleY = trackingData.video_height / canvas.height;
    
    const videoX = canvasX * videoScaleX;
    const videoY = canvasY * videoScaleY;

    // Find player at clicked position
    const frameKey = String(currentFrame);
    const frameData = trackingData?.frames?.[frameKey];
    
    if (!frameData || !frameData.players) return;

    // Check each player's bounding box - find the one clicked
    let clickedPlayer = null;
    let minDistance = Infinity;

    frameData.players.forEach((player) => {
      const bbox = player.bbox;
      if (!bbox || bbox.length < 4) return;

      const [x1, y1, x2, y2] = bbox;

      // Check if click is inside bounding box
      if (videoX >= x1 && videoX <= x2 && videoY >= y1 && videoY <= y2) {
        // Calculate distance to center
        const centerX = (x1 + x2) / 2;
        const centerY = (y1 + y2) / 2;
        const distance = Math.sqrt(
          Math.pow(videoX - centerX, 2) + Math.pow(videoY - centerY, 2)
        );

        if (distance < minDistance) {
          minDistance = distance;
          clickedPlayer = player;
        }
      }
    });

    if (clickedPlayer) {
      onPlayerClick(clickedPlayer.id);
    }
  }, [trackingData, currentFrame, onPlayerClick]);

  /**
   * Play/Pause toggle
   */
  const togglePlayPause = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;

    if (video.paused) {
      video.play();
      setIsPlaying(true);
    } else {
      video.pause();
      setIsPlaying(false);
    }
  }, []);

  /**
   * Reset video to start
   */
  const resetVideo = useCallback(() => {
    const video = videoRef.current;
    if (video) {
      video.currentTime = 0;
      video.pause();
      setIsPlaying(false);
    }
  }, []);

  /**
   * Toggle mute
   */
  const toggleMute = useCallback(() => {
    const video = videoRef.current;
    if (video) {
      video.muted = !video.muted;
      setIsMuted(video.muted);
    }
  }, []);

  /**
   * Handle seek
   */
  const handleSeek = useCallback((e) => {
    const video = videoRef.current;
    if (video) {
      const seekTime = (e.target.value / 100) * duration;
      video.currentTime = seekTime;
      setCurrentTime(seekTime);
    }
  }, [duration]);

  /**
   * Start drawing loop when playing
   */
  useEffect(() => {
    if (isPlaying) {
      animationFrameRef.current = requestAnimationFrame(drawOverlays);
    } else {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isPlaying, drawOverlays]);

  /**
   * Format time for display
   */
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="video-player" ref={containerRef}>
      {/* Video Container */}
      <div className="video-container">
        <video
          ref={videoRef}
          src={videoUrl}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          onEnded={() => setIsPlaying(false)}
          muted={isMuted}
          playsInline
          style={{ display: selectedPlayerId && effectEnabled ? 'none' : 'block' }}
        />
        
        {/* Canvas Overlay - Shows real-time effects */}
        <canvas
          ref={canvasRef}
          className="video-overlay"
          onClick={handleCanvasClick}
          style={{ 
            pointerEvents: 'auto',
            display: 'block'
          }}
        />

        {/* Click instruction overlay */}
        {videoLoaded && !selectedPlayerId && (
          <div className="click-instruction">
            Click on a player to select
          </div>
        )}
        
        {/* Switch player instruction when player is selected */}
        {videoLoaded && selectedPlayerId && (
          <div className="click-instruction switch-mode">
            Player {selectedPlayerId} selected — Click another player to switch
          </div>
        )}
      </div>

      {/* Effect Controls - Real-time adjustable sliders */}
      {selectedPlayerId && (
        <div className="effect-controls">
          <div className="effect-header">
            <span className="effect-title">🎯 Real-time Effects</span>
            <button 
              className={`effect-toggle ${effectEnabled ? 'active' : ''}`}
              onClick={() => setEffectEnabled(!effectEnabled)}
              title={effectEnabled ? 'Disable Effects' : 'Enable Effects'}
            >
              {effectEnabled ? 'ON' : 'OFF'}
            </button>
          </div>
          
          <div className="slider-group">
            <div className="slider-row">
              <label className="slider-label">
                <span>Ring Width</span>
                <span className="slider-value">{ringWidth}px</span>
              </label>
              <input
                type="range"
                min="0"
                max="75"
                value={ringWidth}
                onChange={(e) => setRingWidth(parseInt(e.target.value))}
                className="effect-slider"
                disabled={!effectEnabled}
              />
              <div className="slider-hints">
                <span>0px</span>
                <span>75px</span>
              </div>
            </div>
            
            <div className="slider-row">
              <label className="slider-label">
                <span>Ring Opacity</span>
                <span className="slider-value">{Math.round(ringOpacity * 100)}%</span>
              </label>
              <input
                type="range"
                min="0"
                max="100"
                value={ringOpacity * 100}
                onChange={(e) => setRingOpacity(parseInt(e.target.value) / 100)}
                className="effect-slider"
                disabled={!effectEnabled}
              />
              <div className="slider-hints">
                <span>Clear</span>
                <span>Opaque</span>
              </div>
            </div>
          </div>
          
          <div className="effect-info">
            <div className="effect-legend">
              <span className="legend-item player">● Player: Clear</span>
              <span className="legend-item ring">● Ring ({ringWidth}px): {Math.round((1-ringOpacity)*100)}% visible</span>
              <span className="legend-item blur">● Beyond: Blurred</span>
            </div>
          </div>
        </div>
      )}

      {/* Video Controls */}
      <div className="video-controls">
        {/* Play/Pause Button */}
        <button
          className="control-btn"
          onClick={togglePlayPause}
          title={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? <Pause size={20} /> : <Play size={20} />}
        </button>

        {/* Reset Button */}
        <button
          className="control-btn"
          onClick={resetVideo}
          title="Reset"
        >
          <RotateCcw size={18} />
        </button>

        {/* Seek Bar */}
        <div className="seek-container">
          <input
            type="range"
            min="0"
            max="100"
            value={duration ? (currentTime / duration) * 100 : 0}
            onChange={handleSeek}
            className="seek-bar"
          />
        </div>

        {/* Time Display */}
        <span className="time-display">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>

        {/* Frame Counter */}
        <span className="frame-display">
          Frame: {currentFrame}
        </span>

        {/* Mute Button */}
        <button
          className="control-btn"
          onClick={toggleMute}
          title={isMuted ? 'Unmute' : 'Mute'}
        >
          {isMuted ? <VolumeX size={18} /> : <Volume2 size={18} />}
        </button>
      </div>
    </div>
  );
};

export default VideoPlayer;
