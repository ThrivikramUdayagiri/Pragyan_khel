/**
 * HighlightPlayer Component
 * 
 * Video player for displaying the final highlighted video.
 * Includes download functionality and playback controls.
 */

import React, { useRef, useState, useCallback } from 'react';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  Download, 
  Maximize, 
  Volume2, 
  VolumeX 
} from 'lucide-react';
import '../styles/HighlightPlayer.css';

const HighlightPlayer = ({ videoUrl, playerId }) => {
  const videoRef = useRef(null);
  const containerRef = useRef(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [videoLoaded, setVideoLoaded] = useState(false);

  /**
   * Handle video time update
   */
  const handleTimeUpdate = useCallback(() => {
    const video = videoRef.current;
    if (video) {
      setCurrentTime(video.currentTime);
    }
  }, []);

  /**
   * Handle video loaded
   */
  const handleLoadedMetadata = useCallback(() => {
    const video = videoRef.current;
    if (video) {
      setDuration(video.duration);
      setVideoLoaded(true);
    }
  }, []);

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
   * Toggle fullscreen
   */
  const toggleFullscreen = useCallback(async () => {
    const container = containerRef.current;
    if (!container) return;

    try {
      if (!document.fullscreenElement) {
        await container.requestFullscreen();
        setIsFullscreen(true);
      } else {
        await document.exitFullscreen();
        setIsFullscreen(false);
      }
    } catch (err) {
      console.error('Fullscreen error:', err);
    }
  }, []);

  /**
   * Handle fullscreen change
   */
  const handleFullscreenChange = useCallback(() => {
    setIsFullscreen(!!document.fullscreenElement);
  }, []);

  /**
   * Download video
   */
  const handleDownload = useCallback(() => {
    const link = document.createElement('a');
    link.href = videoUrl;
    link.download = `highlight_player_${playerId}.mp4`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, [videoUrl, playerId]);

  /**
   * Format time for display
   */
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Listen for fullscreen changes
  React.useEffect(() => {
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, [handleFullscreenChange]);

  return (
    <div 
      className={`highlight-player ${isFullscreen ? 'fullscreen' : ''}`}
      ref={containerRef}
    >
      {/* Video Title */}
      <div className="highlight-header">
        <div className="highlight-badge">
          <span className="glow-dot" />
          Player {playerId} Highlighted
        </div>
      </div>

      {/* Video Container */}
      <div className="video-wrapper">
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
          onClick={togglePlayPause}
        />

        {/* Play overlay (shown when paused) */}
        {!isPlaying && videoLoaded && (
          <div className="play-overlay" onClick={togglePlayPause}>
            <div className="play-button-large">
              <Play size={48} />
            </div>
          </div>
        )}

        {/* Loading overlay */}
        {!videoLoaded && (
          <div className="loading-overlay">
            <div className="spinner large" />
            <span>Loading video...</span>
          </div>
        )}
      </div>

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

        {/* Mute Button */}
        <button
          className="control-btn"
          onClick={toggleMute}
          title={isMuted ? 'Unmute' : 'Mute'}
        >
          {isMuted ? <VolumeX size={18} /> : <Volume2 size={18} />}
        </button>

        {/* Fullscreen Button */}
        <button
          className="control-btn"
          onClick={toggleFullscreen}
          title="Fullscreen"
        >
          <Maximize size={18} />
        </button>

        {/* Download Button */}
        <button
          className="control-btn download-btn"
          onClick={handleDownload}
          title="Download Video"
        >
          <Download size={18} />
        </button>
      </div>

      {/* Effects Legend */}
      <div className="effects-legend">
        <div className="legend-item">
          <span className="legend-color glow" />
          <span>Selected Player (Glow Effect)</span>
        </div>
        <div className="legend-item">
          <span className="legend-color blur" />
          <span>Other Players (Blurred + Haze)</span>
        </div>
        <div className="legend-item">
          <span className="legend-color bg-blur" />
          <span>Background (Medium Blur)</span>
        </div>
      </div>
    </div>
  );
};

export default HighlightPlayer;
