/**
 * Cricket Video Player Tracking System - Main App Component
 * 
 * This component orchestrates the entire application flow:
 * 1. Video upload
 * 2. Processing status display
 * 3. Video playback with player selection
 * 4. Highlight video generation
 */

import React, { useState, useCallback } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

import VideoUploader from './components/VideoUploader';
import ProcessingStatus from './components/ProcessingStatus';
import VideoPlayer from './components/VideoPlayer';
import PlayerSelector from './components/PlayerSelector';
import HighlightPlayer from './components/HighlightPlayer';

import './styles/App.css';

// Application states
const AppState = {
  UPLOAD: 'upload',
  PROCESSING: 'processing',
  READY: 'ready',
  SELECTING: 'selecting',
  GENERATING: 'generating',
  COMPLETE: 'complete'
};

function App() {
  // Application state management
  const [appState, setAppState] = useState(AppState.UPLOAD);
  const [videoId, setVideoId] = useState(null);
  const [trackingData, setTrackingData] = useState(null);
  const [selectedPlayerId, setSelectedPlayerId] = useState(null);
  const [highlightId, setHighlightId] = useState(null);
  const [previewVideoUrl, setPreviewVideoUrl] = useState(null);
  const [highlightVideoUrl, setHighlightVideoUrl] = useState(null);

  /**
   * Handle successful video upload
   * Transitions to processing state and starts polling for status
   */
  const handleUploadSuccess = useCallback((response) => {
    setVideoId(response.video_id);
    setAppState(AppState.PROCESSING);
    toast.info('Video uploaded! Processing started...', {
      position: 'top-right',
      autoClose: 3000
    });
  }, []);

  /**
   * Handle upload error
   */
  const handleUploadError = useCallback((error) => {
    toast.error(`Upload failed: ${error.message}`, {
      position: 'top-right',
      autoClose: 5000
    });
  }, []);

  /**
   * Handle processing completion
   * Fetches tracking data and transitions to ready state
   */
  const handleProcessingComplete = useCallback((data) => {
    setTrackingData(data.trackingData);
    setPreviewVideoUrl(`http://localhost:8000/api/video/${videoId}`);
    setAppState(AppState.READY);
    toast.success('Video processing complete! Click on a player to highlight.', {
      position: 'top-right',
      autoClose: 5000
    });
  }, [videoId]);

  /**
   * Handle player selection from video click
   * Allows switching between players by clicking on different players
   */
  const handlePlayerSelect = useCallback((playerId) => {
    setSelectedPlayerId(playerId);
    setAppState(AppState.SELECTING);
    toast.success(`Player ${playerId} selected - Click on another player to switch`, {
      position: 'top-right',
      autoClose: 2000
    });
  }, []);

  /**
   * Handle highlight generation start
   */
  const handleHighlightStart = useCallback((highlightResponse) => {
    setHighlightId(highlightResponse.highlight_id);
    setAppState(AppState.GENERATING);
  }, []);

  /**
   * Handle highlight generation completion
   */
  const handleHighlightComplete = useCallback((highlightData) => {
    setHighlightVideoUrl(highlightData.video_url);
    setAppState(AppState.COMPLETE);
    toast.success('Highlight video ready! 🎉', {
      position: 'top-right',
      autoClose: 5000
    });
  }, []);

  /**
   * Reset application to initial state
   */
  const handleReset = useCallback(() => {
    setAppState(AppState.UPLOAD);
    setVideoId(null);
    setTrackingData(null);
    setSelectedPlayerId(null);
    setHighlightId(null);
    setPreviewVideoUrl(null);
    setHighlightVideoUrl(null);
  }, []);

  /**
   * Go back to player selection
   */
  const handleBackToSelection = useCallback(() => {
    setSelectedPlayerId(null);
    setHighlightId(null);
    setHighlightVideoUrl(null);
    setAppState(AppState.READY);
  }, []);

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="logo">
          <span className="logo-icon">🏏</span>
          <h1>Cricket Player Tracker</h1>
        </div>
        <p className="tagline">AI-Powered Video Analysis & Player Highlighting</p>
      </header>

      {/* Main Content */}
      <main className="app-main">
        {/* Upload State */}
        {appState === AppState.UPLOAD && (
          <div className="content-section">
            <h2>Upload Cricket Video</h2>
            <p className="section-description">
              Upload a cricket match video to detect and track players using AI.
              Supported formats: MP4, AVI, MOV, MKV
            </p>
            <VideoUploader
              onUploadSuccess={handleUploadSuccess}
              onUploadError={handleUploadError}
            />
          </div>
        )}

        {/* Processing State */}
        {appState === AppState.PROCESSING && videoId && (
          <div className="content-section">
            <h2>Processing Video</h2>
            <p className="section-description">
              Analyzing video with YOLOv8 segmentation and DeepSORT tracking...
            </p>
            <ProcessingStatus
              videoId={videoId}
              onComplete={handleProcessingComplete}
            />
          </div>
        )}

        {/* Ready State - Video Player with Player Selection */}
        {(appState === AppState.READY || appState === AppState.SELECTING) && (
          <div className="content-section wide">
            <div className="section-header">
              <h2>Select a Player</h2>
              <button className="btn btn-secondary" onClick={handleReset}>
                ← Upload New Video
              </button>
            </div>
            <p className="section-description">
              Click on any player in the video to highlight them.
              All other players and background will be blurred.
            </p>
            
            <div className="player-selection-container">
              <VideoPlayer
                videoUrl={previewVideoUrl}
                trackingData={trackingData}
                onPlayerClick={handlePlayerSelect}
                selectedPlayerId={selectedPlayerId}
              />
              
              {trackingData && (
                <PlayerSelector
                  trackingData={trackingData}
                  selectedPlayerId={selectedPlayerId}
                  onPlayerSelect={handlePlayerSelect}
                  onGenerateHighlight={handleHighlightStart}
                  videoId={videoId}
                  disabled={!selectedPlayerId}
                />
              )}
            </div>
          </div>
        )}

        {/* Generating Highlight State */}
        {appState === AppState.GENERATING && highlightId && (
          <div className="content-section">
            <h2>Generating Highlight Video</h2>
            <p className="section-description">
              Applying visual effects for Player {selectedPlayerId}...
            </p>
            <ProcessingStatus
              videoId={highlightId}
              isHighlight={true}
              onComplete={handleHighlightComplete}
            />
          </div>
        )}

        {/* Complete State - Show Highlight Video */}
        {appState === AppState.COMPLETE && highlightVideoUrl && (
          <div className="content-section wide">
            <div className="section-header">
              <h2>Highlight Video Ready! 🎉</h2>
              <div className="header-actions">
                <button className="btn btn-secondary" onClick={handleBackToSelection}>
                  ← Select Another Player
                </button>
                <button className="btn btn-secondary" onClick={handleReset}>
                  Upload New Video
                </button>
              </div>
            </div>
            <p className="section-description">
              Player {selectedPlayerId} is highlighted with a glowing effect.
              Other players and background are blurred.
            </p>
            <HighlightPlayer
              videoUrl={`http://localhost:8000${highlightVideoUrl}`}
              playerId={selectedPlayerId}
            />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>Cricket Player Tracker - Powered by YOLOv8 & DeepSORT</p>
        <p className="tech-stack">
          React • FastAPI • OpenCV • Deep Learning
        </p>
      </footer>

      {/* Toast Notifications */}
      <ToastContainer
        position="top-right"
        autoClose={5000}
        hideProgressBar={false}
        newestOnTop
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme="dark"
      />
    </div>
  );
}

export default App;
