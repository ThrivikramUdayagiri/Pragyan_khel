/**
 * PlayerSelector Component
 * 
 * Displays a list of detected players and allows selection.
 * Shows player IDs and provides the generate highlight button.
 */

import React, { useMemo, useState } from 'react';
import axios from 'axios';
import { User, Sparkles, Check } from 'lucide-react';
import '../styles/PlayerSelector.css';

const API_BASE_URL = 'http://localhost:8000';

const PlayerSelector = ({
  trackingData,
  selectedPlayerId,
  onPlayerSelect,
  onGenerateHighlight,
  videoId,
  disabled
}) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState(null);

  /**
   * Get unique player IDs from tracking data
   */
  const playerIds = useMemo(() => {
    if (!trackingData) return [];
    return trackingData.unique_ids || [];
  }, [trackingData]);

  /**
   * Get appearance count for each player (how many frames they appear in)
   */
  const playerAppearances = useMemo(() => {
    if (!trackingData || !trackingData.frames) return {};
    
    const appearances = {};
    
    Object.values(trackingData.frames).forEach((frameData) => {
      if (frameData.players) {
        frameData.players.forEach((player) => {
          const id = player.id;
          appearances[id] = (appearances[id] || 0) + 1;
        });
      }
    });
    
    return appearances;
  }, [trackingData]);

  /**
   * Generate color for player ID
   */
  const getPlayerColor = (playerId) => {
    const hue = (playerId * 137.508) % 360;
    return `hsl(${hue}, 70%, 50%)`;
  };

  /**
   * Handle generate highlight button click
   */
  const handleGenerateHighlight = async () => {
    if (!selectedPlayerId || isGenerating) return;

    setIsGenerating(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/highlight-player`, {
        video_id: videoId,
        player_id: selectedPlayerId
      });

      onGenerateHighlight(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to start highlight generation');
      setIsGenerating(false);
    }
  };

  /**
   * Calculate total frames
   */
  const totalFrames = trackingData?.total_frames || 0;

  return (
    <div className="player-selector">
      <h3>Detected Players</h3>
      
      {/* Player List */}
      <div className="player-list">
        {playerIds.length === 0 ? (
          <div className="no-players">
            No players detected in video
          </div>
        ) : (
          playerIds.map((playerId) => {
            const isSelected = playerId === selectedPlayerId;
            const appearances = playerAppearances[playerId] || 0;
            const presencePercent = totalFrames > 0 
              ? Math.round((appearances / totalFrames) * 100) 
              : 0;

            return (
              <div
                key={playerId}
                className={`player-card ${isSelected ? 'selected' : ''}`}
                onClick={() => onPlayerSelect(playerId)}
              >
                <div 
                  className="player-avatar"
                  style={{ borderColor: getPlayerColor(playerId) }}
                >
                  <User size={24} />
                </div>
                
                <div className="player-info">
                  <span className="player-id">Player {playerId}</span>
                  <span className="player-presence">
                    {presencePercent}% presence ({appearances} frames)
                  </span>
                </div>

                {isSelected && (
                  <div className="selected-indicator">
                    <Check size={18} />
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Selection Summary */}
      {selectedPlayerId && (
        <div className="selection-summary">
          <div className="summary-icon">
            <User size={20} style={{ color: getPlayerColor(selectedPlayerId) }} />
          </div>
          <div className="summary-text">
            <strong>Player {selectedPlayerId}</strong> selected
          </div>
        </div>
      )}

      {/* Generate Highlight Button */}
      <button
        className={`btn btn-highlight ${disabled ? 'disabled' : ''}`}
        onClick={handleGenerateHighlight}
        disabled={disabled || isGenerating}
      >
        {isGenerating ? (
          <>
            <span className="spinner" />
            Generating...
          </>
        ) : (
          <>
            <Sparkles size={18} />
            Generate Highlight Video
          </>
        )}
      </button>

      {/* Effect Preview */}
      <div className="effect-preview">
        <h4>Highlight Effects:</h4>
        <ul>
          <li>
            <span className="effect-dot selected" />
            Selected player: Neon glow + sharp edges
          </li>
          <li>
            <span className="effect-dot blurred" />
            Other players: Blurred + haze effect
          </li>
          <li>
            <span className="effect-dot background" />
            Background: Medium blur
          </li>
        </ul>
      </div>

      {/* Error Display */}
      {error && (
        <div className="error-message">
          {error}
        </div>
      )}
    </div>
  );
};

export default PlayerSelector;
