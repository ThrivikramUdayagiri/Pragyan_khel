/**
 * VideoUploader Component
 * 
 * Handles video file upload using drag-and-drop or file selection.
 * Uses react-dropzone for file handling.
 */

import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { Upload, Film, CheckCircle, XCircle } from 'lucide-react';
import '../styles/VideoUploader.css';

const API_BASE_URL = 'http://localhost:8000';

const VideoUploader = ({ onUploadSuccess, onUploadError }) => {
  const [uploadState, setUploadState] = useState('idle'); // idle, uploading, success, error
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedFile, setSelectedFile] = useState(null);

  /**
   * Handle file drop or selection
   */
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setSelectedFile(file);
      setUploadState('idle');
    }
  }, []);

  /**
   * Configure dropzone
   */
  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'video/mp4': ['.mp4'],
      'video/x-msvideo': ['.avi'],
      'video/quicktime': ['.mov'],
      'video/x-matroska': ['.mkv']
    },
    maxFiles: 1,
    multiple: false
  });

  /**
   * Upload the selected file to the server
   */
  const handleUpload = async () => {
    if (!selectedFile) return;

    setUploadState('uploading');
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          setUploadProgress(progress);
        }
      });

      setUploadState('success');
      onUploadSuccess(response.data);
    } catch (error) {
      setUploadState('error');
      onUploadError(error.response?.data || { message: error.message });
    }
  };

  /**
   * Clear selected file and reset state
   */
  const handleClear = () => {
    setSelectedFile(null);
    setUploadState('idle');
    setUploadProgress(0);
  };

  /**
   * Format file size for display
   */
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="video-uploader">
      {/* Dropzone Area */}
      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? 'active' : ''} ${isDragReject ? 'reject' : ''} ${selectedFile ? 'has-file' : ''}`}
      >
        <input {...getInputProps()} />
        
        <div className="dropzone-content">
          {!selectedFile ? (
            <>
              <div className="dropzone-icon">
                <Upload size={48} />
              </div>
              <h3>Drop your cricket video here</h3>
              <p>or click to browse</p>
              <span className="file-types">
                Supports: MP4, AVI, MOV, MKV
              </span>
            </>
          ) : (
            <div className="selected-file-info">
              <Film size={40} className="file-icon" />
              <div className="file-details">
                <h4>{selectedFile.name}</h4>
                <p>{formatFileSize(selectedFile.size)}</p>
              </div>
              {uploadState === 'success' && (
                <CheckCircle size={24} className="status-icon success" />
              )}
              {uploadState === 'error' && (
                <XCircle size={24} className="status-icon error" />
              )}
            </div>
          )}
        </div>
      </div>

      {/* Progress Bar */}
      {uploadState === 'uploading' && (
        <div className="upload-progress">
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
          <span className="progress-text">{uploadProgress}% uploaded</span>
        </div>
      )}

      {/* Action Buttons */}
      {selectedFile && (
        <div className="upload-actions">
          <button
            className="btn btn-primary"
            onClick={handleUpload}
            disabled={uploadState === 'uploading' || uploadState === 'success'}
          >
            {uploadState === 'uploading' ? (
              <>
                <span className="spinner" /> Uploading...
              </>
            ) : uploadState === 'success' ? (
              <>
                <CheckCircle size={18} /> Uploaded!
              </>
            ) : (
              <>
                <Upload size={18} /> Upload Video
              </>
            )}
          </button>
          
          {uploadState !== 'uploading' && uploadState !== 'success' && (
            <button className="btn btn-secondary" onClick={handleClear}>
              Clear
            </button>
          )}
        </div>
      )}

      {/* Error Message */}
      {uploadState === 'error' && (
        <div className="upload-error">
          <XCircle size={18} />
          <span>Upload failed. Please try again.</span>
        </div>
      )}
    </div>
  );
};

export default VideoUploader;
