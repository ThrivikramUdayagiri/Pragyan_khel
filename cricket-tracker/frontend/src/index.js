/**
 * Cricket Video Player Tracking System - Entry Point
 * 
 * This is the main entry point for the React application.
 * Sets up the root component and global providers.
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import './styles/global.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
