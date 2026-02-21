/**
 * Home Page Component
 * 
 * Main landing page for the Cricket Player Tracker application.
 * This component can be used if routing is added later.
 */

import React from 'react';

const HomePage = ({ children }) => {
  return (
    <div className="home-page">
      {children}
    </div>
  );
};

export default HomePage;
