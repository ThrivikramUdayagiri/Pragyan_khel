package com.example.chronovision

import android.content.Context
import android.content.SharedPreferences

/**
 * Manages app settings using SharedPreferences.
 * Provides easy access to backend configuration and other settings.
 */
class SettingsManager(context: Context) {
    private val prefs: SharedPreferences = context.getSharedPreferences(
        "chronovision_settings",
        Context.MODE_PRIVATE
    )
    
    companion object {
        private const val KEY_BACKEND_IP = "backend_ip"
        private const val KEY_BACKEND_PORT = "backend_port"
        private const val KEY_SAVE_ANALYSIS_HISTORY = "save_analysis_history"
        private const val KEY_AUTO_DOWNLOAD_INTERPOLATED = "auto_download_interpolated"
        private const val KEY_DEFAULT_FPS = "default_fps"
        private const val KEY_DARK_MODE = "dark_mode"
        
        const val DEFAULT_IP = "10.33.231.239"
        const val DEFAULT_PORT = 8000
        const val DEFAULT_FPS = 240
    }
    
    var backendIp: String
        get() = prefs.getString(KEY_BACKEND_IP, DEFAULT_IP) ?: DEFAULT_IP
        set(value) = prefs.edit().putString(KEY_BACKEND_IP, value).apply()
    
    var backendPort: Int
        get() = prefs.getInt(KEY_BACKEND_PORT, DEFAULT_PORT)
        set(value) = prefs.edit().putInt(KEY_BACKEND_PORT, value).apply()
    
    val backendUrl: String
        get() = "http://${backendIp}:${backendPort}/"
    
    var saveAnalysisHistory: Boolean
        get() = prefs.getBoolean(KEY_SAVE_ANALYSIS_HISTORY, true)
        set(value) = prefs.edit().putBoolean(KEY_SAVE_ANALYSIS_HISTORY, value).apply()
    
    var autoDownloadInterpolated: Boolean
        get() = prefs.getBoolean(KEY_AUTO_DOWNLOAD_INTERPOLATED, false)
        set(value) = prefs.edit().putBoolean(KEY_AUTO_DOWNLOAD_INTERPOLATED, value).apply()
    
    var defaultFps: Int
        get() = prefs.getInt(KEY_DEFAULT_FPS, DEFAULT_FPS)
        set(value) = prefs.edit().putInt(KEY_DEFAULT_FPS, value).apply()
    
    var darkMode: Boolean
        get() = prefs.getBoolean(KEY_DARK_MODE, false)
        set(value) = prefs.edit().putBoolean(KEY_DARK_MODE, value).apply()
    
    fun resetToDefaults() {
        prefs.edit().clear().apply()
    }
    
    fun testConnection(onResult: (Boolean, String) -> Unit) {
        // This would make a test request to the backend
        // For now, just validate the URL format
        val url = backendUrl
        val isValid = url.startsWith("http://") && backendIp.isNotEmpty() && backendPort > 0
        onResult(isValid, if (isValid) "URL format valid" else "Invalid URL format")
    }
}
