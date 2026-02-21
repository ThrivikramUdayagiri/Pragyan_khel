package com.example.chronovision

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import androidx.camera.video.*

/**
 * High-speed video recorder for 240 FPS and 120 FPS recording
 */
class HighSpeedVideoRecorder(private val context: Context) {
    
    /**
     * Get appropriate quality and FPS configuration
     */
    fun getQualityConfig(targetFps: Int): Pair<Quality, String> {
        return when (targetFps) {
            240 -> {
                Log.i("HighSpeedRecorder", "Configuring for 240 FPS (720p recommended)")
                Pair(Quality.FHD, "720p @ 240fps")
            }
            120 -> {
                Log.i("HighSpeedRecorder", "Configuring for 120 FPS (1080p possible)")
                Pair(Quality.FHD, "1080p @ 120fps")
            }
            else -> {
                Log.i("HighSpeedRecorder", "Configuring for 60 FPS standard")
                Pair(Quality.HD, "1080p @ 60fps")
            }
        }
    }
    
    /**
     * Log recording start with FPS info
     */
    fun logRecordingStart(fps: Int, cameraCapabilities: CameraCapabilities) {
        Log.d("HighSpeedRecorder", "════════════════════════════════")
        Log.d("HighSpeedRecorder", "🎬 RECORDING STARTED")
        Log.d("HighSpeedRecorder", "Target FPS: $fps")
        Log.d("HighSpeedRecorder", "Device Max FPS: ${cameraCapabilities.getMaxSupportedFps()}")
        Log.d("HighSpeedRecorder", "240 FPS Supported: ${cameraCapabilities.supports240Fps()}")
        Log.d("HighSpeedRecorder", "120 FPS Supported: ${cameraCapabilities.supports120Fps()}")
        Log.d("HighSpeedRecorder", "════════════════════════════════")
    }
    
    /**
     * Calculate expected frame count based on FPS and duration
     */
    fun calculateExpectedFrames(fps: Int, durationSeconds: Int): Int {
        return fps * durationSeconds
    }
}
