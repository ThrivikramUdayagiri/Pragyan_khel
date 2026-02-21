package com.example.chronovision

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.util.Range
import android.util.Log

/**
 * Detects actual video FPS capabilities of the device camera
 */
class CameraCapabilities(private val context: Context) {
    private val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    
    data class FpsCapability(val fps: Int, val isSupported: Boolean, val resolution: String = "720p")
    
    /**
     * Get all supported FPS ranges for the rear camera
     */
    fun getSupportedFpsRanges(): List<Range<Int>> {
        return try {
            val cameraId = cameraManager.cameraIdList.firstOrNull() ?: return emptyList()
            val characteristics = cameraManager.getCameraCharacteristics(cameraId)
            
            val availableFpsRanges = characteristics.get(
                CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES
            ) ?: return emptyList()
            
            availableFpsRanges.toList()
        } catch (e: Exception) {
            Log.e("CameraCapabilities", "Error getting FPS ranges: ${e.message}")
            emptyList()
        }
    }
    
    /**
     * Check if 240 FPS is supported
     */
    fun supports240Fps(): Boolean {
        val ranges = getSupportedFpsRanges()
        return ranges.any { range ->
            // Check if range supports 240 FPS (both min and max should be >= 240)
            range.upper >= 240
        }
    }
    
    /**
     * Check if 120 FPS is supported
     */
    fun supports120Fps(): Boolean {
        val ranges = getSupportedFpsRanges()
        return ranges.any { range ->
            range.upper >= 120
        }
    }
    
    /**
     * Get availability of common FPS modes
     */
    fun getAvailableFpsModes(): List<FpsCapability> {
        val modes = mutableListOf<FpsCapability>()
        
        // Check 240 FPS
        if (supports240Fps()) {
            modes.add(FpsCapability(240, true, "720p"))
            Log.d("CameraCapabilities", "✓ 240 FPS is supported")
        } else {
            Log.d("CameraCapabilities", "✗ 240 FPS is NOT supported")
            modes.add(FpsCapability(240, false, "720p"))
        }
        
        // Check 120 FPS
        if (supports120Fps()) {
            modes.add(FpsCapability(120, true))
            Log.d("CameraCapabilities", "✓ 120 FPS is supported")
        } else {
            Log.d("CameraCapabilities", "✗ 120 FPS is NOT supported")
            modes.add(FpsCapability(120, false))
        }
        
        // 60 FPS is almost always supported
        modes.add(FpsCapability(60, true))
        Log.d("CameraCapabilities", "✓ 60 FPS is supported")
        
        return modes
    }
    
    /**
     * Get max FPS the device can actually achieve
     */
    fun getMaxSupportedFps(): Int {
        val ranges = getSupportedFpsRanges()
        if (ranges.isEmpty()) return 30
        
        val maxFps = ranges.maxOfOrNull { it.upper } ?: 30
        Log.d("CameraCapabilities", "Max supported FPS: $maxFps")
        return maxFps
    }
    
    /**
     * Get all FPS range details as string for debugging
     */
    fun getFpsRangesDebugInfo(): String {
        val ranges = getSupportedFpsRanges()
        return ranges.joinToString("\n") { range ->
            "FPS Range: ${range.lower} - ${range.upper}"
        }
    }
}
