package com.example.chronovision

import com.google.gson.annotations.SerializedName

data class FrameClassification(
    @SerializedName("frame_num")
    val frameNum: Int,
    @SerializedName("classification")
    val classification: String,
    @SerializedName("confidence")
    val confidence: Double
)

data class AIInterpolationInfo(
    @SerializedName("enabled")
    val enabled: Boolean,
    @SerializedName("source_fps")
    val sourceFps: Double = 0.0,
    @SerializedName("target_fps")
    val targetFps: Double = 0.0,
    @SerializedName("achieved_fps")
    val achievedFps: Double = 0.0,
    @SerializedName("interpolation_factor")
    val interpolationFactor: Int = 1,
    @SerializedName("output_frames")
    val outputFrames: Int = 0,
    @SerializedName("download_url")
    val downloadUrl: String = ""
)

data class AnalysisResult(
    @SerializedName("total_frames")
    val totalFrames: Int,
    
    @SerializedName("normal")
    val normal: Int,
    
    @SerializedName("drops")
    val drops: Int,
    
    @SerializedName("merges")
    val merges: Int,
    
    @SerializedName("reversals")
    val reversals: Int,
    
    @SerializedName("health_score")
    val healthScore: Double,
    
    @SerializedName("detected_fps")
    val detectedFps: Double,
    
    @SerializedName("expected_fps")
    val expectedFps: Double,
    
    @SerializedName("avg_ssim")
    val avgSsim: Double,
    
    @SerializedName("avg_flow_magnitude")
    val avgFlowMagnitude: Double,
    
    @SerializedName("avg_hist_diff")
    val avgHistDiff: Double,
    
    @SerializedName("video_duration")
    val videoDuration: Double,
    
    @SerializedName("drop_percentage")
    val dropPercentage: Double,
    
    @SerializedName("merge_percentage")
    val mergePercentage: Double,
    
    @SerializedName("reversal_percentage")
    val reversalPercentage: Double,
    
    @SerializedName("timeline")
    val timeline: List<FrameClassification> = emptyList(),
    
    @SerializedName("ai_interpolation")
    val aiInterpolation: AIInterpolationInfo = AIInterpolationInfo(enabled = false)
)


