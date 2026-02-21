package com.example.chronovision

import okhttp3.MultipartBody
import okhttp3.RequestBody
import okhttp3.ResponseBody
import retrofit2.Response
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.Query
import retrofit2.http.Streaming

interface TemporalXApiService {
    @Multipart
    @POST("analyze")
    suspend fun analyzeVideo(
        @Part file: MultipartBody.Part,
        @Query("fast") fast: Boolean = true,
        @Query("interpolate") interpolate: Boolean = false,
        @Query("target_fps") targetFps: Int = 240
    ): Response<AnalysisResult>
    
    @Multipart
    @POST("export/csv")
    @Streaming
    suspend fun exportCSV(
        @Part file: MultipartBody.Part,
        @Query("fast") fast: Boolean = true
    ): Response<ResponseBody>
    
    @Multipart
    @POST("export/json")
    suspend fun exportJSON(
        @Part file: MultipartBody.Part,
        @Query("fast") fast: Boolean = true
    ): Response<ResponseBody>
}
