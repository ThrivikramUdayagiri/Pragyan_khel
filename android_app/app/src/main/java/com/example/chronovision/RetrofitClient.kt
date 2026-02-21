package com.example.chronovision

import android.content.Context
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

object RetrofitClient {
    private var settingsManager: SettingsManager? = null
    private var retrofit: Retrofit? = null
    
    private val okHttpClient = OkHttpClient.Builder()
        .connectTimeout(180, TimeUnit.SECONDS)  // 3 minutes - for initial connection
        .readTimeout(900, TimeUnit.SECONDS)     // 15 minutes - for AI interpolation processing
        .writeTimeout(300, TimeUnit.SECONDS)    // 5 minutes - for large video upload
        .callTimeout(1200, TimeUnit.SECONDS)    // 20 minutes - total call timeout
        .build()
    
    fun initialize(context: Context) {
        settingsManager = SettingsManager(context.applicationContext)
        retrofit = Retrofit.Builder()
            .baseUrl(settingsManager!!.backendUrl)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
    }
    
    fun updateBaseUrl(context: Context) {
        initialize(context)
    }
    
    val apiService: TemporalXApiService
        get() {
            if (retrofit == null) {
                throw IllegalStateException("RetrofitClient not initialized. Call initialize(context) first.")
            }
            return retrofit!!.create(TemporalXApiService::class.java)
        }
    
    fun getBaseUrl(): String {
        return settingsManager?.backendUrl ?: "http://10.33.231.239:8000/"
    }
}

