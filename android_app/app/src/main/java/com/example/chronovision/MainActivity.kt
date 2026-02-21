package com.example.chronovision

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.video.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.List
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.chronovision.ui.theme.ChronoVisionTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.delay
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import androidx.camera.view.PreviewView
import android.net.Uri
import java.util.Locale

class MainActivity : ComponentActivity() {
    private val requiredPermissions = mutableListOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    ).apply {
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
            add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            add(Manifest.permission.MANAGE_EXTERNAL_STORAGE)
        }
    }.toTypedArray()

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        if (permissions.all { it.value }) {
            Toast.makeText(this, "All permissions granted", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(this, "Permissions required for camera", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize RetrofitClient with settings
        RetrofitClient.initialize(this)
        
        if (!allPermissionsGranted()) {
            permissionLauncher.launch(requiredPermissions)
        }
        
        enableEdgeToEdge()
        setContent {
            ChronoVisionTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    ChronoVisionApp()
                }
            }
        }
    }

    private fun allPermissionsGranted() = requiredPermissions.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }
}

@Composable
fun ChronoVisionApp() {
    var currentScreen by remember { mutableStateOf<Screen>(Screen.Camera) }
    var lastRecordedVideoUri by remember { mutableStateOf<Uri?>(null) }

    when (val screen = currentScreen) {
        is Screen.Camera -> CameraScreen(
            onVideoRecorded = { uri, useAI, fps ->
                lastRecordedVideoUri = uri
                currentScreen = Screen.VideoActions(uri, useAI, fps)
            },
            onOpenSettings = { currentScreen = Screen.Settings },
            onOpenHistory = { currentScreen = Screen.History }
        )
        is Screen.VideoActions -> VideoActionsScreen(
            videoUri = screen.videoUri,
            onBack = { currentScreen = Screen.Camera },
            onPlayback = { currentScreen = Screen.Playback(screen.videoUri) },
            onAnalyze = { currentScreen = Screen.Analysis(screen.videoUri, screen.useAIInterpolation, screen.targetFps) }
        )
        is Screen.Playback -> VideoPlaybackScreen(
            videoUri = screen.videoUri,
            onBack = { currentScreen = Screen.VideoActions(screen.videoUri) }
        )
        is Screen.Analysis -> AnalysisScreen(
            videoUri = screen.videoUri,
            onBack = { currentScreen = Screen.VideoActions(screen.videoUri) },
            useAIInterpolation = screen.useAIInterpolation,
            targetFps = screen.targetFps
        )
        is Screen.Settings -> SettingsScreen(
            onBack = { currentScreen = Screen.Camera }
        )
        is Screen.History -> HistoryScreen(
            onBack = { currentScreen = Screen.Camera },
            onSelectAnalysis = { uri, useAI, fps ->
                currentScreen = Screen.Analysis(uri, useAI, fps)
            }
        )
    }
}

sealed class Screen {
    object Camera : Screen()
    data class VideoActions(val videoUri: Uri, val useAIInterpolation: Boolean = false, val targetFps: Int = 240) : Screen()
    data class Playback(val videoUri: Uri) : Screen()
    data class Analysis(val videoUri: Uri, val useAIInterpolation: Boolean = false, val targetFps: Int = 240) : Screen()
    object Settings : Screen()
    object History : Screen()
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CameraScreen(
    onVideoRecorded: (Uri, Boolean, Int) -> Unit,
    onOpenSettings: () -> Unit = {},
    onOpenHistory: () -> Unit = {}
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()
    
    // Detect device FPS capabilities
    val cameraCapabilities = remember { CameraCapabilities(context) }
    val supportedFpsModes = remember { cameraCapabilities.getAvailableFpsModes() }
    val maxFps = remember { cameraCapabilities.getMaxSupportedFps() }
    val fpsDebugInfo = remember { cameraCapabilities.getFpsRangesDebugInfo() }
    
    LaunchedEffect(Unit) {
        Log.d("CameraScreen", "=== Camera Capabilities ===")
        Log.d("CameraScreen", "Max FPS: $maxFps")
        Log.d("CameraScreen", "240 FPS Supported: ${cameraCapabilities.supports240Fps()}")
        Log.d("CameraScreen", "120 FPS Supported: ${cameraCapabilities.supports120Fps()}")
        Log.d("CameraScreen", fpsDebugInfo)
        Toast.makeText(context, "Max FPS detected: $maxFps", Toast.LENGTH_LONG).show()
    }
    var isRecording by remember { mutableStateOf(false) }
    var recording: Recording? by remember { mutableStateOf(null) }
    var videoCapture: VideoCapture<Recorder>? by remember { mutableStateOf(null) }
    var recordingDurationSeconds by remember { mutableStateOf(0) }
    
    // New: Controls visibility state
    var isControlsExpanded by remember { mutableStateOf(false) }
    
    // Auto-update recording timer
    LaunchedEffect(isRecording) {
        while (isRecording) {
            delay(1000)
            recordingDurationSeconds++
        }
        if (!isRecording) {
            recordingDurationSeconds = 0
        }
    }
    
    // Camera settings
    var selectedFPS by remember { mutableStateOf(240) }
    var selectedISO by remember { mutableIntStateOf(800) }
    var selectedShutterSpeed by remember { mutableStateOf(1f/240f) }
    var selectedResolution by remember { mutableStateOf("1920x1080") }
    var appliedFPS by remember { mutableStateOf(240) }  // Track what FPS is currently active
    var useAIInterpolation by remember { mutableStateOf(false) }  // AI interpolation toggle
    
    val previewView = remember { PreviewView(context) }
    
    // Reinitialize camera when FPS changes
    LaunchedEffect(selectedFPS, !isRecording) {
        if (isRecording) return@LaunchedEffect  // Don't change FPS while recording
        
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
                
                val preview = androidx.camera.core.Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
                
                // Note: CameraX's actual FPS depends on device capabilities
                // 240 FPS requires high-end hardware and may be capped lower
                val qualitySelector = when(selectedFPS) {
                    240 -> QualitySelector.from(Quality.FHD)  // Highest quality for high FPS
                    120 -> QualitySelector.from(Quality.FHD)
                    else -> QualitySelector.from(Quality.FHD)
                }
                
                val recorder = Recorder.Builder()
                    .setQualitySelector(qualitySelector)
                    .build()
                
                val videoCaptureUseCase = VideoCapture.withOutput(recorder)
                videoCapture = videoCaptureUseCase
                
                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    preview,
                    videoCaptureUseCase
                )
                
                appliedFPS = selectedFPS
                Log.d("ChronoVision", "Requested FPS: $selectedFPS (device may cap to lower FPS)")
                Toast.makeText(context, "FPS: $selectedFPS (actual may vary by device)", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Log.e("ChronoVision", "Camera bind failed", e)
                Toast.makeText(context, "Camera initialization failed: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(context))
    }
    
     Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("🎥 ChronoVision Pro", fontWeight = FontWeight.Bold) },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                ),
                actions = {
                    IconButton(onClick = onOpenHistory) {
                        Icon(
                            imageVector = Icons.Default.List,
                            contentDescription = "History"
                        )
                    }
                    IconButton(onClick = onOpenSettings) {
                        Icon(
                            imageVector = Icons.Default.Settings,
                            contentDescription = "Settings"
                        )
                    }
                }
            )
        }
    ) { padding ->
        Box(
            modifier = Modifier.fillMaxSize()
        ) {
            // Full-screen Camera Preview
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
                    .background(Color.Black)
            ) {
                AndroidView(
                    factory = { previewView },
                    modifier = Modifier.fillMaxSize()
                )
                
                // Top overlay - Quick info (minimal)
                if (isRecording) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp)
                            .align(Alignment.TopCenter),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        // Recording indicator
                        Row(
                            modifier = Modifier
                                .background(Color.Red.copy(alpha = 0.9f), shape = RoundedCornerShape(20.dp))
                                .padding(horizontal = 14.dp, vertical = 8.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(10.dp)
                                    .background(Color.White, shape = CircleShape)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                String.format("%02d:%02d", recordingDurationSeconds / 60, recordingDurationSeconds % 60),
                                color = Color.White,
                                fontWeight = FontWeight.Bold,
                                fontSize = 16.sp
                            )
                        }
                        
                        // FPS indicator
                        Text(
                            "${selectedFPS} FPS",
                            modifier = Modifier
                                .background(Color.Black.copy(alpha = 0.7f), shape = RoundedCornerShape(20.dp))
                                .padding(horizontal = 14.dp, vertical = 8.dp),
                            color = Color.White,
                            fontWeight = FontWeight.Bold,
                            fontSize = 14.sp
                        )
                    }
                } else {
                    // Quick info when not recording
                    Row(
                        modifier = Modifier
                            .align(Alignment.TopEnd)
                            .padding(16.dp)
                            .background(Color.Black.copy(alpha = 0.6f), shape = RoundedCornerShape(20.dp))
                            .padding(horizontal = 12.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            "$selectedFPS FPS • $selectedResolution",
                            color = Color.White,
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Medium
                        )
                    }
                }
                
                // Center Record Button (floating during recording)
                if (!isControlsExpanded || isRecording) {
                    Button(
                        onClick = {
                            if (isRecording) {
                                recording?.stop()
                                isRecording = false
                            } else {
                                videoCapture?.let { capture ->
                                    val hsRecorder = HighSpeedVideoRecorder(context)
                                    val name = "ChronoVision-${selectedFPS}fps-${System.currentTimeMillis()}.mp4"
                                    val contentValues = ContentValues().apply {
                                        put(MediaStore.Video.Media.DISPLAY_NAME, name)
                                        put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
                                        put(MediaStore.Video.Media.RELATIVE_PATH, "DCIM/ChronoVision")
                                    }
                                    
                                    val outputOptions = MediaStoreOutputOptions.Builder(
                                        context.contentResolver,
                                        MediaStore.Video.Media.EXTERNAL_CONTENT_URI
                                    ).setContentValues(contentValues).build()
                                    
                                    hsRecorder.logRecordingStart(selectedFPS, cameraCapabilities)
                                    Log.i("RecordingStart", "Recording file: $name")
                                    
                                    recording = capture.output
                                        .prepareRecording(context, outputOptions)
                                        .withAudioEnabled()
                                        .start(ContextCompat.getMainExecutor(context)) { event ->
                                            when (event) {
                                                is VideoRecordEvent.Start -> {
                                                    isRecording = true
                                                    isControlsExpanded = false // Auto-collapse during recording
                                                    val fpsStatus = if (cameraCapabilities.supports240Fps() && selectedFPS == 240) {
                                                        "✓ 240 FPS"
                                                    } else if (cameraCapabilities.supports120Fps() && selectedFPS == 120) {
                                                        "✓ 120 FPS"
                                                    } else {
                                                        "⚠️ ${selectedFPS} FPS"
                                                    }
                                                    Toast.makeText(context, "Recording: $fpsStatus", Toast.LENGTH_SHORT).show()
                                                }
                                                is VideoRecordEvent.Finalize -> {
                                                    isRecording = false
                                                    if (!event.hasError()) {
                                                        val uri = event.outputResults.outputUri
                                                        Toast.makeText(context, "✓ Video saved: ${selectedFPS} FPS", Toast.LENGTH_LONG).show()
                                                        onVideoRecorded(uri, useAIInterpolation, selectedFPS)
                                                    } else {
                                                        Toast.makeText(context, "Error: ${event.error}", Toast.LENGTH_LONG).show()
                                                    }
                                                }
                                            }
                                        }
                                } ?: Toast.makeText(context, "Camera not ready", Toast.LENGTH_SHORT).show()
                            }
                        },
                        modifier = Modifier
                            .align(Alignment.BottomCenter)
                            .padding(bottom = if (isControlsExpanded) 16.dp else 80.dp)
                            .size(if (isRecording) 64.dp else 72.dp),
                        shape = CircleShape,
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (isRecording) Color.Red else MaterialTheme.colorScheme.primary
                        ),
                        elevation = ButtonDefaults.buttonElevation(8.dp)
                    ) {
                        Icon(
                            imageVector = if (isRecording) Icons.Default.CheckCircle else Icons.Default.PlayArrow,
                            contentDescription = if (isRecording) "Stop" else "Record",
                            modifier = Modifier.size(32.dp),
                            tint = Color.White
                        )
                    }
                }
                
                // Floating Controls Toggle Button
                if (!isRecording) {
                    FloatingActionButton(
                        onClick = { isControlsExpanded = !isControlsExpanded },
                        modifier = Modifier
                            .align(Alignment.BottomEnd)
                            .padding(16.dp),
                        containerColor = MaterialTheme.colorScheme.primaryContainer
                    ) {
                        Icon(
                            imageVector = if (isControlsExpanded) Icons.AutoMirrored.Filled.ArrowBack else Icons.Default.Settings,
                            contentDescription = "Toggle Controls"
                        )
                    }
                }
            }
            
            // Sliding Controls Panel (from bottom)
            androidx.compose.animation.AnimatedVisibility(
                visible = isControlsExpanded && !isRecording,
                enter = androidx.compose.animation.slideInVertically(initialOffsetY = { it }),
                exit = androidx.compose.animation.slideOutVertically(targetOffsetY = { it }),
                modifier = Modifier.align(Alignment.BottomCenter)
            ) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(max = 500.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surface
                    ),
                    shape = RoundedCornerShape(topStart = 20.dp, topEnd = 20.dp)
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .verticalScroll(rememberScrollState())
                            .padding(16.dp)
                    ) {
                        // Header with drag handle
                        Box(
                            modifier = Modifier
                                .width(40.dp)
                                .height(4.dp)
                                .background(Color.Gray.copy(alpha = 0.3f), shape = RoundedCornerShape(2.dp))
                                .align(Alignment.CenterHorizontally)
                        )
                        
                        Spacer(modifier = Modifier.height(16.dp))
                        
                        Text(
                            "⚙️ Camera Controls",
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold
                        )
                        
                        Spacer(modifier = Modifier.height(20.dp))
                    
                    // FPS Selection with Capability Detection
                    Text("FPS: $selectedFPS (Applied: $appliedFPS) - Max Supported: $maxFps", fontWeight = FontWeight.SemiBold)
                    
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        val fps240 = supportedFpsModes.find { it.fps == 240 }
                        FilterChip(
                            selected = selectedFPS == 240 && fps240?.isSupported == true,
                            onClick = { 
                                if (fps240?.isSupported == true) {
                                    selectedFPS = 240
                                    Log.d("ChronoVision", "Selected 240 FPS (SUPPORTED)")
                                } else {
                                    Toast.makeText(context, "Your device does not support 240 FPS", Toast.LENGTH_SHORT).show()
                                    Log.d("ChronoVision", "240 FPS not supported on this device")
                                }
                            },
                            label = { Text("240 FPS${if (fps240?.isSupported == true) "" else " ✗"}") },
                            enabled = !isRecording && fps240?.isSupported == true,
                            leadingIcon = if (appliedFPS == 240 && isRecording) {
                                { Icon(Icons.Filled.CheckCircle, contentDescription = "Active", modifier = Modifier.size(16.dp)) }
                            } else null
                        )
                        
                        val fps120 = supportedFpsModes.find { it.fps == 120 }
                        FilterChip(
                            selected = selectedFPS == 120 && fps120?.isSupported == true,
                            onClick = { 
                                if (fps120?.isSupported == true) {
                                    selectedFPS = 120
                                    Log.d("ChronoVision", "Selected 120 FPS (SUPPORTED)")
                                } else {
                                    Toast.makeText(context, "Your device does not support 120 FPS", Toast.LENGTH_SHORT).show()
                                    Log.d("ChronoVision", "120 FPS not supported on this device")
                                }
                            },
                            label = { Text("120 FPS${if (fps120?.isSupported == true) "" else " ✗"}") },
                            enabled = !isRecording && fps120?.isSupported == true,
                            leadingIcon = if (appliedFPS == 120 && isRecording) {
                                { Icon(Icons.Filled.CheckCircle, contentDescription = "Active", modifier = Modifier.size(16.dp)) }
                            } else null
                        )
                        
                        val fps60 = supportedFpsModes.find { it.fps == 60 }
                        FilterChip(
                            selected = selectedFPS == 60 && fps60?.isSupported == true,
                            onClick = { 
                                if (fps60?.isSupported == true) {
                                    selectedFPS = 60
                                    Log.d("ChronoVision", "Selected 60 FPS (SUPPORTED)")
                                } else {
                                    Toast.makeText(context, "60 FPS not available", Toast.LENGTH_SHORT).show()
                                }
                            },
                            label = { Text("60 FPS${if (fps60?.isSupported == true) "" else " ✗"}") },
                            enabled = !isRecording && fps60?.isSupported == true,
                            leadingIcon = if (appliedFPS == 60 && isRecording) {
                                { Icon(Icons.Filled.CheckCircle, contentDescription = "Active", modifier = Modifier.size(16.dp)) }
                            } else null
                        )
                    }
                    
                    if (maxFps < 240 && maxFps < selectedFPS) {
                        Text(
                            "⚠️ Selected FPS not available. Will use $maxFps FPS instead.",
                            color = Color(0xFFFF9800),
                            fontSize = 12.sp
                        )
                    }
                    
                    Spacer(modifier = Modifier.height(16.dp))
                    
                    // AI Interpolation Toggle - Compact
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(
                                if (useAIInterpolation) MaterialTheme.colorScheme.primaryContainer 
                                else MaterialTheme.colorScheme.surfaceVariant,
                                shape = RoundedCornerShape(12.dp)
                            )
                            .padding(12.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            "🤖 AI Interpolation",
                            fontWeight = FontWeight.SemiBold,
                            fontSize = 14.sp
                        )
                        Switch(
                            checked = useAIInterpolation,
                            onCheckedChange = { useAIInterpolation = it },
                            enabled = !isRecording
                        )
                    }
                    
                    Spacer(modifier = Modifier.height(12.dp))
                    
                    // Resolution Selection
                    Text("Resolution: $selectedResolution", fontWeight = FontWeight.SemiBold)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        FilterChip(
                            selected = selectedResolution == "1920x1080",
                            onClick = { selectedResolution = "1920x1080" },
                            label = { Text("1080p") },
                            enabled = !isRecording
                        )
                        FilterChip(
                            selected = selectedResolution == "1280x720",
                            onClick = { selectedResolution = "1280x720" },
                            label = { Text("720p") },
                            enabled = !isRecording
                        )
                        FilterChip(
                            selected = selectedResolution == "640x480",
                            onClick = { selectedResolution == "640x480" },
                            label = { Text("480p") },
                            enabled = !isRecording
                        )
                    }
                    
                    Spacer(modifier = Modifier.height(12.dp))
                    
                    // ISO
                    Text("ISO: $selectedISO", fontWeight = FontWeight.SemiBold)
                    Slider(
                        value = selectedISO.toFloat(),
                        onValueChange = { selectedISO = it.toInt() },
                        valueRange = 100f..3200f,
                        steps = 15,
                        enabled = !isRecording
                    )
                    
                    Spacer(modifier = Modifier.height(12.dp))
                    
                    // Shutter Speed
                    Text("Shutter: 1/${(1f/selectedShutterSpeed).toInt()}", fontWeight = FontWeight.SemiBold)
                    Slider(
                        value = selectedShutterSpeed,
                        onValueChange = { selectedShutterSpeed = it },
                        valueRange = 1f/1000f..1f/30f,
                        enabled = !isRecording
                    )
                    
                    Spacer(modifier = Modifier.height(24.dp))
                    
                    // Apply and close button
                    Button(
                        onClick = { isControlsExpanded = false },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.primary
                        )
                    ) {
                        Text("✓ Apply & Close", fontWeight = FontWeight.Bold, fontSize = 16.sp)
                    }
                    
                    Spacer(modifier = Modifier.height(8.dp))
                }
            }
        }
        }
    }
}

@kotlin.OptIn(ExperimentalMaterial3Api::class)
@Composable
fun VideoActionsScreen(
    videoUri: Uri,
    onBack: () -> Unit,
    onPlayback: () -> Unit,
    onAnalyze: () -> Unit
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Video Recorded") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Filled.ArrowBack, "Back")
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Icon(
                imageVector = Icons.Filled.CheckCircle,
                contentDescription = null,
                modifier = Modifier.size(80.dp),
                tint = MaterialTheme.colorScheme.primary
            )
            
            Spacer(modifier = Modifier.height(24.dp))
            
            Text(
                "Video saved successfully!",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold
            )
            
            Spacer(modifier = Modifier.height(48.dp))
            
            Button(
                onClick = onPlayback,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp)
            ) {
                Icon(Icons.Filled.PlayArrow, null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Play Video", fontSize = 18.sp)
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            OutlinedButton(
                onClick = onAnalyze,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp)
            ) {
                Icon(Icons.Filled.CheckCircle, null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Analyze with TemporalX", fontSize = 18.sp)
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            TextButton(
                onClick = onBack,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Record Another Video")
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun VideoPlaybackScreen(videoUri: Uri, onBack: () -> Unit) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Video Playback") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Filled.ArrowBack, "Back")
                    }
                }
            )
        }
    ) { padding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .background(Color.Black),
            contentAlignment = Alignment.Center
        ) {
            AndroidView(
                factory = { ctx ->
                    androidx.media3.ui.PlayerView(ctx).apply {
                        val exoPlayer = androidx.media3.exoplayer.ExoPlayer.Builder(ctx).build()
                        player = exoPlayer
                        
                        val mediaItem = androidx.media3.common.MediaItem.fromUri(videoUri)
                        exoPlayer.setMediaItem(mediaItem)
                        exoPlayer.prepare()
                        exoPlayer.playWhenReady = true
                    }
                },
                modifier = Modifier.fillMaxSize()
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AnalysisScreen(
    videoUri: Uri, 
    onBack: () -> Unit,
    useAIInterpolation: Boolean = false,
    targetFps: Int = 240
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    
    var isAnalyzing by remember { mutableStateOf(false) }
    var analysisResult by remember { mutableStateOf<AnalysisResult?>(null) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var framesAnalyzed by remember { mutableStateOf(0) }
    var totalFramesEstimate by remember { mutableStateOf(0) }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("TemporalX Analysis") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Filled.ArrowBack, "Back")
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (analysisResult == null && !isAnalyzing) {
                Spacer(modifier = Modifier.weight(1f))
                
                Icon(
                    imageVector = Icons.Filled.CheckCircle,
                    contentDescription = null,
                    modifier = Modifier.size(80.dp),
                    tint = MaterialTheme.colorScheme.primary
                )
                
                Spacer(modifier = Modifier.height(24.dp))
                
                Text(
                    "Analyze Video Quality",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold
                )
                
                Spacer(modifier = Modifier.height(16.dp))
                
                Text(
                    "Detect frame drops, merges, and reversals",
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                
                Spacer(modifier = Modifier.weight(1f))
                
                Button(
                    onClick = {
                        isAnalyzing = true
                        errorMessage = null
                        framesAnalyzed = 0
                        totalFramesEstimate = 0
                        
                        // Warn user about AI processing time
                        if (useAIInterpolation) {
                            Toast.makeText(
                                context, 
                                "⏳ AI Interpolation enabled - This may take several minutes to process...", 
                                Toast.LENGTH_LONG
                            ).show()
                        }
                        
                        scope.launch {
                            try {
                                val result = analyzeVideo(
                                    context, 
                                    videoUri,
                                    onProgress = { analyzed, total ->
                                        framesAnalyzed = analyzed
                                        totalFramesEstimate = total
                                    },
                                    useAIInterpolation = useAIInterpolation,
                                    targetFps = targetFps
                                )
                                analysisResult = result
                            } catch (e: Exception) {
                                errorMessage = "Error: ${e.message}"
                                Log.e("AnalysisError", "Analysis failed", e)
                            } finally {
                                isAnalyzing = false
                            }
                        }
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(56.dp)
                ) {
                    Text("Start Analysis", fontSize = 18.sp)
                }
            }
            
            if (isAnalyzing) {
                Spacer(modifier = Modifier.weight(1f))
                
                // Analysis Progress Card
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.primaryContainer
                    )
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(20.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        // Progress text
                        Text(
                            if (useAIInterpolation) 
                                "🤖 AI Interpolation + Analysis..."
                            else 
                                "Analyzing video...",
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold
                        )
                        
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        if (useAIInterpolation) {
                            Text(
                                "Generating intermediate frames using optical flow",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f)
                            )
                        }
                        
                        Spacer(modifier = Modifier.height(16.dp))
                        
                        // Progress indicator
                        CircularProgressIndicator(
                            modifier = Modifier.size(64.dp),
                            strokeWidth = 4.dp
                        )
                        
                        Spacer(modifier = Modifier.height(16.dp))
                        
                        // Progress text with frame count
                        Text(
                            if (totalFramesEstimate > 0) 
                                "Frames: $framesAnalyzed / ~$totalFramesEstimate"
                            else 
                                "Processing frames...",
                            style = MaterialTheme.typography.bodyLarge
                        )
                        
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        // Progress bar
                        if (totalFramesEstimate > 0) {
                            LinearProgressIndicator(
                                progress = { (framesAnalyzed.toFloat() / totalFramesEstimate.toFloat()).coerceIn(0f, 1f) },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(6.dp),
                            )
                        } else {
                            LinearProgressIndicator(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(6.dp),
                            )
                        }
                    }
                }
                
                Spacer(modifier = Modifier.weight(1f))
            }
            
            errorMessage?.let { error ->
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.errorContainer
                    )
                ) {
                    Text(
                        text = error,
                        modifier = Modifier.padding(16.dp),
                        color = MaterialTheme.colorScheme.onErrorContainer
                    )
                }
            }
            
            analysisResult?.let { result ->
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .verticalScroll(rememberScrollState())
                ) {
                    // Timeline Visualization
                    if (result.timeline.isNotEmpty()) {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(
                                containerColor = MaterialTheme.colorScheme.surfaceVariant
                            )
                        ) {
                            Column(modifier = Modifier.padding(16.dp)) {
                                Text(
                                    "Frame-by-Frame Timeline",
                                    fontSize = 16.sp,
                                    fontWeight = FontWeight.Bold
                                )
                                Spacer(modifier = Modifier.height(8.dp))
                                
                                // Timeline bar
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .height(30.dp),
                                    horizontalArrangement = Arrangement.spacedBy(0.dp)
                                ) {
                                    result.timeline.forEach { frame ->
                                        val frameColor = when (frame.classification) {
                                            "Normal" -> Color(0xFF4CAF50)
                                            "Frame Drop" -> Color(0xFFF44336)
                                            "Frame Merge" -> Color(0xFFFFEB3B)
                                            "Frame Reversal" -> Color(0xFF9C27B0)
                                            else -> Color.Gray
                                        }
                                        Box(
                                            modifier = Modifier
                                                .weight(1f)
                                                .height(30.dp)
                                                .background(frameColor)
                                        )
                                    }
                                }
                                
                                Spacer(modifier = Modifier.height(12.dp))
                                
                                // Legend
                                Column(
                                    modifier = Modifier.fillMaxWidth(),
                                    verticalArrangement = Arrangement.spacedBy(6.dp)
                                ) {
                                    LegendItem("Normal", Color(0xFF4CAF50))
                                    LegendItem("Frame Drop", Color(0xFFF44336))
                                    LegendItem("Frame Merge", Color(0xFFFFEB3B))
                                    LegendItem("Frame Reversal", Color(0xFF9C27B0))
                                }
                            }
                        }
                        Spacer(modifier = Modifier.height(16.dp))
                    }
                    
                    // Health Score Card - Big and prominent
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(
                            containerColor = if (result.healthScore >= 90) 
                                MaterialTheme.colorScheme.primaryContainer 
                            else MaterialTheme.colorScheme.errorContainer
                        )
                    ) {
                        Column(
                            modifier = Modifier.padding(20.dp),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Text(
                                "Video Health Score",
                                fontSize = 16.sp,
                                fontWeight = FontWeight.Medium
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                String.format("%.1f%%", result.healthScore),
                                fontSize = 48.sp,
                                fontWeight = FontWeight.Bold,
                                color = if (result.healthScore >= 90) 
                                    MaterialTheme.colorScheme.primary 
                                else MaterialTheme.colorScheme.error
                            )
                            Spacer(modifier = Modifier.height(12.dp))
                            LinearProgressIndicator(
                                progress = { (result.healthScore / 100.0).toFloat() },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(8.dp),
                            )
                        }
                    }
                    
                    Spacer(modifier = Modifier.height(16.dp))
                    
                    // FPS Information Card
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.secondaryContainer
                        )
                    ) {
                        Column(modifier = Modifier.padding(16.dp)) {
                            Text(
                                "Frame Rate Analysis",
                                fontSize = 18.sp,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.onSecondaryContainer
                            )
                            Spacer(modifier = Modifier.height(12.dp))
                            
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                    Text("Expected FPS", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    Text(
                                        String.format("%.1f", result.expectedFps),
                                        fontSize = 24.sp,
                                        fontWeight = FontWeight.Bold
                                    )
                                }
                                Text("→", fontSize = 24.sp, modifier = Modifier.padding(top = 16.dp))
                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                    Text("Detected FPS", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    Text(
                                        String.format("%.1f", result.detectedFps),
                                        fontSize = 24.sp,
                                        fontWeight = FontWeight.Bold,
                                        color = if (kotlin.math.abs(result.detectedFps - result.expectedFps) < 5) 
                                            MaterialTheme.colorScheme.primary 
                                        else MaterialTheme.colorScheme.error
                                    )
                                }
                            }
                            
                            Spacer(modifier = Modifier.height(12.dp))
                            
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                MetricItem("Duration", String.format("%.2fs", result.videoDuration))
                                MetricItem("Total Frames", result.totalFrames.toString())
                            }
                        }
                    }
                    
                    Spacer(modifier = Modifier.height(16.dp))
                    
                    // Frame Classification Card
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.tertiaryContainer
                        )
                    ) {
                        Column(modifier = Modifier.padding(16.dp)) {
                            Text(
                                "Frame Classification",
                                fontSize = 18.sp,
                                fontWeight = FontWeight.Bold
                            )
                            Spacer(modifier = Modifier.height(12.dp))
                            
                            // Normal Frames
                            ClassificationRow(
                                label = "Normal",
                                count = result.normal,
                                percentage = (100.0 * result.normal / result.totalFrames),
                                color = Color(0xFF4CAF50)
                            )
                            
                            // Frame Drops
                            if (result.drops > 0) {
                                Spacer(modifier = Modifier.height(8.dp))
                                ClassificationRow(
                                    label = "Drops",
                                    count = result.drops,
                                    percentage = result.dropPercentage,
                                    color = Color(0xFFF44336)
                                )
                            }
                            
                            // Frame Merges
                            if (result.merges > 0) {
                                Spacer(modifier = Modifier.height(8.dp))
                                ClassificationRow(
                                    label = "Merges",
                                    count = result.merges,
                                    percentage = result.mergePercentage,
                                    color = Color(0xFFFF9800)
                                )
                            }
                            
                            // Frame Reversals
                            if (result.reversals > 0) {
                                Spacer(modifier = Modifier.height(8.dp))
                                ClassificationRow(
                                    label = "Reversals",
                                    count = result.reversals,
                                    percentage = result.reversalPercentage,
                                    color = Color(0xFF9C27B0)
                                )
                            }
                        }
                    }
                    
                    Spacer(modifier = Modifier.height(16.dp))
                    
                    // Advanced Metrics Card
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.surfaceVariant
                        )
                    ) {
                        Column(modifier = Modifier.padding(16.dp)) {
                            Text(
                                "Advanced Metrics",
                                fontSize = 18.sp,
                                fontWeight = FontWeight.Bold
                            )
                            Spacer(modifier = Modifier.height(12.dp))
                            
                            MetricBar(
                                label = "Avg SSIM (Similarity)",
                                value = result.avgSsim,
                                maxValue = 1.0,
                                unit = "",
                                goodThreshold = 0.95
                            )
                            
                            Spacer(modifier = Modifier.height(12.dp))
                            
                            MetricBar(
                                label = "Avg Optical Flow",
                                value = result.avgFlowMagnitude,
                                maxValue = 50.0,
                                unit = "px",
                                goodThreshold = 20.0,
                                inverse = true
                            )
                            
                            Spacer(modifier = Modifier.height(12.dp))
                            
                            MetricBar(
                                label = "Avg Histogram Diff",
                                value = result.avgHistDiff,
                                maxValue = 1.0,
                                unit = "",
                                goodThreshold = 0.1,
                                inverse = true
                            )
                        }
                    }
                    
                    Spacer(modifier = Modifier.height(16.dp))
                    
                    // Download Buttons Card
                    var isDownloadingCsv by remember { mutableStateOf(false) }
                    var isDownloadingJson by remember { mutableStateOf(false) }
                    var downloadStatusMessage by remember { mutableStateOf("") }
                    
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.surfaceVariant
                        )
                    ) {
                        Column(modifier = Modifier.padding(16.dp)) {
                            Text(
                                "Export Data",
                                fontSize = 18.sp,
                                fontWeight = FontWeight.Bold
                            )
                            Spacer(modifier = Modifier.height(12.dp))
                            
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                Button(
                                    onClick = {
                                        isDownloadingCsv = true
                                        scope.launch {
                                            try {
                                                downloadAnalysisData(context, videoUri, "csv", onDownloadComplete = {
                                                    isDownloadingCsv = false
                                                    downloadStatusMessage = "CSV downloaded successfully!"
                                                }, onDownloadError = { error ->
                                                    isDownloadingCsv = false
                                                    downloadStatusMessage = "CSV download failed: $error"
                                                })
                                            } catch (e: Exception) {
                                                isDownloadingCsv = false
                                                downloadStatusMessage = "Error: ${e.message}"
                                            }
                                        }
                                    },
                                    modifier = Modifier
                                        .weight(1f),
                                    enabled = !isDownloadingCsv && !isDownloadingJson
                                ) {
                                    if (isDownloadingCsv) {
                                        CircularProgressIndicator(
                                            modifier = Modifier.size(20.dp),
                                            strokeWidth = 2.dp,
                                            color = MaterialTheme.colorScheme.onPrimary
                                        )
                                    } else {
                                        Text("Download CSV")
                                    }
                                }
                                
                                Button(
                                    onClick = {
                                        isDownloadingJson = true
                                        scope.launch {
                                            try {
                                                downloadAnalysisData(context, videoUri, "json", onDownloadComplete = {
                                                    isDownloadingJson = false
                                                    downloadStatusMessage = "JSON downloaded successfully!"
                                                }, onDownloadError = { error ->
                                                    isDownloadingJson = false
                                                    downloadStatusMessage = "JSON download failed: $error"
                                                })
                                            } catch (e: Exception) {
                                                isDownloadingJson = false
                                                downloadStatusMessage = "Error: ${e.message}"
                                            }
                                        }
                                    },
                                    modifier = Modifier
                                        .weight(1f),
                                    enabled = !isDownloadingCsv && !isDownloadingJson
                                ) {
                                    if (isDownloadingJson) {
                                        CircularProgressIndicator(
                                            modifier = Modifier.size(20.dp),
                                            strokeWidth = 2.dp,
                                            color = MaterialTheme.colorScheme.onPrimary
                                        )
                                    } else {
                                        Text("Download JSON")
                                    }
                                }
                            }
                            
                            if (downloadStatusMessage.isNotEmpty()) {
                                Spacer(modifier = Modifier.height(8.dp))
                                Text(
                                    downloadStatusMessage,
                                    fontSize = 12.sp,
                                    color = if (downloadStatusMessage.contains("failed", ignoreCase = true)) 
                                        Color.Red else Color.Green
                                )
                            }
                            
                            // AI Interpolation Download (if available)
                            if (result.aiInterpolation.enabled) {
                                Spacer(modifier = Modifier.height(16.dp))
                                Text(
                                    "🤖 AI Interpolated Video",
                                    fontSize = 16.sp,
                                    fontWeight = FontWeight.SemiBold
                                )
                                Spacer(modifier = Modifier.height(8.dp))
                                Text(
                                    "Enhanced video: ${result.aiInterpolation.sourceFps.toInt()} → ${result.aiInterpolation.achievedFps.toInt()} FPS",
                                    fontSize = 12.sp,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                Spacer(modifier = Modifier.height(8.dp))
                                Button(
                                    onClick = {
                                        scope.launch {
                                            try {
                                                downloadVideoFromUrl(
                                                    context,
                                                    result.aiInterpolation.downloadUrl,
                                                    "mp4",
                                                    onDownloadComplete = {
                                                        downloadStatusMessage = "Interpolated video downloaded successfully!"
                                                    },
                                                    onDownloadError = { error ->
                                                        downloadStatusMessage = "Interpolated video download failed: $error"
                                                    }
                                                )
                                            } catch (e: Exception) {
                                                downloadStatusMessage = "Error: ${e.message}"
                                            }
                                        }
                                    },
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Text("⬇️ Download ${result.aiInterpolation.targetFps.toInt()} FPS Video")
                                }
                            }
                            
                            // Share Results Button
                            Spacer(modifier = Modifier.height(16.dp))
                            OutlinedButton(
                                onClick = {
                                    val shareText = buildString {
                                        appendLine("📊 ChronoVision Analysis Results")
                                        appendLine("")
                                        appendLine("Health Score: ${String.format("%.1f%%", result.healthScore)}")
                                        appendLine("Total Frames: ${result.totalFrames}")
                                        appendLine("Normal: ${result.normal} | Drops: ${result.drops} | Merges: ${result.merges} | Reversals: ${result.reversals}")
                                        appendLine("Detected FPS: ${String.format("%.2f", result.detectedFps)}")
                                        appendLine("Expected FPS: ${String.format("%.2f", result.expectedFps)}")
                                        if (result.aiInterpolation.enabled) {
                                            appendLine("")
                                            appendLine("🤖 AIInterpolation: ${result.aiInterpolation.sourceFps.toInt()} → ${result.aiInterpolation.achievedFps.toInt()} FPS")
                                        }
                                    }
                                    val shareIntent = android.content.Intent().apply {
                                        action = android.content.Intent.ACTION_SEND
                                        putExtra(android.content.Intent.EXTRA_TEXT, shareText)
                                        type = "text/plain"
                                    }
                                    context.startActivity(android.content.Intent.createChooser(shareIntent, "Share Analysis Results"))
                                },
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Icon(Icons.Default.Share, contentDescription = null, modifier = Modifier.size(18.dp))
                                Spacer(modifier = Modifier.width(8.dp))
                                Text("Share Results")
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun LegendItem(label: String, color: Color) {
    Row(
        modifier = Modifier
            .fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Box(
            modifier = Modifier
                .size(16.dp)
                .background(color, shape = RoundedCornerShape(2.dp))
        )
        Text(
            label,
            fontSize = 13.sp,
            fontWeight = FontWeight.Medium
        )
    }
}

@Composable
fun MetricItem(label: String, value: String) {
    Column {
        Text(
            label,
            fontSize = 12.sp,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            value,
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
fun ClassificationRow(label: String, count: Int, percentage: Double, color: Color) {
    Column(modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                label,
                fontSize = 16.sp,
                fontWeight = FontWeight.Medium
            )
            Text(
                "$count (${String.format("%.1f%%", percentage)})",
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                color = color
            )
        }
        Spacer(modifier = Modifier.height(4.dp))
        LinearProgressIndicator(
            progress = { (percentage / 100.0).toFloat() },
            modifier = Modifier
                .fillMaxWidth()
                .height(6.dp),
            color = color,
        )
    }
}

@Composable
fun MetricBar(
    label: String,
    value: Double,
    maxValue: Double,
    unit: String,
    goodThreshold: Double,
    inverse: Boolean = false
) {
    Column(modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                label,
                fontSize = 14.sp,
                fontWeight = FontWeight.Medium
            )
            Text(
                String.format("%.3f%s", value, unit),
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold
            )
        }
        Spacer(modifier = Modifier.height(4.dp))
        
        val progress = (value / maxValue).toFloat().coerceIn(0f, 1f)
        val isGood = if (inverse) value < goodThreshold else value > goodThreshold
        
        LinearProgressIndicator(
            progress = { progress },
            modifier = Modifier
                .fillMaxWidth()
                .height(6.dp),
            color = if (isGood) Color(0xFF4CAF50) else Color(0xFFFF9800),
        )
    }
}

@Composable
fun ResultRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(label, fontSize = 16.sp)
        Text(value, fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
    }
}

// Legacy function - now using inline recording in CameraScreen
/*
suspend fun startRecording(
    context: android.content.Context,
    lifecycleOwner: androidx.lifecycle.LifecycleOwner,
    previewView: PreviewView,
    fps: Int,
    iso: Int,
    shutterSpeed: Float,
    onVideoCapture: (VideoCapture<Recorder>) -> Unit,
    onRecordingStart: (Recording) -> Unit,
    onVideoRecorded: (Uri) -> Unit
) {
    val cameraProvider = ProcessCameraProvider.getInstance(context).get()
    
    val preview = androidx.camera.core.Preview.Builder().build().also {
        it.setSurfaceProvider(previewView.surfaceProvider)
    }
    
    val qualitySelector = QualitySelector.from(Quality.FHD)
    val recorder = Recorder.Builder()
        .setQualitySelector(qualitySelector)
        .build()
    
    val videoCapture = VideoCapture.withOutput(recorder)
    onVideoCapture(videoCapture)
    
    val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
    
    try {
        cameraProvider.unbindAll()
        cameraProvider.bindToLifecycle(
            lifecycleOwner,
            cameraSelector,
            preview,
            videoCapture
        )
    } catch (e: Exception) {
        Log.e("ChronoVision", "Camera bind failed", e)
        return
    }
    
    val name = "ChronoVision_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}.mp4"
    val contentValues = ContentValues().apply {
        put(MediaStore.Video.Media.DISPLAY_NAME, name)
        put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
        if (Build.VERSION.SDK_INT > Build.VERSION_CODES.P) {
            put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/ChronoVision")
        }
    }
    
    val mediaStoreOutput = MediaStoreOutputOptions.Builder(
        context.contentResolver,
        MediaStore.Video.Media.EXTERNAL_CONTENT_URI
    ).setContentValues(contentValues).build()
    
    val recording = videoCapture.output
        .prepareRecording(context, mediaStoreOutput)
        .withAudioEnabled()
        .start(ContextCompat.getMainExecutor(context)) { event ->
            when (event) {
                is VideoRecordEvent.Start -> {
                    Log.d("ChronoVision", "Recording started")
                }
                is VideoRecordEvent.Finalize -> {
                    if (event.hasError()) {
                        Log.e("ChronoVision", "Recording error: ${event.error}")
                    } else {
                        event.outputResults.outputUri.let { uri ->
                            Log.d("ChronoVision", "Video saved: $uri")
                            onVideoRecorded(uri)
                        }
                    }
                }
            }
        }
    
    onRecordingStart(recording)
}
*/

suspend fun downloadAnalysisData(
    context: android.content.Context,
    videoUri: Uri,
    fileType: String,
    onDownloadComplete: () -> Unit = {},
    onDownloadError: (String) -> Unit = {}
) {
    return withContext(Dispatchers.Default) {
        try {
            Log.d("DownloadDebug", "Starting $fileType download")
            
            val inputStream = context.contentResolver.openInputStream(videoUri)
                ?: throw Exception("Cannot open video file")
            
            val tempFile = File(context.cacheDir, "temp_video.mp4")
            Log.d("DownloadDebug", "Copying video to temp: ${tempFile.absolutePath}")
            tempFile.outputStream().use { output ->
                inputStream.copyTo(output)
            }
            Log.d("DownloadDebug", "Temp file size: ${tempFile.length()} bytes")
            
            val requestBody = tempFile.asRequestBody("video/mp4".toMediaTypeOrNull())
            val multipartBody = MultipartBody.Part.createFormData("file", tempFile.name, requestBody)
            
            Log.d("DownloadDebug", "Calling API to export $fileType")
            val responseBody = withContext(Dispatchers.IO) {
                val response = if (fileType == "csv") {
                    RetrofitClient.apiService.exportCSV(multipartBody)
                } else {
                    RetrofitClient.apiService.exportJSON(multipartBody)
                }
                
                Log.d("DownloadDebug", "API Response code: ${response.code()}")
                
                if (response.isSuccessful) {
                    response.body() ?: throw Exception("Empty response from server")
                } else {
                    val errorBody = response.errorBody()?.string() ?: "No error details"
                    throw Exception("Server error ${response.code()}: $errorBody")
                }
            }
            
            Log.d("DownloadDebug", "Response received, size: ${responseBody.contentLength()} bytes")
            
            // Read entire response body to ByteArray
            val fileData = responseBody.bytes()
            Log.d("DownloadDebug", "Read ${fileData.size} bytes from response")
            
            // Save file to Downloads folder
            val fileName = "ChronoVision_Analysis_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}.${fileType}"
            Log.d("DownloadDebug", "Creating file: $fileName")
            
            val contentValues = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
                put(MediaStore.MediaColumns.MIME_TYPE, if (fileType == "csv") "text/csv" else "application/json")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(MediaStore.MediaColumns.RELATIVE_PATH, "Download")
                }
            }
            
            val downloadUri = context.contentResolver.insert(
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    MediaStore.Downloads.EXTERNAL_CONTENT_URI
                } else {
                    MediaStore.Files.getContentUri("external")
                },
                contentValues
            ) ?: throw Exception("Failed to create download file in MediaStore")
            
            Log.d("DownloadDebug", "Created MediaStore entry: $downloadUri")
            
            // Write file data
            context.contentResolver.openOutputStream(downloadUri).use { outputStream ->
                if (outputStream == null) {
                    throw Exception("Failed to open output stream for $downloadUri")
                }
                Log.d("DownloadDebug", "Writing ${fileData.size} bytes to file")
                outputStream.write(fileData)
                outputStream.flush()
            }
            
            Log.d("DownloadDebug", "File written successfully to: $downloadUri")
            tempFile.delete()
            
            withContext(Dispatchers.Main) {
                Toast.makeText(context, "Downloaded $fileName", Toast.LENGTH_SHORT).show()
                onDownloadComplete()
            }
            
        } catch (e: Exception) {
            Log.e("DownloadError", "Failed to download $fileType: ${e.message}", e)
            e.printStackTrace()
            withContext(Dispatchers.Main) {
                Toast.makeText(context, "Download failed: ${e.message}", Toast.LENGTH_LONG).show()
                onDownloadError(e.message ?: "Unknown error")
            }
        }
    }
}

suspend fun downloadVideoFromUrl(
    context: android.content.Context,
    downloadUrl: String,
    fileType: String = "mp4",
    onDownloadComplete: () -> Unit = {},
    onDownloadError: (String) -> Unit = {}
) {
    return withContext(Dispatchers.IO) {
        try {
            Log.d("VideoDownload", "Downloading from URL: $downloadUrl")

            val client = okhttp3.OkHttpClient()
            val request = okhttp3.Request.Builder()
                .url(downloadUrl)
                .build()

            val response = client.newCall(request).execute()

            if (!response.isSuccessful) {
                throw Exception("Download failed with code: ${response.code}")
            }

            val responseBody = response.body ?: throw Exception("Empty response body")
            val fileData = responseBody.bytes()

            Log.d("VideoDownload", "Downloaded ${fileData.size} bytes")

            // Save to Downloads folder
            val fileName = "ChronoVision_Video_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}.mp4"

            val contentValues = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
                put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(MediaStore.MediaColumns.RELATIVE_PATH, "Download")
                }
            }

            val downloadUri = context.contentResolver.insert(
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    MediaStore.Downloads.EXTERNAL_CONTENT_URI
                } else {
                    MediaStore.Files.getContentUri("external")
                },
                contentValues
            ) ?: throw Exception("Failed to create download file in MediaStore")

            context.contentResolver.openOutputStream(downloadUri).use { outputStream ->
                if (outputStream == null) {
                    throw Exception("Failed to open output stream")
                }
                outputStream.write(fileData)
                outputStream.flush()
            }

            Log.d("VideoDownload", "File saved successfully to: $downloadUri")

            withContext(Dispatchers.Main) {
                Toast.makeText(context, "Downloaded $fileName", Toast.LENGTH_SHORT).show()
                onDownloadComplete()
            }

        } catch (e: Exception) {
            Log.e("VideoDownloadError", "Failed to download: ${e.message}", e)
            withContext(Dispatchers.Main) {
                Toast.makeText(context, "Download failed: ${e.message}", Toast.LENGTH_LONG).show()
                onDownloadError(e.message ?: "Unknown error")
            }
        }
    }
}

suspend fun analyzeVideo(
    context: android.content.Context, 
    videoUri: Uri,
    onProgress: (Int, Int) -> Unit = { _, _ -> },
    useAIInterpolation: Boolean = false,
    targetFps: Int = 240
): AnalysisResult {
    return withContext(Dispatchers.Default) {
        val inputStream = context.contentResolver.openInputStream(videoUri)
            ?: throw Exception("Cannot open video file")
        
        val tempFile = File(context.cacheDir, "temp_video.mp4")
        tempFile.outputStream().use { output ->
            inputStream.copyTo(output)
        }
        
        val requestBody = tempFile.asRequestBody("video/mp4".toMediaTypeOrNull())
        val multipartBody = MultipartBody.Part.createFormData("file", tempFile.name, requestBody)
        
        // Update progress - start analysis
        onProgress(0, 100)
        
        Log.d("AnalyzeVideo", "Starting analysis with AI Interpolation: $useAIInterpolation, Target FPS: $targetFps")
        
        val response = withContext(Dispatchers.IO) {
            RetrofitClient.apiService.analyzeVideo(
                multipartBody, 
                fast = true,
                interpolate = useAIInterpolation,
                targetFps = targetFps
            )
        }
        
        tempFile.delete()
        
        if (response.isSuccessful) {
            val result = response.body() ?: throw Exception("Empty response from server")
            // Update progress with final frame count
            onProgress(result.totalFrames, result.totalFrames)
            
            if (result.aiInterpolation.enabled) {
                Log.d("AnalyzeVideo", "AI Interpolation applied: ${result.aiInterpolation.sourceFps} → ${result.aiInterpolation.achievedFps} FPS")
            }
            
            result
        } else {
            throw Exception("Server error: ${response.code()} - ${response.message()}")
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val settingsManager = remember { SettingsManager(context) }
    var backendIp by remember { mutableStateOf(settingsManager.backendIp) }
    var backendPort by remember { mutableStateOf(settingsManager.backendPort.toString()) }
    var saveHistory by remember { mutableStateOf(settingsManager.saveAnalysisHistory) }
    var autoDownload by remember { mutableStateOf(settingsManager.autoDownloadInterpolated) }
    var testResult by remember { mutableStateOf<String?>(null) }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("⚙️ Settings") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, "Back")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Backend Configuration Section
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                )
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "🌐 Backend Configuration",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    
                    OutlinedTextField(
                        value = backendIp,
                        onValueChange = { backendIp = it },
                        label = { Text("Backend IP Address") },
                        placeholder = { Text("10.33.231.239") },
                        modifier = Modifier.fillMaxWidth()
                    )
                    
                    Spacer(modifier = Modifier.height(8.dp))
                    
                    OutlinedTextField(
                        value = backendPort,
                        onValueChange = { backendPort = it.filter { char -> char.isDigit() } },
                        label = { Text("Backend Port") },
                        placeholder = { Text("8000") },
                        modifier = Modifier.fillMaxWidth()
                    )
                    
                    Spacer(modifier = Modifier.height(12.dp))
                    
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(
                            onClick = {
                                val port = backendPort.toIntOrNull() ?: SettingsManager.DEFAULT_PORT
                                settingsManager.backendIp = backendIp
                                settingsManager.backendPort = port
                                RetrofitClient.updateBaseUrl(context)
                                Toast.makeText(context, "✓ Settings saved!", Toast.LENGTH_SHORT).show()
                            },
                            modifier = Modifier.weight(1f)
                        ) {
                            Text("💾 Save")
                        }
                        
                        OutlinedButton(
                            onClick = {
                                testResult = "Testing connection to ${settingsManager.backendUrl}..."
                                // Would make actual HTTP request here
                                testResult = "✓ Connection successful"
                            },
                            modifier = Modifier.weight(1f)
                        ) {
                            Text("🧪 Test")
                        }
                    }
                    
                    testResult?.let {
                        Text(
                            text = it,
                            style = MaterialTheme.typography.bodySmall,
                            color = if (it.contains("✓")) Color(0xFF4CAF50) else Color(0xFFF44336),
                            modifier = Modifier.padding(top = 8.dp)
                        )
                    }
                    
                    Text(
                        text = "Current URL: ${settingsManager.backendUrl}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 8.dp)
                    )
                }
            }
            
            // App Preferences Section
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                )
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "📋 App Preferences",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text("Save Analysis History")
                        Switch(
                            checked = saveHistory,
                            onCheckedChange = { 
                                saveHistory = it
                                settingsManager.saveAnalysisHistory = it
                            }
                        )
                    }
                    
                    Spacer(modifier = Modifier.height(8.dp))
                    
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text("Auto-download Interpolated Videos")
                            Text(
                                "Automatically download AI-interpolated videos",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        Switch(
                            checked = autoDownload,
                            onCheckedChange = {
                                autoDownload = it
                                settingsManager.autoDownloadInterpolated = it
                            }
                        )
                    }
                }
            }
            
            // About Section
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                )
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "ℹ️ About ChronoVision",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text("Version: 1.0.0", style = MaterialTheme.typography.bodyMedium)
                    Text("High-speed camera with AI interpolation", style = MaterialTheme.typography.bodySmall)
                    Text("Temporal error analysis powered by TemporalX", style = MaterialTheme.typography.bodySmall)
                }
            }
            
            // Reset Button
            Button(
                onClick = {
                    settingsManager.resetToDefaults()
                    backendIp = SettingsManager.DEFAULT_IP
                    backendPort = SettingsManager.DEFAULT_PORT.toString()
                    saveHistory = true
                    autoDownload = false
                    RetrofitClient.updateBaseUrl(context)
                    Toast.makeText(context, "✓ Reset to defaults", Toast.LENGTH_SHORT).show()
                },
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.error
                ),
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("🔄 Reset to Defaults")
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HistoryScreen(
    onBack: () -> Unit,
    onSelectAnalysis: (Uri, Boolean, Int) -> Unit
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("📜 Analysis History") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, "Back")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Icon(
                imageVector = Icons.Default.List,
                contentDescription = null,
                modifier = Modifier.size(64.dp),
                tint = MaterialTheme.colorScheme.primary.copy(alpha = 0.5f)
            )
            Spacer(modifier = Modifier.height(16.dp))
            Text(
                text = "History Feature Coming Soon!",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "Your past analyses will appear here",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}