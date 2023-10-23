package com.shym.camerawithtfsample

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Size
import android.view.WindowManager
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.common.util.concurrent.ListenableFuture
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import java.util.concurrent.Executors
import kotlin.random.Random


class MainActivity : AppCompatActivity() {
    private val permissionsRequestCode = Random.nextInt(0, 10000)
    private val permissions = listOf(Manifest.permission.CAMERA)

    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>
    private val executor = Executors.newSingleThreadExecutor()

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == permissionsRequestCode && hasPermissions(this)) {
            bindCameraUseCases()
        } else {
            finish() // If we don't have the required permissions, we can't run
        }
    }

    /** Convenience method used to check if all permissions required by this app are granted */
    private fun hasPermissions(context: Context) = permissions.all {
        ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onResume() {
        super.onResume()

        // Request permissions each time the app resumes, since they can be revoked at any time
        if (!hasPermissions(this)) {
            ActivityCompat.requestPermissions(
                this, permissions.toTypedArray(), permissionsRequestCode
            )
        } else {
            bindCameraUseCases()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    private lateinit var bitmapBuffer: Bitmap
    private var imageRotationDegrees: Int = 0
    private var pauseAnalysis = false
    private val tfImageBuffer = TensorImage(DataType.UINT8)

    companion object {
        private const val MODEL_PATH = "coco_ssd_mobilenet_v1_1.0_quant.tflite"
        private const val LABELS_PATH = "coco_ssd_mobilenet_v1_1.0_labels.txt"
    }

    private val tfImageProcessor by lazy {
        val cropSize = minOf(bitmapBuffer.width, bitmapBuffer.height)
        ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(
                ResizeOp(
                    tfInputSize.height, tfInputSize.width, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR
                )
            )
            .add(Rot90Op(-imageRotationDegrees / 90))
            .add(NormalizeOp(0f, 1f))
            .build()
    }

    private val nnApiDelegate by lazy {
        NnApiDelegate()
    }

    private val tflite by lazy {
        Interpreter(
            FileUtil.loadMappedFile(this, MODEL_PATH),
            Interpreter.Options().addDelegate(nnApiDelegate)
        )
    }

    private val detector by lazy {
        ObjectDetectionHelper(
            tflite,
            FileUtil.loadLabels(this, LABELS_PATH)
        )
    }

    private val tfInputSize by lazy {
        val inputIndex = 0
        val inputShape = tflite.getInputTensor(inputIndex).shape()
        Size(inputShape[2], inputShape[1]) // Order of axis is: {1, height, width, 3}
    }

    private fun bindCameraUseCases() {
        cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener(Runnable {
            val cameraProvider = cameraProviderFuture.get()
            bindPreview(cameraProvider)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindPreview(cameraProvider: ProcessCameraProvider) {
        val previewView = this.findViewById<PreviewView>(R.id.previewView)
        val textView1 = this.findViewById<TextView>(R.id.textView1)
        val textView2 = this.findViewById<TextView>(R.id.textView2)

        val preview: Preview = Preview.Builder()
            .build()

        val cameraSelector: CameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview.setSurfaceProvider(previewView.surfaceProvider)

        val imageAnalysis = ImageAnalysis.Builder()
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
        imageAnalysis.setAnalyzer(executor, ImageAnalysis.Analyzer { image ->
            if (!::bitmapBuffer.isInitialized) {
                // The image rotation and RGB image buffer are initialized only once
                // the analyzer has started running
                imageRotationDegrees = image.imageInfo.rotationDegrees
                bitmapBuffer = Bitmap.createBitmap(
                    image.width, image.height, Bitmap.Config.ARGB_8888
                )
            }

            // Early exit: image analysis is in paused state
            if (pauseAnalysis) {
                image.close()
                return@Analyzer
            }

            // Copy out RGB bits to our shared buffer
            image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }

            // Process the image in Tensorflow
            val tfImage = tfImageProcessor.process(tfImageBuffer.apply { load(bitmapBuffer) })

            // Perform the object detection for the current frame
            val predictions = detector.predict(tfImage)

            runOnUiThread {
                textView1.text = predictions.maxByOrNull { it.score }?.label.toString()
                textView2.text = predictions.maxByOrNull { it.score }?.score.toString()
            }
        })

        var camera = cameraProvider.bindToLifecycle(
            this as LifecycleOwner,
            cameraSelector,
            preview,
            imageAnalysis
        )
    }
}