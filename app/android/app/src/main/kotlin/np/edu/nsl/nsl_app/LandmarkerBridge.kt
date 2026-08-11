package np.edu.nsl.nsl_app

import android.content.Context
import android.graphics.Bitmap
import android.os.Handler
import android.os.Looper
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import kotlin.math.hypot

/**
 * On-device replacement for MediaPipe Holistic.
 *
 * The model was trained on Holistic's 225-vector, and Holistic has no mobile
 * build — so the phone has to rebuild the identical vector from two independent
 * Tasks detectors. This file is the Kotlin port of `nslr/tasks_landmarks.py`,
 * and the parity between the two was measured before it was written: 36 paired
 * clips, 100% -> 100% top-1, mean |dprob| L1 = 0.006.
 *
 * Three things here are correctness-critical, not stylistic:
 *
 *  1. **Normalized landmarks, never world landmarks.** `landmarks()` returns
 *     x,y in [0,1] and a z on roughly the x scale — exactly what Holistic gave
 *     `record.py`. `worldLandmarks()` is metric and would be a different
 *     feature space entirely.
 *
 *  2. **Hands are assigned by nearest pose wrist, not by the handedness
 *     classifier.** Holistic never had to decide: it crops each hand from the
 *     pose's wrist, so "left hand" means "the hand on the pose's left wrist" by
 *     construction. The classifier disagreed with that rule on ~7% of the parity
 *     frames, and following it would swap the two 63-float hand blocks on those
 *     frames. `handednessAgreement` is reported so that stays measurable.
 *
 *  3. **VIDEO running mode with monotonic timestamps**, so both detectors track
 *     across frames the way Holistic did inside `record.py`'s capture loop. A
 *     non-increasing timestamp makes MediaPipe throw.
 */
class LandmarkerBridge(private val context: Context) : FlutterPlugin, MethodChannel.MethodCallHandler {

    companion object {
        const val CHANNEL = "np.edu.nsl.nsl_app/landmarker"

        // Slice layout of the 225-vector — must match nslr/config.py exactly.
        private const val POSE_LANDMARKS = 33
        private const val HAND_LANDMARKS = 21
        private const val POSE_DIM = POSE_LANDMARKS * 3      // 99
        private const val HAND_DIM = HAND_LANDMARKS * 3      // 63
        private const val FEATURE_DIM = POSE_DIM + HAND_DIM * 2  // 225

        // Pose wrist indices used for hand assignment (nslr/tasks_landmarks.py).
        private const val POSE_LEFT_WRIST = 15
        private const val POSE_RIGHT_WRIST = 16
    }

    private var channel: MethodChannel? = null
    private var pose: PoseLandmarker? = null
    private var hands: HandLandmarker? = null

    /** Single thread on purpose: the detectors hold tracking state in VIDEO mode
     *  and are not safe to drive concurrently. It also serializes frames, which
     *  is what keeps timestamps monotonic. */
    private val worker = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())

    /** Guards against a frame arriving while the previous one is still in the
     *  detectors. Dropping is correct — the clip is time-decimated anyway, and a
     *  queue would only add latency and break the fps target. */
    @Volatile
    private var busy = false

    @Volatile
    private var lastTimestamp = -1L

    /** Added to every incoming timestamp so the stream stays monotonic ACROSS clips.
     *
     * MediaPipe's VIDEO mode requires monotonically increasing timestamps for the
     * lifetime of the landmarker, not per clip. The Dart side measures time from
     * the start of each recording, so every new clip restarts near zero — which
     * MediaPipe rejects outright, and the detectors are then unusable. Carrying an
     * offset forward keeps within-clip spacing intact (the trackers use it to
     * reason about motion) while guaranteeing the global sequence never goes
     * backwards. */
    @Volatile
    private var timestampOffset = 0L

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel = MethodChannel(binding.binaryMessenger, CHANNEL).also {
            it.setMethodCallHandler(this)
        }
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel?.setMethodCallHandler(null)
        channel = null
        closeDetectors()
        worker.shutdown()
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "init" -> handleInit(call, result)
            "detect" -> handleDetect(call, result)
            "reset" -> {
                // New clip. The detectors keep their tracking state, which matches
                // record.py holding one Holistic graph open across clips.
                //
                // lastTimestamp is deliberately NOT cleared: the Dart clock
                // restarts at zero for each clip, so we rebase onto the end of the
                // previous one. Clearing it here is what made the second recording
                // fail with a monotonicity error while the first worked.
                timestampOffset = lastTimestamp + 1
                result.success(null)
            }
            "close" -> {
                closeDetectors()
                result.success(null)
            }
            else -> result.notImplemented()
        }
    }

    private fun handleInit(call: MethodCall, result: MethodChannel.Result) {
        val poseKey = call.argument<String>("poseAsset")
        val handKey = call.argument<String>("handAsset")
        val useGpu = call.argument<Boolean>("useGpu") ?: true
        if (poseKey == null || handKey == null) {
            result.error("bad_args", "poseAsset and handAsset are required", null)
            return
        }
        worker.execute {
            try {
                closeDetectors()

                // setModelAssetBuffer rather than setModelAssetPath: the .task
                // files ship as Flutter assets, whose on-disk layout inside the
                // APK is Flutter's business and has changed between versions.
                // Reading the bytes ourselves is version-proof.
                val poseBuf = readAssetDirect(poseKey)
                val handBuf = readAssetDirect(handKey)

                // GPU is worth trying for the landmarkers (they are per-frame and
                // convolutional), but falls back rather than failing: emulators and
                // some low-end GPUs have no working delegate.
                val delegate = if (useGpu) {
                    com.google.mediapipe.tasks.core.Delegate.GPU
                } else {
                    com.google.mediapipe.tasks.core.Delegate.CPU
                }

                try {
                    buildDetectors(poseBuf, handBuf, delegate)
                } catch (e: Throwable) {
                    if (delegate == com.google.mediapipe.tasks.core.Delegate.CPU) throw e
                    // Rewind: a failed build may have consumed the buffers.
                    poseBuf.rewind()
                    handBuf.rewind()
                    buildDetectors(poseBuf, handBuf, com.google.mediapipe.tasks.core.Delegate.CPU)
                }

                // Fresh detectors have no timestamp history, so the global sequence
                // genuinely starts over here (unlike "reset", which must not).
                lastTimestamp = -1L
                timestampOffset = 0L
                main.post { result.success(null) }
            } catch (e: Throwable) {
                main.post { result.error("init_failed", e.message, null) }
            }
        }
    }

    private fun buildDetectors(
        poseBuf: ByteBuffer,
        handBuf: ByteBuffer,
        delegate: com.google.mediapipe.tasks.core.Delegate,
    ) {
        // Confidences match record.py's Holistic (0.5 / 0.5) and the parity
        // spike. They are part of the measured contract, not tuning knobs.
        val poseOptions = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetBuffer(poseBuf)
                    .setDelegate(delegate)
                    .build()
            )
            .setRunningMode(RunningMode.VIDEO)
            .setNumPoses(1)
            .setMinPoseDetectionConfidence(0.5f)
            .setMinPosePresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setOutputSegmentationMasks(false)
            .build()

        val handOptions = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetBuffer(handBuf)
                    .setDelegate(delegate)
                    .build()
            )
            .setRunningMode(RunningMode.VIDEO)
            .setNumHands(2)
            .setMinHandDetectionConfidence(0.5f)
            .setMinHandPresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .build()

        pose = PoseLandmarker.createFromOptions(context, poseOptions)
        hands = HandLandmarker.createFromOptions(context, handOptions)
    }

    /** MediaPipe requires a *direct* ByteBuffer for setModelAssetBuffer. */
    private fun readAssetDirect(lookupKey: String): ByteBuffer {
        context.assets.open(lookupKey).use { stream ->
            val bytes = stream.readBytes()
            return ByteBuffer.allocateDirect(bytes.size).apply {
                put(bytes)
                rewind()
            }
        }
    }

    private fun handleDetect(call: MethodCall, result: MethodChannel.Result) {
        val p = pose
        val h = hands
        if (p == null || h == null) {
            result.error("not_initialised", "call init first", null)
            return
        }
        val rgba = call.argument<ByteArray>("rgba")
        val width = call.argument<Int>("width") ?: 0
        val height = call.argument<Int>("height") ?: 0
        var timestampMs = (call.argument<Int>("timestampMs") ?: 0).toLong()
        if (rgba == null || width <= 0 || height <= 0) {
            result.error("bad_args", "rgba/width/height required", null)
            return
        }
        if (rgba.size < width * height * 4) {
            result.error("bad_args", "rgba is ${rgba.size}, expected ${width * height * 4}", null)
            return
        }
        if (busy) {
            result.success(mapOf("dropped" to true))
            return
        }
        busy = true

        worker.execute {
            try {
                // Rebase onto the global sequence, then clamp. VIDEO mode rejects a
                // non-increasing timestamp with an exception, and clamping is better
                // than dropping the frame: the value only has to be monotonic and
                // roughly correctly spaced for the trackers. The model itself reads
                // frame COUNT as duration and never sees these stamps.
                timestampMs += timestampOffset
                if (timestampMs <= lastTimestamp) timestampMs = lastTimestamp + 1
                lastTimestamp = timestampMs

                val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(rgba))
                val image = BitmapImageBuilder(bitmap).build()

                val poseResult = p.detectForVideo(image, timestampMs)
                val handResult = h.detectForVideo(image, timestampMs)
                val payload = assemble(poseResult, handResult)
                bitmap.recycle()

                main.post {
                    busy = false
                    result.success(payload)
                }
            } catch (e: Throwable) {
                main.post {
                    busy = false
                    result.error("detect_failed", e.message, null)
                }
            }
        }
    }

    /**
     * Build the 225-vector: [pose 99 | left hand 63 | right hand 63], with
     * undetected blocks left as zeros. Zero-filling is not a fallback — it is
     * what `record.py` stored for a missing block, and the model's Masking layer
     * plus the normalization formulas both depend on it (zero maps to zero).
     */
    private fun assemble(
        poseResult: PoseLandmarkerResult,
        handResult: HandLandmarkerResult,
    ): Map<String, Any?> {
        // DoubleArray, not FloatArray. Flutter's StandardMessageCodec has encoded
        // double[] since forever; float[] support is newer and version-dependent,
        // and when the codec cannot encode a value it throws while writing the
        // REPLY -- which surfaces on the Dart side as a failed call rather than as
        // anything identifiable here. 225 doubles per frame at 16 fps is ~28 KB/s,
        // which is not worth a compatibility risk.
        val vector = DoubleArray(FEATURE_DIM)
        var posePresent = false
        var leftPresent = false
        var rightPresent = false
        var handednessAgrees: Boolean? = null

        // --- pose block ---
        var poseXy: Array<FloatArray>? = null
        if (poseResult.landmarks().isNotEmpty()) {
            val lms = poseResult.landmarks()[0]
            if (lms.size >= POSE_LANDMARKS) {
                poseXy = Array(POSE_LANDMARKS) { i -> floatArrayOf(lms[i].x(), lms[i].y()) }
                for (i in 0 until POSE_LANDMARKS) {
                    vector[i * 3] = lms[i].x().toDouble()
                    vector[i * 3 + 1] = lms[i].y().toDouble()
                    vector[i * 3 + 2] = lms[i].z().toDouble()
                }
                posePresent = true
            }
        }

        // --- hand blocks ---
        val handLists = handResult.landmarks()
        if (handLists.isNotEmpty()) {
            // Flatten each detected hand once, alongside the classifier's label.
            val candidates = ArrayList<Triple<DoubleArray, FloatArray, String>>() // flat, wristXy, label
            for (i in handLists.indices) {
                val lms = handLists[i]
                if (lms.size < HAND_LANDMARKS) continue
                val flat = DoubleArray(HAND_DIM)
                for (j in 0 until HAND_LANDMARKS) {
                    flat[j * 3] = lms[j].x().toDouble()
                    flat[j * 3 + 1] = lms[j].y().toDouble()
                    flat[j * 3 + 2] = lms[j].z().toDouble()
                }
                val label = handResult.handedness().getOrNull(i)?.firstOrNull()?.categoryName() ?: ""
                candidates.add(Triple(flat, floatArrayOf(lms[0].x(), lms[0].y()), label))
            }

            val blocks = HashMap<String, DoubleArray>()
            if (poseXy != null) {
                // Nearest-wrist assignment in image xy only. The pose and hand
                // models put z on different scales, so including it would make
                // the distance meaningless.
                val leftAnchor = poseXy[POSE_LEFT_WRIST]
                val rightAnchor = poseXy[POSE_RIGHT_WRIST]

                data class Claim(
                    val side: String,
                    val distance: Float,
                    val flat: DoubleArray,
                    val label: String,
                )

                val claims = candidates.map { (flat, wrist, label) ->
                    val dLeft = hypot((wrist[0] - leftAnchor[0]).toDouble(), (wrist[1] - leftAnchor[1]).toDouble()).toFloat()
                    val dRight = hypot((wrist[0] - rightAnchor[0]).toDouble(), (wrist[1] - rightAnchor[1]).toDouble()).toFloat()
                    Claim(
                        if (dLeft <= dRight) "left" else "right",
                        minOf(dLeft, dRight),
                        flat,
                        label,
                    )
                }.sortedBy { it.distance }  // both hands claiming one wrist: closer wins

                val taken = HashSet<String>()
                for (claim in claims) {
                    val slot = if (claim.side !in taken) {
                        claim.side
                    } else {
                        if (claim.side == "left") "right" else "left"
                    }
                    if (slot in taken) continue
                    taken.add(slot)
                    blocks[slot] = claim.flat
                    if (claim.label.isNotEmpty()) {
                        val agrees = claim.label.lowercase() == slot
                        handednessAgrees = if (handednessAgrees == null) agrees else (handednessAgrees!! && agrees)
                    }
                }
            } else {
                // No pose this frame -> the classifier is all we have.
                for ((flat, _, label) in candidates) {
                    val slot = label.lowercase()
                    if ((slot == "left" || slot == "right") && !blocks.containsKey(slot)) {
                        blocks[slot] = flat
                    }
                }
            }

            blocks["left"]?.let {
                it.copyInto(vector, POSE_DIM)
                leftPresent = true
            }
            blocks["right"]?.let {
                it.copyInto(vector, POSE_DIM + HAND_DIM)
                rightPresent = true
            }
        }

        return mapOf(
            "dropped" to false,
            "vector" to vector,
            "pose" to posePresent,
            "leftHand" to leftPresent,
            "rightHand" to rightPresent,
            "handednessAgrees" to handednessAgrees,
        )
    }

    private fun closeDetectors() {
        try {
            pose?.close()
        } catch (_: Throwable) {
        }
        try {
            hands?.close()
        } catch (_: Throwable) {
        }
        pose = null
        hands = null
    }
}
