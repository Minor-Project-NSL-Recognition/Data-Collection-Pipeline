plugins {
    id("com.android.application")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "np.edu.nsl.nsl_app"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "np.edu.nsl.nsl_app"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        // 24, not flutter.minSdkVersion: MediaPipe tasks-vision requires API 24+
        // for the on-device landmarkers. The offline path is the whole point of
        // this app, so this is a floor rather than a preference.
        minSdk = 24
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    // .task bundles and .tflite are read by native code via the AssetManager and
    // must stay uncompressed, or MediaPipe/TFLite cannot mmap them and loading
    // fails at runtime with an opaque error.
    androidResources {
        noCompress.add("task")
        noCompress.add("tflite")
    }

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")

            // Minification is OFF, deliberately.
            //
            // With R8 enabled, MediaPipe died at startup with
            // ExceptionInInitializerError inside com.google.mediapipe.framework
            // .Graph.<clinit> -- the native .so loaded fine 7 ms earlier, so
            // something R8 removed or renamed broke the static initializer. Broad
            // `-keep class com.google.mediapipe.**` / `com.google.protobuf.**`
            // rules did NOT fix it, and R8 does not report what it stripped, so
            // narrowing it down is guesswork against 90-second build cycles.
            //
            // Verified by bisection: the identical code in an unminified debug
            // build initialises all four landmarker sub-graphs cleanly.
            //
            // The trade is a few MB in an APK already carrying 18 MB of models,
            // against an entire class of release-only crash. Worth revisiting if
            // size ever matters; proguard-rules.pro is kept for that day.
            isMinifyEnabled = false
            isShrinkResources = false
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

dependencies {
    // On-device landmark extraction (LandmarkerBridge.kt). MediaPipe Holistic --
    // what the model was trained on -- has no mobile build, so the 225-vector is
    // rebuilt from PoseLandmarker + HandLandmarker. Parity between the two was
    // measured before this was written: 36 paired clips, 100% -> 100% top-1.
    implementation("com.google.mediapipe:tasks-vision:0.10.14") {
        // tasks-vision leaks AutoValue onto the RUNTIME classpath. AutoValue is a
        // compile-time annotation processor: its shaded copy of auto-common
        // references javax.lang.model.*, which does not exist on Android, and R8
        // fails the release build over the dangling references. The generated
        // AutoValue_* classes it produced are ordinary classes inside
        // com.google.mediapipe.* and do not need the processor at runtime.
        exclude(group = "com.google.auto.value", module = "auto-value")
    }
}

// tflite_flutter 0.11.0 pins TF Lite 2.11.0, whose tensorflow-lite,
// tensorflow-lite-api and tensorflow-lite-gpu AARs all declare the SAME
// namespace (org.tensorflow.lite). Current AGP rejects that outright:
//
//   Namespace 'org.tensorflow.lite' is used in multiple modules and/or libraries
//
// Upstream gave the artifacts distinct namespaces in later releases, so pinning
// them forward is the fix. Keep the three versions identical — mixing TF Lite
// artifact versions produces link errors that surface only at runtime.
configurations.all {
    resolutionStrategy {
        force("org.tensorflow:tensorflow-lite:2.16.1")
        force("org.tensorflow:tensorflow-lite-api:2.16.1")
        force("org.tensorflow:tensorflow-lite-gpu:2.16.1")
        force("org.tensorflow:tensorflow-lite-gpu-api:2.16.1")
    }
}

flutter {
    source = "../.."
}
