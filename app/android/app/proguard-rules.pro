# R8 rules for the offline recognition path.
#
# Both TF Lite and MediaPipe reach their Java classes from native code, so R8
# cannot see those references and will happily strip or rename them. The failures
# that causes are runtime UnsatisfiedLinkError / ClassNotFoundException in a
# release build only — never in debug — so they are exactly the kind of thing that
# shows up for the first time on the day of a demo.

# --- TF Lite (runs the BiLSTM: assets/models/model.tflite) ---
-keep class org.tensorflow.lite.** { *; }
-keep interface org.tensorflow.lite.** { *; }

# The GPU delegate is referenced by tflite_flutter but not shipped in full, and
# this model cannot use it regardless: the exported BiLSTM contains WHILE control
# flow, which the GPU delegate does not support, so it runs on CPU/XNNPACK. These
# references are unreachable — silence them rather than adding the dependency.
-dontwarn org.tensorflow.lite.gpu.**

# --- MediaPipe Tasks (pose + hand landmarkers) ---
# The task graphs instantiate these over JNI by name.
-keep class com.google.mediapipe.** { *; }
-keep interface com.google.mediapipe.** { *; }
-dontwarn com.google.mediapipe.**

# Protobuf-generated classes underneath MediaPipe, likewise reflective.
-keep class com.google.protobuf.** { *; }
-dontwarn com.google.protobuf.**

# AutoValue's annotation processor is excluded from the runtime classpath in
# build.gradle.kts (it is compile-time only and references javax.lang.model.*,
# which Android does not have). These stay as a backstop in case a transitive
# dependency drags a shaded copy back in — the generated AutoValue_* classes
# themselves live under com.google.mediapipe.* and are already kept above.
-dontwarn com.google.auto.**
-dontwarn autovalue.shaded.**
-dontwarn javax.lang.model.**
-dontwarn javax.annotation.processing.**
