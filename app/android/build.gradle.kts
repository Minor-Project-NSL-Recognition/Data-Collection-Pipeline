allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
// Pin every plugin subproject to JVM 17, on both the Java and Kotlin side.
//
// The plugins in this app disagree about JVM target: tflite_flutter sets no
// jvmTarget at all (so Kotlin lands on the toolchain default) while flutter_tts
// pins Java to 11. Gradle then fails the build with "Inconsistent JVM-target
// compatibility detected". Fixing it centrally beats forking three plugins.
//
// The timing is the fiddly part. This callback is registered from the root
// script, so for each subproject it runs *after* that plugin's own build script
// (which is where flutter_tts pins Java 11) but *before* AGP's own afterEvaluate,
// which is when the JavaCompile tasks are actually created. So the android
// extension is the thing to set: AGP reads compileOptions at task-creation time
// and our value is what it sees. Setting the tasks directly does not work — AGP
// creates them later and overwrites.
//
// 17 matches what :app already uses.
subprojects {
    afterEvaluate {
        (extensions.findByName("android") as? com.android.build.gradle.BaseExtension)
            ?.let { android ->
                android.compileOptions {
                    sourceCompatibility = JavaVersion.VERSION_17
                    targetCompatibility = JavaVersion.VERSION_17
                }
                // tflite_flutter still declares compileSdk 31, which is below what
                // the forced TF Lite 2.16 AARs require (>= 33). Raising it here is
                // safe: compileSdk only controls which APIs are visible at compile
                // time, not runtime behaviour (targetSdk) or device support
                // (minSdk), both of which stay as the plugin declared them.
                if (android.compileSdkVersion == null ||
                    (android.compileSdkVersion?.removePrefix("android-")?.toIntOrNull() ?: 0) < 34
                ) {
                    android.compileSdkVersion(36)
                }
            }
        tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinJvmCompile>().configureEach {
            compilerOptions {
                jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
            }
        }
    }
}

// Must stay AFTER the afterEvaluate hook above: this line evaluates every
// subproject immediately, and registering an afterEvaluate callback on an
// already-evaluated project is an error.
subprojects {
    project.evaluationDependsOn(":app")
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
