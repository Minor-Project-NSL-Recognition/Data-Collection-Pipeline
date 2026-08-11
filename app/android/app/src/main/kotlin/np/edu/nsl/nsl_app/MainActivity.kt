package np.edu.nsl.nsl_app

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine

class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        // Registered manually rather than as a pubspec plugin: it exists only for
        // this app and has no standalone package.
        flutterEngine.plugins.add(LandmarkerBridge(applicationContext))
    }
}
