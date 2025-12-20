import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_v2/tflite_v2.dart';
import 'package:permission_handler/permission_handler.dart';

List<CameraDescription>? cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Request camera permission before starting
  await Permission.camera.request();

  // Get list of available cameras
  cameras = await availableCameras();

  runApp(const SignLanguageApp());
}

class SignLanguageApp extends StatelessWidget {
  const SignLanguageApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      home: const DetectionScreen(),
    );
  }
}

class DetectionScreen extends StatefulWidget {
  const DetectionScreen({super.key});

  @override
  State<DetectionScreen> createState() => _DetectionScreenState();
}

class _DetectionScreenState extends State<DetectionScreen> {
  CameraController? cameraController;
  String output = "Initializing...";
  double confidence = 0.0;
  bool isBusy = false;

  @override
  void initState() {
    super.initState();
    loadModel().then((_) {
      initCamera();
    });
  }

  // 1. Load the Model and Labels
  Future<void> loadModel() async {
    try {
      String? res = await Tflite.loadModel(
        model: "assets/model.tflite",
        labels: "assets/labels.txt",
        numThreads: 1,
        isAsset: true,
        useGpuDelegate: false,
      );
      print("Model loaded: $res");
    } catch (e) {
      print("Failed to load model: $e");
    }
  }

  // 2. Setup Camera Stream
  void initCamera() {
    if (cameras == null || cameras!.isEmpty) return;

    cameraController = CameraController(
      cameras![0], // Use the back camera
      ResolutionPreset.medium,
      enableAudio: false,
    );

    cameraController!
        .initialize()
        .then((_) {
          if (!mounted) return;

          // Start the live frame stream
          cameraController!.startImageStream((CameraImage image) {
            if (!isBusy) {
              isBusy = true;
              runModelOnFrame(image);
            }
          });
          setState(() {});
        })
        .catchError((e) {
          print("Camera init error: $e");
        });
  }

  // 3. Run Inference on Frame
  Future<void> runModelOnFrame(CameraImage image) async {
    try {
      var recognitions = await Tflite.runModelOnFrame(
        bytesList: image.planes.map((plane) => plane.bytes).toList(),
        imageHeight: image.height,
        imageWidth: image.width,
        imageMean: 0.0,    // For 0-1 normalization (divide by 255)
        imageStd: 255.0,   // For 0-1 normalization (divide by 255)
        rotation: 90,      // Common for portrait Android devices
        numResults: 1,     // We only need the top result
        threshold: 0.1,    // Lower threshold for better detection
        asynch: true,
      );

      if (recognitions != null && recognitions.isNotEmpty) {
        setState(() {
          output = recognitions[0]['label'].toString();
          confidence = (recognitions[0]['confidence'] as double) * 100;
        });
      }
    } catch (e) {
      print("Inference error: $e");
    } finally {
      isBusy = false;
    }
  }

  @override
  void dispose() {
    cameraController?.dispose();
    Tflite.close(); // Important: Release memory
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (cameraController == null || !cameraController!.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(title: const Text("ASL Detection")),
      body: Stack(
        children: [
          // Camera Preview (Full Screen)
          SizedBox(
            width: double.infinity,
            height: double.infinity,
            child: CameraPreview(cameraController!),
          ),

          // Result Box
          Align(
            alignment: Alignment.bottomCenter,
            child: Container(
              width: double.infinity,
              margin: const EdgeInsets.all(20),
              padding: const EdgeInsets.symmetric(horizontal: 15, vertical: 20),
              decoration: BoxDecoration(
                color: Colors.black.withValues(alpha: 0.7),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: Colors.blueAccent, width: 2),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min, // Shrink-wrap the content
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    "Detected Sign: $output",
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      fontSize: 26,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    "Confidence: ${confidence.toStringAsFixed(1)}%",
                    style: const TextStyle(
                      fontSize: 18,
                      color: Colors.greenAccent,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
