import 'package:flutter/material.dart';
import 'screens/home_screen.dart';
import 'screens/prediction_screen.dart';
import 'screens/batch_prediction_screen.dart';
import 'screens/results_screen.dart';

void main() {
  runApp(const YouthDigitalReadinessApp());
}

class YouthDigitalReadinessApp extends StatelessWidget {
  const YouthDigitalReadinessApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Youth Digital Readiness Predictor',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.blue,
          foregroundColor: Colors.white,
          elevation: 4,
        ),
      ),
      home: const HomeScreen(),
      routes: {
        '/home': (context) => const HomeScreen(),
        '/prediction': (context) => const PredictionScreen(),
        '/batch': (context) => const BatchPredictionScreen(),
        '/results': (context) => const ResultsScreen(),
      },
      debugShowCheckedModeBanner: false,
    );
  }
}
