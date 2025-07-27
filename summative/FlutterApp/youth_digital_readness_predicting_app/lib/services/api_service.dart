import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = 'https://alu-ml-summatives-latest.onrender.com';

  static const Map<String, String> headers = {
    'Content-Type': 'application/json',
  };

  // Health check
  static Future<Map<String, dynamic>> healthCheck() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/health'),
        headers: headers,
      );

      if (response.statusCode == 200) {
        return json.decode(response.body);
      } else {
        throw Exception('Health check failed: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }

  // Get model info
  static Future<Map<String, dynamic>> getModelInfo() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/model/info'),
        headers: headers,
      );

      if (response.statusCode == 200) {
        return json.decode(response.body);
      } else {
        throw Exception('Failed to get model info: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }

  // Single user prediction
  static Future<Map<String, dynamic>> predictSingleUser(
      UserProfile user) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/predict'),
        headers: headers,
        body: json.encode(user.toJson()),
      );

      if (response.statusCode == 200) {
        return json.decode(response.body);
      } else {
        final error = json.decode(response.body);
        throw Exception('Prediction failed: ${error['detail']}');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }

  // Multiple users prediction
  static Future<Map<String, dynamic>> predictMultipleUsers(
      List<UserProfile> users) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/predict/users'),
        headers: headers,
        body: json.encode({
          'users': users.map((user) => user.toJson()).toList(),
        }),
      );

      if (response.statusCode == 200) {
        return json.decode(response.body);
      } else {
        final error = json.decode(response.body);
        throw Exception('Batch prediction failed: ${error['detail']}');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }
}

class UserProfile {
  final int locationType;
  final int householdSize;
  final int ageOfRespondent;
  final int genderOfRespondent;
  final int relationshipWithHead;
  final int maritalStatus;
  final int jobType;

  UserProfile({
    required this.locationType,
    required this.householdSize,
    required this.ageOfRespondent,
    required this.genderOfRespondent,
    required this.relationshipWithHead,
    required this.maritalStatus,
    required this.jobType,
  });

  Map<String, dynamic> toJson() {
    return {
      'location_type': locationType,
      'household_size': householdSize,
      'age_of_respondent': ageOfRespondent,
      'gender_of_respondent': genderOfRespondent,
      'relationship_with_head': relationshipWithHead,
      'marital_status': maritalStatus,
      'job_type': jobType,
    };
  }
}
