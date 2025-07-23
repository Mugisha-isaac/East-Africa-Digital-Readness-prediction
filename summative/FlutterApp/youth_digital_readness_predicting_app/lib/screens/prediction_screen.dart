import 'package:flutter/material.dart';
import '../services/api_service.dart';

class PredictionScreen extends StatefulWidget {
  const PredictionScreen({super.key});

  @override
  State<PredictionScreen> createState() => _PredictionScreenState();
}

class _PredictionScreenState extends State<PredictionScreen> {
  final _formKey = GlobalKey<FormState>();
  bool _isLoading = false;
  Map<String, dynamic>? _result;

  // Form fields matching your API exactly
  int _locationType = 1; // 0=Rural, 1=Urban
  int _householdSize = 4;
  int _age = 22;
  int _gender = 1; // 0=Female, 1=Male
  int _relationshipWithHead = 2;
  int _maritalStatus = 3;
  int _jobType = 3;

  // Options for dropdowns - matching your API's expected values
  final Map<int, String> _locationTypes = {0: 'Rural', 1: 'Urban'};
  final Map<int, String> _genders = {0: 'Female', 1: 'Male'};
  final Map<int, String> _relationships = {
    0: 'Head of household',
    1: 'Spouse',
    2: 'Child',
    3: 'Parent',
    4: 'Other relative',
    5: 'Non-relative'
  };
  final Map<int, String> _maritalStatuses = {
    0: 'Single',
    1: 'Married',
    2: 'Divorced',
    3: 'Widowed',
    4: 'Separated'
  };
  final Map<int, String> _jobTypes = {
    0: 'No formal job',
    1: 'Farming',
    2: 'Self-employed',
    3: 'Government employee',
    4: 'Private employee',
    5: 'Student',
    6: 'Business owner',
    7: 'Casual worker',
    8: 'Professional',
    9: 'Trader',
    10: 'Manufacturing',
    11: 'Construction',
    12: 'Transport',
    13: 'Education',
    14: 'Health',
    15: 'Other'
  };

  Future<void> _submitPrediction() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      _isLoading = true;
    });

    try {
      final user = UserProfile(
        locationType: _locationType,
        householdSize: _householdSize,
        ageOfRespondent: _age,
        genderOfRespondent: _gender,
        relationshipWithHead: _relationshipWithHead,
        maritalStatus: _maritalStatus,
        jobType: _jobType,
      );

      final result = await ApiService.predictSingleUser(user);

      setState(() {
        _result = result;
        _isLoading = false;
      });

      _showResultDialog();
    } catch (e) {
      setState(() {
        _isLoading = false;
      });

      _showErrorDialog(e.toString());
    }
  }

  void _showResultDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Prediction Result'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Digital Readiness Level:',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: _getReadinessColor(_result!['digital_readiness_level'])
                    .withOpacity(0.1),
                border: Border.all(
                    color: _getReadinessColor(
                        _result!['digital_readiness_level'])),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                _result!['digital_readiness_level'],
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                  color:
                      _getReadinessColor(_result!['digital_readiness_level']),
                ),
              ),
            ),
            const SizedBox(height: 12),
            Text('Score: ${_result!['prediction'].toStringAsFixed(3)}'),
            Text('Confidence: ${_result!['confidence']}'),
            const SizedBox(height: 12),
            const Text(
              'Digital readiness combines phone access, bank account ownership, and education level.',
              style: TextStyle(fontSize: 12, fontStyle: FontStyle.italic),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  void _showErrorDialog(String error) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Error'),
        content: Text(error),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  Color _getReadinessColor(String level) {
    if (level.contains('High')) return Colors.green;
    if (level.contains('Moderate')) return Colors.orange;
    return Colors.red;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Single User Prediction'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _buildInfoCard(),
              const SizedBox(height: 20),
              _buildFormFields(),
              const SizedBox(height: 20),
              _buildSubmitButton(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildInfoCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Youth Profile Input',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: 8),
            const Text(
              'Enter the details for a youth aged 16-30 to predict their digital readiness level. All 7 fields are required for accurate prediction.',
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildFormFields() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // Age (Field 1)
            TextFormField(
              initialValue: _age.toString(),
              decoration: const InputDecoration(
                labelText: 'Age (16-30)',
                helperText: 'Youth focus: 16 to 30 years old',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.cake),
              ),
              keyboardType: TextInputType.number,
              validator: (value) {
                if (value == null || value.isEmpty) return 'Age is required';
                final age = int.tryParse(value);
                if (age == null || age < 16 || age > 30) {
                  return 'Age must be between 16 and 30 (youth focus)';
                }
                return null;
              },
              onChanged: (value) {
                final age = int.tryParse(value);
                if (age != null) _age = age;
              },
            ),
            const SizedBox(height: 16),

            // Location Type (Field 2)
            DropdownButtonFormField<int>(
              value: _locationType,
              decoration: const InputDecoration(
                labelText: 'Location Type',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.location_on),
              ),
              items: _locationTypes.entries.map((entry) {
                return DropdownMenuItem(
                  value: entry.key,
                  child: Text(entry.value),
                );
              }).toList(),
              onChanged: (value) {
                if (value != null) {
                  setState(() {
                    _locationType = value;
                  });
                }
              },
            ),
            const SizedBox(height: 16),

            // Household Size (Field 3)
            TextFormField(
              initialValue: _householdSize.toString(),
              decoration: const InputDecoration(
                labelText: 'Household Size (1-20)',
                helperText: 'Number of people living in the household',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.home),
              ),
              keyboardType: TextInputType.number,
              validator: (value) {
                if (value == null || value.isEmpty)
                  return 'Household size is required';
                final size = int.tryParse(value);
                if (size == null || size < 1 || size > 20) {
                  return 'Household size must be between 1 and 20';
                }
                return null;
              },
              onChanged: (value) {
                final size = int.tryParse(value);
                if (size != null) _householdSize = size;
              },
            ),
            const SizedBox(height: 16),

            // Gender (Field 4)
            DropdownButtonFormField<int>(
              value: _gender,
              decoration: const InputDecoration(
                labelText: 'Gender',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.person),
              ),
              items: _genders.entries.map((entry) {
                return DropdownMenuItem(
                  value: entry.key,
                  child: Text(entry.value),
                );
              }).toList(),
              onChanged: (value) {
                if (value != null) {
                  setState(() {
                    _gender = value;
                  });
                }
              },
            ),
            const SizedBox(height: 16),

            // Relationship with Head (Field 5)
            DropdownButtonFormField<int>(
              value: _relationshipWithHead,
              decoration: const InputDecoration(
                labelText: 'Relationship with Household Head',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.family_restroom),
              ),
              items: _relationships.entries.map((entry) {
                return DropdownMenuItem(
                  value: entry.key,
                  child: Text(entry.value),
                );
              }).toList(),
              onChanged: (value) {
                if (value != null) {
                  setState(() {
                    _relationshipWithHead = value;
                  });
                }
              },
            ),
            const SizedBox(height: 16),

            // Marital Status (Field 6)
            DropdownButtonFormField<int>(
              value: _maritalStatus,
              decoration: const InputDecoration(
                labelText: 'Marital Status',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.favorite),
              ),
              items: _maritalStatuses.entries.map((entry) {
                return DropdownMenuItem(
                  value: entry.key,
                  child: Text(entry.value),
                );
              }).toList(),
              onChanged: (value) {
                if (value != null) {
                  setState(() {
                    _maritalStatus = value;
                  });
                }
              },
            ),
            const SizedBox(height: 16),

            // Job Type (Field 7)
            DropdownButtonFormField<int>(
              value: _jobType,
              decoration: const InputDecoration(
                labelText: 'Job Type',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.work),
              ),
              items: _jobTypes.entries.map((entry) {
                return DropdownMenuItem(
                  value: entry.key,
                  child: Text(entry.value),
                );
              }).toList(),
              onChanged: (value) {
                if (value != null) {
                  setState(() {
                    _jobType = value;
                  });
                }
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSubmitButton() {
    return ElevatedButton(
      onPressed: _isLoading ? null : _submitPrediction,
      style: ElevatedButton.styleFrom(
        padding: const EdgeInsets.all(16),
        backgroundColor: Colors.blue,
        foregroundColor: Colors.white,
      ),
      child: _isLoading
          ? const Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                CircularProgressIndicator(color: Colors.white),
                SizedBox(width: 16),
                Text('Predicting...'),
              ],
            )
          : const Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.analytics),
                SizedBox(width: 8),
                Text('Predict Digital Readiness'),
              ],
            ),
    );
  }
}
