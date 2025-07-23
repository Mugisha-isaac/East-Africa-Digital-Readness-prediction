import 'package:flutter/material.dart';
import '../services/api_service.dart';

class BatchPredictionScreen extends StatefulWidget {
  const BatchPredictionScreen({super.key});

  @override
  State<BatchPredictionScreen> createState() => _BatchPredictionScreenState();
}

class _BatchPredictionScreenState extends State<BatchPredictionScreen> {
  bool _isLoading = false;
  Map<String, dynamic>? _results;
  List<UserProfile> _users = [];

  @override
  void initState() {
    super.initState();
    _addSampleUsers();
  }

  void _addSampleUsers() {
    _users = [
      UserProfile(
        locationType: 1, // Urban
        householdSize: 4,
        ageOfRespondent: 22,
        genderOfRespondent: 1, // Male
        relationshipWithHead: 2, // Child
        maritalStatus: 0, // Single
        jobType: 8, // Professional
      ),
      UserProfile(
        locationType: 0, // Rural
        householdSize: 6,
        ageOfRespondent: 19,
        genderOfRespondent: 0, // Female
        relationshipWithHead: 2, // Child
        maritalStatus: 0, // Single
        jobType: 1, // Farming
      ),
      UserProfile(
        locationType: 1, // Urban
        householdSize: 3,
        ageOfRespondent: 28,
        genderOfRespondent: 1, // Male
        relationshipWithHead: 0, // Head
        maritalStatus: 1, // Married
        jobType: 6, // Business owner
      ),
    ];
  }

  Future<void> _submitBatchPrediction() async {
    if (_users.isEmpty) {
      _showErrorDialog('Please add at least one user profile');
      return;
    }

    setState(() {
      _isLoading = true;
    });

    try {
      final results = await ApiService.predictMultipleUsers(_users);

      setState(() {
        _results = results;
        _isLoading = false;
      });

      Navigator.pushNamed(
        context,
        '/results',
        arguments: results,
      );
    } catch (e) {
      setState(() {
        _isLoading = false;
      });

      _showErrorDialog(e.toString());
    }
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

  void _addUser() {
    setState(() {
      _users.add(UserProfile(
        locationType: 1,
        householdSize: 4,
        ageOfRespondent: 20 + _users.length,
        genderOfRespondent: _users.length % 2,
        relationshipWithHead: 2,
        maritalStatus: 0,
        jobType: 3,
      ));
    });
  }

  void _removeUser(int index) {
    setState(() {
      _users.removeAt(index);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Batch Prediction'),
        actions: [
          IconButton(
            onPressed: _addUser,
            icon: const Icon(Icons.add),
            tooltip: 'Add User',
          ),
        ],
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Batch Prediction',
                      style: Theme.of(context).textTheme.titleLarge,
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Predict digital readiness for multiple users at once. You can add up to 100 users.',
                      style: Theme.of(context).textTheme.bodyMedium,
                    ),
                    const SizedBox(height: 12),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text('Users added: ${_users.length}'),
                        ElevatedButton.icon(
                          onPressed: _isLoading ? null : _submitBatchPrediction,
                          icon: _isLoading
                              ? const SizedBox(
                                  width: 16,
                                  height: 16,
                                  child:
                                      CircularProgressIndicator(strokeWidth: 2),
                                )
                              : const Icon(Icons.analytics),
                          label: const Text('Predict All'),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
          Expanded(
            child: _users.isEmpty
                ? const Center(
                    child: Text('No users added. Tap + to add users.'),
                  )
                : ListView.builder(
                    padding: const EdgeInsets.symmetric(horizontal: 16.0),
                    itemCount: _users.length,
                    itemBuilder: (context, index) {
                      return _buildUserCard(index);
                    },
                  ),
          ),
        ],
      ),
    );
  }

  Widget _buildUserCard(int index) {
    final user = _users[index];
    return Card(
      margin: const EdgeInsets.only(bottom: 8.0),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'User ${index + 1}',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                IconButton(
                  onPressed: () => _removeUser(index),
                  icon: const Icon(Icons.delete, color: Colors.red),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 16,
              runSpacing: 8,
              children: [
                _buildUserInfo('Age', '${user.ageOfRespondent}'),
                _buildUserInfo(
                    'Location', user.locationType == 1 ? 'Urban' : 'Rural'),
                _buildUserInfo(
                    'Gender', user.genderOfRespondent == 1 ? 'Male' : 'Female'),
                _buildUserInfo('Household Size', '${user.householdSize}'),
                _buildUserInfo('Job Type', '${user.jobType}'),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildUserInfo(String label, String value) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.grey[200],
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(
        '$label: $value',
        style: const TextStyle(fontSize: 12),
      ),
    );
  }
}
