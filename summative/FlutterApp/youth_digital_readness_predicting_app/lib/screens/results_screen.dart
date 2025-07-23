import 'package:flutter/material.dart';

class ResultsScreen extends StatelessWidget {
  const ResultsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final results =
        ModalRoute.of(context)!.settings.arguments as Map<String, dynamic>;
    final predictions = results['predictions'] as List<dynamic>;
    final summary = results['summary'] as Map<String, dynamic>;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Prediction Results'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildSummaryCard(summary),
            const SizedBox(height: 20),
            _buildDistributionCard(summary),
            const SizedBox(height: 20),
            _buildPredictionsHeader(predictions.length),
            const SizedBox(height: 12),
            ...predictions.asMap().entries.map((entry) {
              return _buildPredictionCard(entry.key + 1, entry.value);
            }).toList(),
          ],
        ),
      ),
    );
  }

  Widget _buildSummaryCard(Map<String, dynamic> summary) {
    return Card(
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.summarize, color: Colors.blue),
                const SizedBox(width: 8),
                Text(
                  'Summary Statistics',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            _buildSummaryRow('Total Users', '${summary['total_processed']}'),
            _buildSummaryRow('Average Digital Readiness',
                '${summary['average_digital_readiness'].toStringAsFixed(3)}'),
            _buildSummaryRow('Minimum Score',
                '${summary['min_digital_readiness'].toStringAsFixed(3)}'),
            _buildSummaryRow('Maximum Score',
                '${summary['max_digital_readiness'].toStringAsFixed(3)}'),
          ],
        ),
      ),
    );
  }

  Widget _buildDistributionCard(Map<String, dynamic> summary) {
    final distribution =
        summary['readiness_distribution'] as Map<String, dynamic>;

    return Card(
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.pie_chart, color: Colors.orange),
                const SizedBox(width: 8),
                Text(
                  'Readiness Distribution',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            ...distribution.entries.map((entry) {
              return _buildDistributionRow(entry.key, entry.value);
            }).toList(),
          ],
        ),
      ),
    );
  }

  Widget _buildPredictionsHeader(int count) {
    return Row(
      children: [
        const Icon(Icons.list, color: Colors.green),
        const SizedBox(width: 8),
        Text(
          'Individual Predictions ($count)',
          style: TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }

  Widget _buildPredictionCard(int userNumber, dynamic prediction) {
    final level = prediction['digital_readiness_level'] as String;
    final score = prediction['prediction'] as double;
    final confidence = prediction['confidence'] as String;
    final userProfile = prediction['user_profile'] as Map<String, dynamic>;

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
                  'User $userNumber',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: _getReadinessColor(level).withOpacity(0.2),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    level,
                    style: TextStyle(
                      color: _getReadinessColor(level),
                      fontWeight: FontWeight.bold,
                      fontSize: 12,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child:
                      _buildMetricContainer('Score', score.toStringAsFixed(3)),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: _buildMetricContainer('Confidence', confidence),
                ),
              ],
            ),
            const SizedBox(height: 12),
            const Text(
              'Profile:',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 4),
            Wrap(
              spacing: 8,
              runSpacing: 4,
              children: [
                _buildProfileChip('Age', '${userProfile['age_of_respondent']}'),
                _buildProfileChip('Location',
                    userProfile['location_type'] == 1 ? 'Urban' : 'Rural'),
                _buildProfileChip(
                    'Gender',
                    userProfile['gender_of_respondent'] == 1
                        ? 'Male'
                        : 'Female'),
                _buildProfileChip(
                    'Household', '${userProfile['household_size']}'),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSummaryRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label),
          Text(
            value,
            style: const TextStyle(fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }

  Widget _buildDistributionRow(String level, int count) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.0),
      child: Row(
        children: [
          Container(
            width: 12,
            height: 12,
            decoration: BoxDecoration(
              color: _getReadinessColor(level),
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 8),
          Expanded(child: Text(level)),
          Text(
            '$count',
            style: const TextStyle(fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }

  Widget _buildMetricContainer(String label, String value) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.grey[100],
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        children: [
          Text(
            label,
            style: const TextStyle(
              fontSize: 12,
              color: Colors.grey,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: const TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 16,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildProfileChip(String label, String value) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.blue[50],
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.blue[200]!),
      ),
      child: Text(
        '$label: $value',
        style: TextStyle(
          fontSize: 12,
          color: Colors.blue[800],
        ),
      ),
    );
  }

  Color _getReadinessColor(String level) {
    if (level.contains('High')) return Colors.green;
    if (level.contains('Moderate')) return Colors.orange;
    return Colors.red;
  }
}
