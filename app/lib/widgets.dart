import 'package:flutter/material.dart';

/// Small UI pieces shared by the offline and server pages.
///
/// Extracted so the two recognition paths look identical to the user and to a
/// reviewer — the only difference between them should be where inference runs.

/// Boolean telemetry pill (pose / hands / server). Tappable when there is an
/// action attached, e.g. reconnecting.
class StatusChip extends StatelessWidget {
  const StatusChip(this.label, this.on, {super.key, this.onTap});
  final String label;
  final bool on;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
        decoration: BoxDecoration(
          color: on ? Colors.green.shade700 : Colors.red.shade900,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Text(label,
            style: const TextStyle(color: Colors.white, fontSize: 12)),
      ),
    );
  }
}

/// Numeric telemetry pill (frames / fps / ms).
class InfoChip extends StatelessWidget {
  const InfoChip(this.label, this.value, {super.key});
  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.grey.shade800,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text('$label $value',
          style: const TextStyle(color: Colors.white70, fontSize: 12)),
    );
  }
}

/// The result panel.
class ResultBanner extends StatelessWidget {
  const ResultBanner({
    super.key,
    required this.color,
    required this.title,
    required this.body,
  });
  final Color color;
  final String title;
  final String body;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      color: color,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title,
              style: const TextStyle(
                  color: Colors.white,
                  fontSize: 20,
                  fontWeight: FontWeight.bold)),
          if (body.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text(body,
                  style: const TextStyle(color: Colors.white70, fontSize: 12)),
            ),
        ],
      ),
    );
  }
}

/// Full-page message for loading and fatal states.
class CenteredMessage extends StatelessWidget {
  const CenteredMessage({super.key, required this.icon, required this.text});
  final IconData icon;
  final String text;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: Colors.white54, size: 48),
            const SizedBox(height: 12),
            Text(text,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.white70)),
          ],
        ),
      ),
    );
  }
}
