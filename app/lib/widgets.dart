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
    this.nepali,
  });
  final Color color;
  final String title;
  final String body;

  /// The Nepali phrase, when there is one to show.
  ///
  /// Given top billing over [title] deliberately: it is what the app just said
  /// out loud, and what a Nepali-speaking bystander needs to read. Null for
  /// every non-accepted state, so a rejection cannot display a phrase.
  final String? nepali;

  @override
  Widget build(BuildContext context) {
    final ne = nepali;
    return Container(
      width: double.infinity,
      color: color,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (ne != null)
            Padding(
              padding: const EdgeInsets.only(bottom: 2),
              // Devanagari needs more vertical room than Latin at the same
              // point size: matras sit above the headline and below the
              // baseline, and the default height clips them.
              child: Text(ne,
                  style: const TextStyle(
                      color: Colors.white,
                      fontSize: 26,
                      height: 1.5,
                      fontWeight: FontWeight.bold)),
            ),
          Text(title,
              style: TextStyle(
                  color: ne == null ? Colors.white : Colors.white70,
                  fontSize: ne == null ? 20 : 14,
                  fontWeight:
                      ne == null ? FontWeight.bold : FontWeight.normal)),
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
