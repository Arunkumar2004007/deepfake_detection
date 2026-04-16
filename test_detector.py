"""
test_detector.py — Diagnostic tool for heuristic_detector.py
─────────────────────────────────────────────────────────────
Run from the project root:
    python test_detector.py <video_path> [<video_path2> ...]

Prints per-signal scores for each video so you can verify calibration.
"""
import sys
import os

# Make sure project imports work
sys.path.insert(0, os.path.dirname(__file__))

from models.heuristic_detector import score_video, THRESHOLD


def analyse(path: str):
    if not os.path.exists(path):
        print(f"  ✗ File not found: {path}")
        return

    print(f"\n{'='*60}")
    print(f"  File : {os.path.basename(path)}")
    print(f"{'='*60}")

    result = score_video(path, max_frames=16)

    verdict = "⚠  DEEPFAKE" if result["is_deepfake"] else "✓  REAL"
    print(f"  Verdict    : {verdict}")
    print(f"  Score      : {result['video_score']:.4f}   (threshold={THRESHOLD})")
    print(f"  Confidence : {result['confidence']:.4f}")
    print(f"  Temp Jitter: {result['temporal_jitter']:.2f}")
    print()
    print("  Per-signal means (0=real, 1=deepfake):")
    sigs = result.get("signals", {})
    labels = {
        "fft"              : "FFT HF energy (low=real texture)",
        "chroma"           : "Face chroma boundary (face-swap)",
        "gradient"         : "Gradient contrast ratio (face vs bg)",
        "saturation"       : "Colour saturation coherence",
        "temporal_flicker" : "Temporal flicker (AI pixel noise)",
    }
    for k, label in labels.items():
        v   = sigs.get(k, 0.0)
        bar = "█" * int(v * 20)
        print(f"    {label:<45} {v:.4f}  |{bar:<20}|")

    print()
    print(f"  Frame-by-frame scores: {result['frame_scores']}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_detector.py <video1.mp4> [video2.mp4 ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        analyse(path)

    print("\nDone.  If verdicts are wrong, report the per-signal values above so")
    print("thresholds can be adjusted in models/heuristic_detector.py.")
