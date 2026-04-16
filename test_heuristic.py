import sys
sys.path.insert(0, '.')

print("Testing heuristic_detector v5...", flush=True)

try:
    import numpy as np
    from models.heuristic_detector import THRESHOLD, score_frame_detailed, score_video

    print(f"THRESHOLD = {THRESHOLD}", flush=True)

    # Test 1: blank frame
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    r = score_frame_detailed(blank)
    print(f"Blank frame score: {r['score']}", flush=True)
    print(f"Signal keys: {list(r.keys())}", flush=True)

    # Test 2: random noise frame (real-like)
    rng = np.random.RandomState(42)
    noise = (rng.randn(200, 200, 3) * 30 + 128).clip(0, 255).astype('uint8')
    r2 = score_frame_detailed(noise)
    print(f"Noise frame score: {r2['score']} (fft={r2['fft']}, skin={r2['skin']})", flush=True)

    print("ALL TESTS PASSED", flush=True)

except Exception as e:
    import traceback
    print(f"ERROR: {e}", flush=True)
    traceback.print_exc()
