import cv2
import json
import base64
import subprocess
import time
import numpy as np
import os
from pathlib import Path
from PIL import Image
from imagehash import phash

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VIDEO_IN               = Path("/Users/abhiveer/Downloads/Class_8_cctv_video_1.mov")
VIDEO_OUT              = Path("compressed_output.mp4")
REPORT_HTML_OUT        = Path("compression_report.html")
SEGMENTS_JSON_OUT      = Path("segments_kept.json")

PHASH_THRESHOLD        = 3      # Hamming distance threshold for imagehash.phash
MOTION_KEEP_THRESH     = 0.15   # Standard threshold for motion
CONTEXT_EVERY_SEC      = 3      # Step 4: Force frame every 3s
OUTPUT_FPS             = 12     # Step 5: Output framerate
OUTPUT_CRF             = 28     

# ---------------------------------------------------------------------------
# PERCEPTUAL HASH (Step 1)
# ---------------------------------------------------------------------------

def compute_phash(frame: np.ndarray):
    """Uses imagehash library as per your logic."""
    # Small grayscale for speed
    gray_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
    return phash(Image.fromarray(gray_small))

def phash_similarity(h1, h2) -> float:
    """Returns Hamming distance for direct comparison."""
    if h1 is None or h2 is None: return 999
    return h1 - h2

# ---------------------------------------------------------------------------
# FACE PRESENCE CHECK (Step 3)
# ---------------------------------------------------------------------------

def has_face(frame: np.ndarray, cascade) -> bool:
    """Detection on a small proxy for speed."""
    gray_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
    faces = cascade.detectMultiScale(gray_small, 1.2, 3)
    return len(faces) > 0

# ---------------------------------------------------------------------------
# THUMBNAIL HELPER (For JSON Contract)
# ---------------------------------------------------------------------------

def frame_to_b64_thumb(frame: np.ndarray) -> str:
    h, w = frame.shape[:2]
    thumb = cv2.resize(frame, (200, int(h * 200 / w)))
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buf).decode("utf-8")

# ---------------------------------------------------------------------------
# VIDEO WRITING (Step 5 - Mac Accelerated)
# ---------------------------------------------------------------------------

def write_frames_to_video(kept_frames: list, output_path: Path, fps: float, frame_size: tuple):
    temp_raw = "temp_raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_raw, fourcc, 12, frame_size)
    
    for f in kept_frames:
        out.write(f)
    out.release()

    # Hardware Acceleration for Mac (The <60s secret)
    subprocess.run([
        "ffmpeg", "-y", "-i", temp_raw, 
        "-vcodec", "h264_videotoolbox", 
        "-b:v", "2M", str(output_path)
    ])
    if os.path.exists(temp_raw): os.remove(temp_raw)

# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(str(VIDEO_IN))
    if not cap.isOpened():
        print("❌ Could not open video."); exit()

    total_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw, fh = int(cap.get(3)), int(cap.get(4))
    orig_mb = VIDEO_IN.stat().st_size / 1_000_000

    kept_frames, segments = [], []
    last_kept_hash, last_kept_t = None, -999.0
    cur_seg = None
    disc_dup, disc_stat = 0, 0

    print(f"🚀 Processing {total_in} frames (720p Optimized)...")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Speed Optimization: Process 1 out of every 2 frames
        if frame_idx % 2 != 0:
            frame_idx += 1
            continue

        ts = frame_idx / fps_in
        curr_hash = compute_phash(frame)
        
        keep, reason = False, ""
        
        # 1. pHash Check
        if last_kept_hash is None or phash_similarity(curr_hash, last_kept_hash) > PHASH_THRESHOLD:
            keep, reason = True, "perceptual_hash"
        
        # 3. Face Check (If pHash didn't trigger)
        if not keep and has_face(frame, cascade):
            keep, reason = True, "face_detected"
        
        # 4. Context Check (Heartbeat every 3s)
        if not keep and (ts - last_kept_t) >= CONTEXT_EVERY_SEC:
            keep, reason = True, "scene_continuity"

        if keep:
            kept_frames.append(frame.copy())
            last_kept_hash = curr_hash
            last_kept_t = ts
            
            # Segment Tracking for JSON contract
            if cur_seg is None or (ts - cur_seg["end_sec"]) > 2.0 or reason != cur_seg["reason_kept"]:
                if cur_seg: segments.append(cur_seg)
                cur_seg = {
                    "segment_id": len(segments) + 1, "start_sec": round(ts, 2), "end_sec": round(ts, 2),
                    "frames_in_segment": 1, "reason_kept": reason,
                    "thumbnail_b64": frame_to_b64_thumb(frame)
                }
            else:
                cur_seg["end_sec"] = round(ts, 2)
                cur_seg["frames_in_segment"] += 1
        
        frame_idx += 1

    if cur_seg: segments.append(cur_seg)
    cap.release()

    print("📦 Encoding video...")
    write_frames_to_video(kept_frames, VIDEO_OUT, OUTPUT_FPS, (fw, fh))

    # Deliverables
    proc_time = time.time() - t_start
    comp_mb = VIDEO_OUT.stat().st_size / 1_000_000
    
    stats = {
        "original_size_mb": round(orig_mb, 2),
        "compressed_size_mb": round(comp_mb, 2),
        "reduction_pct": round((1 - comp_mb/orig_mb)*100, 1),
        "processing_time_sec": round(proc_time, 2),
        "segments": segments
    }

    with open(SEGMENTS_JSON_OUT, 'w') as f:
        json.dump(stats, f, indent=4)

    # HTML Report
    html = f"<html><body><h1>Sentio Mind Report</h1><p>Time: {proc_time:.2f}s</p><p>Reduction: {stats['reduction_pct']}%</p></body></html>"
    with open(REPORT_HTML_OUT, "w") as f: f.write(html)

    print(f"✅ Done in {proc_time:.2f}s | Reduction: {stats['reduction_pct']}%")