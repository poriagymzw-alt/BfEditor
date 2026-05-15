import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import librosa
import numpy as np

INPUT = Path("input")
OUTPUT = Path("output")
VARIANTS = OUTPUT / "variants"
OUTPUT.mkdir(exist_ok=True)
VARIANTS.mkdir(exist_ok=True)

STYLES = {
    "rage_phonk": {"clip": 0.55, "shake": 22, "contrast": 1.45, "sat": 1.35, "noise": 8, "speed": 1.18},
    "dark_phonk": {"clip": 0.75, "shake": 14, "contrast": 1.35, "sat": 0.95, "noise": 5, "speed": 1.08},
    "velocity": {"clip": 0.42, "shake": 18, "contrast": 1.35, "sat": 1.25, "noise": 4, "speed": 1.45},
    "cinematic": {"clip": 1.25, "shake": 5, "contrast": 1.20, "sat": 0.90, "noise": 2, "speed": 0.95},
    "glitch_beast": {"clip": 0.48, "shake": 30, "contrast": 1.55, "sat": 1.45, "noise": 16, "speed": 1.22},
}

AUTO_STYLES = ["rage_phonk", "dark_phonk", "velocity", "glitch_beast"]

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print(p.stdout)
        raise RuntimeError("Command failed: " + " ".join(cmd))
    return p.stdout

def need(tool):
    if shutil.which(tool) is None:
        raise RuntimeError(tool + " not found")

def norm(x):
    x = np.array(x, dtype=np.float32)
    if len(x) == 0:
        return x
    mn = float(x.min())
    mx = float(x.max())
    if abs(mx - mn) < 1e-9:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def analyze_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frames / fps if frames else 30

    step = max(1, int(fps / 3))
    prev = None
    rows = []
    i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if i % step == 0:
            t = i / fps
            small = cv2.resize(frame, (320, 180))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            motion = 0 if prev is None else float(np.mean(cv2.absdiff(gray, prev)))
            contrast = float(np.std(gray))
            sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            sat = float(np.mean(hsv[:, :, 1]))
            rows.append([t, motion, contrast, sharp, sat])
            prev = gray

        i += 1

    cap.release()

    arr = np.array(rows, dtype=np.float32)
    if len(arr) == 0:
        raise RuntimeError("No frames found")

    motion = norm(arr[:, 1])
    contrast = norm(arr[:, 2])
    sharp = norm(arr[:, 3])
    sat = norm(arr[:, 4])
    score = 1.7 * motion + 0.8 * contrast + 0.4 * sharp + 0.25 * sat

    moments = []
    for idx in range(len(arr)):
        moments.append({
            "t": float(arr[idx, 0]),
            "score": float(score[idx]),
            "motion": float(motion[idx]),
            "contrast": float(contrast[idx]),
        })

    moments = sorted(moments, key=lambda x: x["score"], reverse=True)
    return moments, duration

def analyze_music(music_path, duration):
    y, sr = librosa.load(str(music_path), sr=22050, mono=True)
    music_duration = librosa.get_duration(y=y, sr=sr)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    tempo = float(np.asarray(tempo).mean())
    beat_times = librosa.frames_to_time(beats, sr=sr)

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)

    if music_duration <= duration:
        music_start = 0.0
    else:
        step = rms_times[1] - rms_times[0] if len(rms_times) > 1 else 0.05
        win = max(1, int(duration / step))
        energy = np.convolve(norm(rms), np.ones(win), mode="valid") if win < len(rms) else norm(rms)
        music_start = float(rms_times[int(np.argmax(energy))])

    marks = [0.0]
    for b in beat_times:
        if music_start <= b <= music_start + duration:
            marks.append(float(b - music_start))
    marks.append(float(duration))
    marks = sorted(set(round(x, 3) for x in marks if 0 <= x <= duration))

    if len(marks) < 6:
        base = 60 / tempo if tempo > 0 else 0.5
        base = max(0.35, min(base, 0.75))
        marks = [round(i * base, 3) for i in range(int(duration / base) + 1)]
        if marks[-1] < duration:
            marks.append(float(duration))

    return {"tempo": tempo, "music_start": music_start, "marks": marks}

def pick_moments(moments, video_duration, count):
    picked = []
    for m in moments:
        if m["t"] < 0.5 or m["t"] > video_duration - 1.5:
            continue
        if all(abs(m["t"] - p["t"]) > 0.8 for p in picked):
            picked.append(m)
        if len(picked) >= count:
            break
    return picked or moments[:count]

def build_plan(moments, music_info, style, duration):
    preset = STYLES[style]
    marks = music_info["marks"]
    target_clip = preset["clip"]
    clip_lengths = []

    for a, b in zip(marks[:-1], marks[1:]):
        d = b - a
        if d <= 0.1:
            continue
        if d > target_clip * 1.4:
            n = max(1, int(round(d / target_clip)))
            clip_lengths.extend([d / n] * n)
        else:
            clip_lengths.append(d)

    effects = ["flash", "shake", "zoom", "glitch", "hard"]
    plan = []
    total = 0.0

    for i, d in enumerate(clip_lengths):
        if total >= duration:
            break

        d = min(d, duration - total)
        m = moments[i % len(moments)]

        speed = preset["speed"]
        if style == "velocity":
            speed = [1.65, 1.25, 1.85, 1.10][i % 4]
        elif style == "cinematic":
            speed = [0.90, 0.95, 1.00, 0.92][i % 4]

        input_duration = max(0.20, d * speed)
        start = max(0.0, m["t"] - input_duration * 0.25)

        plan.append({
            "start": start,
            "duration": d,
            "input_duration": input_duration,
            "speed": speed,
            "score": m["score"],
            "effect": effects[i % len(effects)],
        })

        total += d

    return plan

def video_filter(style, effect, out_duration, speed):
    s = STYLES[style]
    shake = s["shake"]
    base = "scale=1240:2205:force_original_aspect_ratio=increase"

    if effect in ["shake", "glitch", "flash"]:
        crop = f"crop=1080:1920:x='(iw-ow)/2+{shake}*sin(55*t)':y='(ih-oh)/2+{max(4, shake//2)}*cos(47*t)'"
    elif effect == "zoom":
        crop = f"crop=1080:1920:x='(iw-ow)/2+{max(3, shake//3)}*sin(28*t)':y='(ih-oh)/2'"
    else:
        crop = "crop=1080:1920:x='(iw-ow)/2':y='(ih-oh)/2'"

    color = f"eq=contrast={s['contrast']}:saturation={s['sat']}:brightness=-0.03,unsharp=5:5:1.0"
    filters = [base, crop, color, "vignette=PI/5"]

    if s["noise"] > 0:
        filters.append(f"noise=alls={s['noise']}:allf=t")

    if effect in ["flash", "glitch"]:
        end = max(0.04, out_duration - 0.06)
        filters.append("drawbox=x=0:y=0:w=iw:h=ih:color=white@0.25:t=fill:enable='lt(t,0.05)'")
        filters.append(f"drawbox=x=0:y=0:w=iw:h=ih:color=white@0.18:t=fill:enable='between(t,{end:.3f},{out_duration:.3f})'")

    filters.append("fps=30")
    filters.append(f"setpts=PTS/{speed:.4f}")
    return ",".join(filters)

def render_segment(video, clip, style, out):
    vf = video_filter(style, clip["effect"], clip["duration"], clip["speed"])
    run([
        "ffmpeg", "-y",
        "-ss", f"{clip['start']:.3f}",
        "-t", f"{clip['input_duration']:.3f}",
        "-i", str(video),
        "-an",
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "19",
        "-pix_fmt", "yuv420p",
        str(out),
    ])

def concat(parts, out):
    list_file = out.parent / "concat.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for p in parts:
            f.write(f"file '{p.as_posix()}'\n")

    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(out)])

def add_music(video_only, music, out, music_start, duration):
    run([
        "ffmpeg", "-y",
        "-i", str(video_only),
        "-ss", f"{music_start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(music),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(out)
    ])

def render_variant(video, music, style, moments, video_duration, music_info, duration):
    picked = pick_moments(moments, video_duration, max(20, int(duration / 0.4)))
    plan = build_plan(picked, music_info, style, duration)
    out_path = VARIANTS / f"{style}.mp4"

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        parts = []

        for i, clip in enumerate(plan):
            part = td / f"part_{i:04d}.mp4"
            render_segment(video, clip, style, part)
            parts.append(part)

        video_only = td / "video_only.mp4"
        concat(parts, video_only)

        total = sum(c["duration"] for c in plan)
        add_music(video_only, music, out_path, music_info["music_start"], total)

    score = float(np.mean([c["score"] for c in plan])) + len(plan) * 0.002
    return {"style": style, "path": str(out_path), "score": score, "plan": plan, "top_moments": picked[:12]}

def make_preview(final_video):
    try:
        run(["ffmpeg", "-y", "-i", str(final_video), "-t", "7", "-vf", "fps=8,scale=360:-1:flags=lanczos", str(OUTPUT / "preview.gif")])
    except Exception:
        pass

    try:
        run(["ffmpeg", "-y", "-ss", "1", "-i", str(final_video), "-frames:v", "1", "-q:v", "2", str(OUTPUT / "thumbnail.jpg")])
    except Exception:
        pass

def main():
    need("ffmpeg")
    need("ffprobe")

    config = json.loads((INPUT / "config.json").read_text(encoding="utf-8"))

    video = Path(config["video"])
    music = Path(config["music"])
    style = config.get("style", "auto")
    duration = int(config.get("duration", 18))

    if duration > 35:
        duration = 35

    moments, video_duration = analyze_video(video)
    music_info = analyze_music(music, duration)

    if style == "auto":
        styles = AUTO_STYLES
    elif style in STYLES:
        styles = [style]
    else:
        styles = AUTO_STYLES

    results = []
    for st in styles:
        results.append(render_variant(video, music, st, moments, video_duration, music_info, duration))

    best = max(results, key=lambda x: x["score"])
    final = OUTPUT / "final_edit.mp4"
    shutil.copyfile(best["path"], final)

    make_preview(final)

    caption = f"Edit Beast V3 Auto Edit\nStyle: {best['style']}\nTempo: {music_info['tempo']:.1f} BPM\n"
    hashtags = "#animeedit #amv #phonk #velocityedit #rageedit #darkphonk #shorts #reels #edit"

    (OUTPUT / "caption.txt").write_text(caption, encoding="utf-8")
    (OUTPUT / "hashtags.txt").write_text(hashtags, encoding="utf-8")

    report = []
    report.append("# Edit Beast V3 Report")
    report.append("")
    report.append(f"Winner: `{best['style']}`")
    report.append(f"Score: `{best['score']:.4f}`")
    report.append(f"Tempo: `{music_info['tempo']:.2f}` BPM")
    report.append(f"Music start: `{music_info['music_start']:.2f}s`")
    report.append("")
    report.append("## Effects")
    report.append("- scene hunting")
    report.append("- beat detection")
    report.append("- heavy shake")
    report.append("- speed cuts")
    report.append("- flash transitions")
    report.append("- rage/dark color")
    report.append("- glitch noise")
    report.append("- 9:16 vertical")
    report.append("")
    report.append("## Variants")
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        report.append(f"- {r['style']} score={r['score']:.4f}")

    (OUTPUT / "edit_report.md").write_text("\n".join(report), encoding="utf-8")

    plan = {
        "version": "Edit Beast V3",
        "winner": best["style"],
        "duration": duration,
        "music_info": music_info,
        "results": results,
    }

    (OUTPUT / "edit_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    print("DONE")
    print("winner:", best["style"])
    print("output/final_edit.mp4")

if __name__ == "__main__":
    main()
