import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests

INPUT = Path("input")
INPUT.mkdir(exist_ok=True)

event_path = os.environ.get("GITHUB_EVENT_PATH")
event = {}

if event_path and Path(event_path).exists():
    event = json.loads(Path(event_path).read_text(encoding="utf-8"))

token = os.environ.get("GITHUB_TOKEN", "")

def parse_config(text):
    style = os.environ.get("STYLE_INPUT") or "auto"
    duration = int(os.environ.get("DURATION_INPUT") or "18")

    m = re.search(r"style\s*:\s*([a-zA-Z0-9_\-]+)", text, re.I)
    if m:
        style = m.group(1).strip()

    m = re.search(r"duration\s*:\s*(\d+)", text, re.I)
    if m:
        duration = int(m.group(1))

    m = re.search(r"/render\s+.*?style=([a-zA-Z0-9_\-]+)", text, re.I)
    if m:
        style = m.group(1).strip()

    m = re.search(r"/render\s+.*?duration=(\d+)", text, re.I)
    if m:
        duration = int(m.group(1))

    return style, duration

def find_links(text):
    links = []

    markdown_links = re.findall(r"\[([^\]]+)\]\((https?://[^\)]+)\)", text)
    for name, url in markdown_links:
        links.append((name.strip(), url.strip()))

    raw_links = re.findall(r"(https?://[^\s\)]+)", text)
    for url in raw_links:
        name = Path(urlparse(url).path).name or "asset"
        links.append((name, url.strip()))

    clean = []
    seen = set()
    for name, url in links:
        if url not in seen:
            clean.append((name, url))
            seen.add(url)

    return clean

def classify(name, url):
    s = (name + " " + url).lower()

    if any(x in s for x in [".mp4", ".mov", ".mkv", ".webm", "anime", "video", "clip"]):
        return "video"

    if any(x in s for x in [".mp3", ".wav", ".m4a", ".aac", ".ogg", "music", "audio", "song", "beat"]):
        return "music"

    return "unknown"

def download(url, dest):
    headers = {"User-Agent": "Edit-Beast-V3"}

    if token:
        headers["Authorization"] = f"Bearer {token}"

    print(f"[download] {url}")

    r = requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=180,
        allow_redirects=True
    )
    r.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"[saved] {dest} size={dest.stat().st_size}")

event_name = os.environ.get("GITHUB_EVENT_NAME", "")

video_url = os.environ.get("VIDEO_URL") or ""
music_url = os.environ.get("MUSIC_URL") or ""

issue_text = ""

if event_name == "issue_comment":
    issue = event.get("issue", {})
    comment = event.get("comment", {})
    issue_text = (issue.get("body") or "") + "\n\n" + (comment.get("body") or "")

style, duration = parse_config(issue_text)

video = None
music = None

if video_url:
    video = ("video.mp4", video_url)

if music_url:
    music = ("music.mp3", music_url)

for name, url in find_links(issue_text):
    kind = classify(name, url)

    if kind == "video" and video is None:
        video = (name, url)

    if kind == "music" and music is None:
        music = (name, url)

if not video or not music:
    print("ERROR: video/music not found.")
    print("Make an Issue, attach anime.mp4 and music.mp3, then comment /render")
    sys.exit(1)

video_ext = Path(video[0]).suffix or ".mp4"
music_ext = Path(music[0]).suffix or ".mp3"

video_path = INPUT / f"video{video_ext}"
music_path = INPUT / f"music{music_ext}"

download(video[1], video_path)
download(music[1], music_path)

config = {
    "video": str(video_path),
    "music": str(music_path),
    "style": style,
    "duration": duration
}

Path("input/config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

print("[config]", config)
