import json
import os
import sys
import zipfile
from pathlib import Path

import requests

INPUT = Path("input")
INPUT.mkdir(exist_ok=True)

token = os.environ.get("GITHUB_TOKEN", "")

def download(url, dest):
    headers = {"User-Agent": "Edit-Beast-V3"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    print("[download]", url)
    r = requests.get(url, headers=headers, stream=True, timeout=300, allow_redirects=True)
    r.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    print("[saved]", dest, "size=", dest.stat().st_size)

def find_first_file(folder, exts):
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            return p
    return None

package_url = os.environ.get("PACKAGE_URL") or ""
style = os.environ.get("STYLE_INPUT") or "rage_phonk"
duration = int(os.environ.get("DURATION_INPUT") or "18")

if not package_url:
    print("ERROR: PACKAGE_URL is empty")
    sys.exit(1)

zip_path = INPUT / "package.zip"
extract_dir = INPUT / "package"
extract_dir.mkdir(exist_ok=True)

download(package_url, zip_path)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_dir)

video_found = find_first_file(extract_dir, {".mp4", ".mov", ".mkv", ".webm"})
music_found = find_first_file(extract_dir, {".mp3", ".wav", ".m4a", ".aac", ".ogg"})

if not video_found:
    print("ERROR: No video file found inside ZIP")
    sys.exit(1)

if not music_found:
    print("ERROR: No music file found inside ZIP")
    sys.exit(1)

video_path = INPUT / "anime.mp4"
music_path = INPUT / "music.mp3"

video_path.write_bytes(video_found.read_bytes())
music_path.write_bytes(music_found.read_bytes())

config = {
    "video": str(video_path),
    "music": str(music_path),
    "style": style,
    "duration": duration
}

Path("input/config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
print("[config]", config)
