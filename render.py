"""Reference render script for tkdasofficial/video-agent.

Copy to the repo root as `render.py` (alongside `requirements.txt`).
Pipeline: Pixazo scene images -> Cloudflare Workers AI narration -> FFmpeg MP4 (out.mp4).
Progress is written back to Supabase so the Lovable UI terminal streams it live.
"""

import json
import os
import subprocess
import sys
import wave

import requests

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SERVICE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
VIDEO_ID = os.environ["VIDEO_ID"]
PROMPT = os.environ.get("PROMPT", "")
NEGATIVE_PROMPT = os.environ.get("NEGATIVE_PROMPT", "")
VOICE_GENDER = os.environ.get("VOICE_GENDER", "female")
IMAGE_STYLE = os.environ.get("IMAGE_STYLE", "cinematic")
ASPECT_RATIO = os.environ.get("ASPECT_RATIO", "9:16")

PIXAZO_API_KEY = os.environ.get("PIXAZO_API_KEY", "")
CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CF_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")

SIZES = {"9:16": (1080, 1920), "16:9": (1920, 1080), "1:1": (1080, 1080)}
WIDTH, HEIGHT = SIZES.get(ASPECT_RATIO, SIZES["9:16"])
SCENE_COUNT = 4


def patch(payload: dict) -> None:
    requests.patch(
        f"{SUPABASE_URL}/rest/v1/videos?id=eq.{VIDEO_ID}",
        headers={
            "apikey": SERVICE_KEY,
            "Authorization": f"Bearer {SERVICE_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=30,
    )


def log(message: str, step: str | None = None, progress: int | None = None) -> None:
    print(message, flush=True)
    payload: dict = {"logs": message}
    if step:
        payload["step"] = step
    if progress is not None:
        payload["progress"] = progress
    patch(payload)


def generate_scenes() -> list[str]:
    log("Generating scene images with Pixazo", "Generating scenes", 20)
    paths = []
    for i in range(SCENE_COUNT):
        path = f"scene_{i}.jpg"
        response = requests.post(
            "https://api.pixazo.com/v1/images/generations",
            headers={"Authorization": f"Bearer {PIXAZO_API_KEY}"},
            json={
                "prompt": f"{PROMPT}, {IMAGE_STYLE}, scene {i + 1} of {SCENE_COUNT}",
                "negative_prompt": NEGATIVE_PROMPT,
                "width": WIDTH,
                "height": HEIGHT,
            },
            timeout=180,
        )
        response.raise_for_status()
        body = response.json()
        image_url = body["data"][0].get("url")
        if image_url:
            with open(path, "wb") as handle:
                handle.write(requests.get(image_url, timeout=180).content)
        else:
            import base64

            with open(path, "wb") as handle:
                handle.write(base64.b64decode(body["data"][0]["b64_json"]))
        paths.append(path)
    return paths


def generate_voice() -> str:
    log("Synthesising narration with Cloudflare AI", "Generating voice", 50)
    model = "@cf/myshell-ai/melotts"
    speaker = "EN-US" if VOICE_GENDER == "female" else "EN-BR"
    response = requests.post(
        f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{model}",
        headers={"Authorization": f"Bearer {CF_API_TOKEN}"},
        json={"prompt": PROMPT, "lang": "en", "speaker": speaker},
        timeout=180,
    )
    response.raise_for_status()
    path = "voice.mp3"
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        import base64

        body = response.json()
        audio = body["result"].get("audio") or body["result"].get("audio_base64")
        with open(path, "wb") as handle:
            handle.write(base64.b64decode(audio))
    else:
        with open(path, "wb") as handle:
            handle.write(response.content)
    return path


def audio_duration(path: str) -> float:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path,
        ],
        capture_output=True, text=True, check=True,
    )
    return max(float(probe.stdout.strip()), 4.0)


def compose(scenes: list[str], voice: str) -> None:
    log("Composing final video with FFmpeg", "Rendering video", 75)
    duration = audio_duration(voice)
    per_scene = duration / len(scenes)
    with open("scenes.txt", "w") as handle:
        for path in scenes:
            handle.write(f"file '{path}'\nduration {per_scene:.3f}\n")
        handle.write(f"file '{scenes[-1]}'\n")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", "scenes.txt",
            "-i", voice,
            "-vf", f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
                   f"crop={WIDTH}:{HEIGHT},format=yuv420p",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            "-r", "30", "-shortest", "-movflags", "+faststart",
            "out.mp4",
        ],
        check=True,
    )


def main() -> int:
    try:
        scenes = generate_scenes()
        voice = generate_voice()
        compose(scenes, voice)
        log("Render complete", "Finished", 95)
        return 0
    except Exception as error:  # noqa: BLE001
        log(f"Render failed: {error}")
        patch({"status": "failed", "error": str(error)[:500]})
        return 1


if __name__ == "__main__":
    sys.exit(main())
