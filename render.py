"""Reference render script for tkdasofficial/video-agent.

Copy to the repo root as `render.py` (alongside `requirements.txt`).

Pipeline
--------
1. Cloudflare Workers AI (`@cf/meta/llama-3-8b-instruct`) writes the narration
   script and the per-scene image prompts (the "brain").
2. Pixazo AI (https://api.pixazo.ai) renders the scene images (free-tier image
   model) and synthesises the voiceover (Pixazo TTS).
3. FFmpeg composes scenes + narration, applies the motion template, and burns
   captions from a generated SRT when captions are enabled.

Progress is written back to Supabase so the Lovable UI terminal streams it live.
"""

import base64
import hashlib
import json
import math
import os
import subprocess
import sys

import requests

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SERVICE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
VIDEO_ID = os.environ["VIDEO_ID"]

def env(name: str, default: str = "") -> str:
    """Unset or blank GitHub Action inputs arrive as empty strings."""
    return (os.environ.get(name) or "").strip() or default


PROMPT = env("PROMPT")
NEGATIVE_PROMPT = env("NEGATIVE_PROMPT")
VOICE_GENDER = env("VOICE_GENDER", "female").lower()
VOICE_PERSONA = env("VOICE_PERSONA", "Cinematic Narrator")
IMAGE_STYLE = env("IMAGE_STYLE", "Cinematic 3D")
MOTION_TEMPLATE = env("MOTION_TEMPLATE", "Auto Zoom-In")
ASPECT_RATIO = env("ASPECT_RATIO", "9:16")
CAPTION_STYLE = env("CAPTION_STYLE", "Neon Glow")
CAPTIONS = env("CAPTIONS", "false").lower() in {"1", "true", "yes", "on"}
QUALITY = env("QUALITY", "1080p")
BITRATE = env("BITRATE", "High")

try:
    DURATION = max(1, min(60, int(float(env("DURATION_SECONDS", "15")))))
except ValueError:
    DURATION = 15

# --- Providers -------------------------------------------------------------
PIXAZO_API_KEY = env("PIXAZO_API_KEY")
# A blank PIXAZO_BASE_URL used to leave the request URL schemeless
# ("No scheme supplied"), so always normalise it to an absolute https URL.
PIXAZO_BASE_URL = env("PIXAZO_BASE_URL", "https://api.pixazo.ai").rstrip("/")
if not PIXAZO_BASE_URL.startswith(("http://", "https://")):
    PIXAZO_BASE_URL = f"https://{PIXAZO_BASE_URL}"
PIXAZO_IMAGE_MODEL = env("PIXAZO_IMAGE_MODEL", "pixazo-image-free")
PIXAZO_TTS_MODEL = env("PIXAZO_TTS_MODEL", "pixazo-tts-1")

CF_ACCOUNT_ID = env("CLOUDFLARE_ACCOUNT_ID")
CF_API_TOKEN = env("CLOUDFLARE_API_TOKEN")
CF_IMAGE_MODELS = [
    model.strip()
    for model in env(
        "CF_IMAGE_MODEL",
        "@cf/black-forest-labs/flux-1-schnell,"
        "@cf/stabilityai/stable-diffusion-xl-base-1.0,"
        "@cf/bytedance/stable-diffusion-xl-lightning",
    ).split(",")
    if model.strip()
]
# Workers AI retires older model ids (HTTP 410 Gone), so try current ones in order.
CF_LLM_MODELS = [
    model.strip()
    for model in env(
        "CF_LLM_MODEL",
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast,"
        "@cf/meta/llama-3.1-8b-instruct,"
        "@cf/meta/llama-3.1-8b-instruct-fast,"
        "@cf/mistralai/mistral-small-3.1-24b-instruct",
    ).split(",")
    if model.strip()
]

SIZES_1080 = {"9:16": (1080, 1920), "16:9": (1920, 1080), "1:1": (1080, 1080)}
SIZES_720 = {"9:16": (720, 1280), "16:9": (1280, 720), "1:1": (720, 720)}
SIZES = SIZES_720 if QUALITY == "720p" else SIZES_1080
WIDTH, HEIGHT = SIZES.get(ASPECT_RATIO, SIZES["9:16"])
VIDEO_BITRATE = "5M" if BITRATE == "High" else "2500k"

# One scene per ~5 seconds, at least two, at most eight.
SCENE_COUNT = max(2, min(8, math.ceil(DURATION / 5)))


# --- Supabase progress -----------------------------------------------------
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


# --- Stage A: the brain (Cloudflare Workers AI) -----------------------------
def cloudflare_run(model: str, body: dict, *, timeout: int = 180) -> requests.Response:
    if not (CF_ACCOUNT_ID and CF_API_TOKEN):
        raise RuntimeError("CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN are not configured")
    response = requests.post(
        f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{model}",
        headers={"Authorization": f"Bearer {CF_API_TOKEN}"},
        # Workers AI rejects null values, so never send empty fields.
        json={key: value for key, value in body.items() if value not in (None, "")},
        timeout=timeout,
    )
    if not response.ok:
        raise RuntimeError(
            f"Workers AI {model} failed ({response.status_code}): {response.text[:300]}"
        )
    return response


def write_script() -> tuple[str, list[str]]:
    """Returns (narration script, one image prompt per scene)."""
    log(f"Writing the {DURATION}s script with Cloudflare Workers AI", "Writing script", 12)
    words = max(12, int(DURATION * 2.4))  # ~145 wpm narration
    instruction = (
        f"Topic: {PROMPT}\n"
        f"Write a narration script of about {words} words for a {DURATION} second video, "
        f"then {SCENE_COUNT} vivid image prompts in the '{IMAGE_STYLE}' style.\n"
        f"Avoid: {NEGATIVE_PROMPT or 'nothing in particular'}.\n"
        'Reply with JSON only: {"script": "...", "scenes": ["...", "..."]}'
    )
    body = {
        "messages": [
            {
                "role": "system",
                "content": "You are a short-form video director. Reply with strict JSON only.",
            },
            {"role": "user", "content": instruction},
        ],
        "max_tokens": 900,
    }
    text = ""
    for model in CF_LLM_MODELS:
        try:
            raw = cloudflare_run(model, body).json()
        except Exception as error:  # noqa: BLE001 - retired/unavailable model, try the next
            log(f"Script model {model} unavailable ({error}); trying the next one")
            continue

        # Cloudflare Workers AI response shape changed over time:
        # { "result": { "response": "..." } } or { "response": "..." }
        result = raw.get("result") or raw
        if isinstance(result, dict):
            candidate = result.get("response")
            if candidate is None:
                candidate = result.get("result")
            text = str(candidate) if candidate is not None else ""
        else:
            text = str(result) if result is not None else ""
        if text and text != "None":
            break

    # Always coerce to a plain string before string operations.
    text = str(text)
    start, end = text.find("{"), text.rfind("}")
    script, scenes = "", []
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            script = str(parsed.get("script") or "").strip()
            scenes = [str(s).strip() for s in (parsed.get("scenes") or []) if str(s).strip()]
        except json.JSONDecodeError:
            pass

    if not script:
        log("Script model returned no usable JSON; narrating the prompt directly", None, None)
        script = PROMPT
    while len(scenes) < SCENE_COUNT:
        scenes.append(f"{PROMPT}, {IMAGE_STYLE}, scene {len(scenes) + 1}")
    return script, scenes[:SCENE_COUNT]


# --- Stage B: Pixazo AI images + TTS ---------------------------------------
def pixazo_post(path: str, body: dict, *, timeout: int = 240) -> requests.Response:
    if not PIXAZO_API_KEY:
        raise RuntimeError("PIXAZO_API_KEY is not configured")
    response = requests.post(
        f"{PIXAZO_BASE_URL}{path}",
        headers={
            "Authorization": f"Bearer {PIXAZO_API_KEY}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout,
    )
    response.raise_for_status()
    return response


def save_binary_or_b64(response: requests.Response, path: str, keys: tuple[str, ...]) -> None:
    """Writes a provider response to `path`, accepting raw bytes, base64 or a URL."""
    if "application/json" not in response.headers.get("content-type", ""):
        with open(path, "wb") as handle:
            handle.write(response.content)
        return

    body = response.json()
    candidates: list[dict] = [body]
    if isinstance(body.get("data"), list) and body["data"]:
        candidates.insert(0, body["data"][0])
    if isinstance(body.get("result"), dict):
        candidates.insert(0, body["result"])
    if isinstance(body.get("output"), dict):
        candidates.insert(0, body["output"])

    for item in candidates:
        for key in ("url", "audio_url", "image_url"):
            if isinstance(item.get(key), str) and item[key].startswith("http"):
                with open(path, "wb") as handle:
                    handle.write(requests.get(item[key], timeout=240).content)
                return
        for key in keys:
            value = item.get(key)
            if isinstance(value, str) and value:
                payload = value.split(",", 1)[-1] if value.startswith("data:") else value
                with open(path, "wb") as handle:
                    handle.write(base64.b64decode(payload))
                return
    raise RuntimeError(f"Pixazo response contained no media: {json.dumps(body)[:300]}")


def cloudflare_image(scene_prompt: str, path: str) -> None:
    """Fallback image generator so a Pixazo outage does not fail the whole render."""
    # FLUX.1 Schnell's current REST contract needs only a prompt. Avoid sending
    # dimensions or sampling fields shared by other image models: Workers AI
    # rejects unsupported fields with HTTP 400.
    prompt = f"{scene_prompt}, {IMAGE_STYLE}"[:1900]
    errors: list[str] = []
    for model in CF_IMAGE_MODELS:
        if "flux" in model:
            seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
            body = {"prompt": prompt, "seed": seed}
        else:
            body = {
                "prompt": prompt,
                "negative_prompt": NEGATIVE_PROMPT,
                "width": min(WIDTH, 1024),
                "height": min(HEIGHT, 1024),
            }
        try:
            response = cloudflare_run(model, body)
            save_binary_or_b64(response, path, ("b64_json", "image", "image_base64"))
            return
        except Exception as error:  # noqa: BLE001 - try the next image model
            errors.append(f"{model}: {error}")
    raise RuntimeError("no Workers AI image model succeeded — " + " | ".join(errors))


def generate_scenes(scene_prompts: list[str]) -> list[str]:
    log(f"Generating {len(scene_prompts)} scene images with Pixazo AI", "Generating scenes", 28)
    paths: list[str] = []
    fallback_notified = False
    for index, scene_prompt in enumerate(scene_prompts):
        path = f"scene_{index}.jpg"
        try:
            response = pixazo_post(
                "/v1/images/generations",
                {
                    "model": PIXAZO_IMAGE_MODEL,
                    "prompt": f"{scene_prompt}, {IMAGE_STYLE}",
                    "negative_prompt": NEGATIVE_PROMPT,
                    "width": WIDTH,
                    "height": HEIGHT,
                    "n": 1,
                },
            )
            save_binary_or_b64(response, path, ("b64_json", "image", "image_base64"))
        except Exception as error:  # noqa: BLE001 - fall back to Cloudflare images
            if not fallback_notified:
                log(f"Pixazo images unavailable ({error}); using Cloudflare Workers AI images")
                fallback_notified = True
            cloudflare_image(scene_prompt, path)
        paths.append(path)
        log(f"scene {index + 1}/{len(scene_prompts)} ready")
    return paths


def generate_voice(script: str) -> str:
    log(f"Synthesising narration with Pixazo TTS ({VOICE_GENDER})", "Generating voice", 52)
    path = "voice.mp3"
    response = pixazo_post(
        "/v1/audio/speech",
        {
            "model": PIXAZO_TTS_MODEL,
            "input": script,
            "voice": VOICE_GENDER,
            "persona": VOICE_PERSONA,
            "response_format": "mp3",
        },
    )
    save_binary_or_b64(response, path, ("b64_json", "audio", "audio_base64"))
    return path


# --- Stage C: captions + FFmpeg composition --------------------------------
def audio_duration(path: str) -> float:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path,
        ],
        capture_output=True, text=True, check=True,
    )
    try:
        return max(float(probe.stdout.strip()), 1.0)
    except ValueError:
        return float(DURATION)


def srt_timestamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(script: str, total: float, path: str = "captions.srt") -> str:
    """Distributes the narration across short cues proportional to word count."""
    words = script.split()
    if not words:
        words = [PROMPT or "…"]
    per_cue = 7
    cues = [words[i : i + per_cue] for i in range(0, len(words), per_cue)]
    weight = sum(len(" ".join(c)) for c in cues) or 1
    clock = 0.0
    with open(path, "w", encoding="utf-8") as handle:
        for index, cue in enumerate(cues, start=1):
            text = " ".join(cue)
            span = max(0.6, total * (len(text) / weight))
            start, end = clock, min(total, clock + span)
            clock = end
            handle.write(f"{index}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n{text}\n\n")
    log(f"Burning {len(cues)} caption cues ({CAPTION_STYLE})", "Rendering captions", 68)
    return path


def caption_filter(srt_path: str) -> str:
    """ASS style per caption preset, applied through the FFmpeg subtitles filter."""
    base = f"Fontsize={max(16, HEIGHT // 34)},Alignment=2,MarginV={max(40, HEIGHT // 14)},Outline=2,Shadow=1"
    styles = {
        "Yellow Pop-Up": f"FontName=DejaVu Sans,{base},Bold=1,PrimaryColour=&H0000E5FF,OutlineColour=&H00000000",
        "Neon Glow": f"FontName=DejaVu Sans,{base},Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00FF00CC,Outline=3",
        "Minimalist White": f"FontName=DejaVu Sans,{base},PrimaryColour=&H00FFFFFF,OutlineColour=&H64000000",
        "Monospace Subtitles": f"FontName=DejaVu Sans Mono,{base},PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000",
    }
    style = styles.get(CAPTION_STYLE, styles["Minimalist White"])
    return f"subtitles={srt_path}:force_style='{style}'"


def motion_filter(per_scene: float) -> str:
    """Ken Burns / transition behaviour for the selected motion template."""
    frames = max(2, int(per_scene * 30))
    if MOTION_TEMPLATE == "Pan & Scan":
        return f"zoompan=z=1.15:x='iw/2-(iw/zoom/2)+sin(on/40)*80':y='ih/2-(ih/zoom/2)':d={frames}:s={WIDTH}x{HEIGHT}:fps=30"
    if MOTION_TEMPLATE == "Dynamic Keyframe":
        return f"zoompan=z='if(lte(mod(on,{frames}),{frames}/2),1.0+0.25*mod(on,{frames})/{frames},1.25-0.2*mod(on,{frames})/{frames})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={WIDTH}x{HEIGHT}:fps=30"
    if MOTION_TEMPLATE == "Fade Transitions":
        return f"zoompan=z=1.02:d={frames}:s={WIDTH}x{HEIGHT}:fps=30,fade=t=in:st=0:d=0.4"
    # Auto Zoom-In (default)
    return f"zoompan=z='min(zoom+0.0012,1.30)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={WIDTH}x{HEIGHT}:fps=30"


def compose(scenes: list[str], voice: str, script: str) -> None:
    log("Composing final video with FFmpeg", "Rendering video", 78)
    narration = audio_duration(voice)
    total = max(1.0, min(float(DURATION), narration)) if narration > DURATION else float(DURATION)
    per_scene = total / len(scenes)

    with open("scenes.txt", "w", encoding="utf-8") as handle:
        for path in scenes:
            handle.write(f"file '{path}'\nduration {per_scene:.3f}\n")
        handle.write(f"file '{scenes[-1]}'\n")

    filters = [
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase",
        f"crop={WIDTH}:{HEIGHT}",
        motion_filter(per_scene),
    ]
    if CAPTIONS:
        filters.append(caption_filter(write_srt(script, total)))
    else:
        log("Captions disabled for this render", None, None)
    filters.append("format=yuv420p")

    command = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", "scenes.txt",
        "-i", voice,
        "-vf", ",".join(filters),
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-b:v", VIDEO_BITRATE,
        "-c:a", "aac", "-b:a", "192k",
        "-r", "30", "-t", f"{total:.3f}", "-movflags", "+faststart",
        "out.mp4",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()[-6:]
        raise RuntimeError("FFmpeg failed: " + " | ".join(tail))


def main() -> int:
    try:
        log(
            f"Video Agent starting · {DURATION}s · {ASPECT_RATIO} · {QUALITY} · "
            f"captions {'on' if CAPTIONS else 'off'}",
            "Initializing",
            5,
        )
        script, scene_prompts = write_script()
        scenes = generate_scenes(scene_prompts)
        voice = generate_voice(script)
        compose(scenes, voice, script)
        log("Render complete", "Finished", 95)
        return 0
    except Exception as error:  # noqa: BLE001
        log(f"Render failed: {error}")
        patch({"status": "failed", "error": str(error)[:500]})
        return 1


if __name__ == "__main__":
    sys.exit(main())
