"""Reference render script for tkdasofficial/video-agent.

Copy to the repo root as `render.py` (alongside `requirements.txt`).

Pipeline
--------
1. Cloudflare Workers AI (Llama instruct models) writes the narration script and
   the per-scene image prompts (the "brain"), and synthesises the voiceover with
   Workers AI TTS (Deepgram Aura, MeloTTS fallback).
2. Pixazo AI (https://api.pixazo.ai) renders the scene images ONLY, with Workers
   AI image models as fallback. Pixazo exposes no TTS API.
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
# Pixazo is images only. Narration/TTS runs on Cloudflare Workers AI.

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
CF_TTS_MODELS = [
    model.strip()
    for model in env(
        "CF_TTS_MODEL",
        "@cf/deepgram/aura-1,@cf/myshell-ai/melotts",
    ).split(",")
    if model.strip()
]
# Deepgram Aura speaker names, chosen by the requested voice gender.
CF_AURA_SPEAKER = env(
    "CF_AURA_SPEAKER",
    "orion" if VOICE_GENDER.startswith("m") else "luna",
)

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


def word_budget(duration: int) -> int:
    """Hard word cap for the narration at a 130 WPM speaking standard.

    Anchored to the product spec: 15s -> 30 words, 30s -> 65 words,
    60s -> 120 words (absolute maximum), interpolated in between.
    """
    duration = max(1, min(60, int(duration)))
    if duration <= 15:
        return max(6, round(duration * 30 / 15))
    if duration <= 30:
        return round(30 + (duration - 15) * (65 - 30) / 15)
    return min(120, round(65 + (duration - 30) * (120 - 65) / 30))


WORD_BUDGET = word_budget(DURATION)

NATURE_STYLE_PREFIX = (
    "breathtaking humanless nature and cosmic scenery, planets, nebulae, starfields, "
    "mountains, oceans, forests, macro micro-details, natural textures, volumetric light, "
    "ultra detailed, no people"
)
HUMANLESS_NEGATIVE = "human, person, face, character, crowd, watermark, text"


def image_prompt(scene_prompt: str) -> str:
    """Humanless nature/cosmic prompt builder shared by every image provider."""
    return f"{NATURE_STYLE_PREFIX}, {scene_prompt}, {IMAGE_STYLE}"[:1900]


def image_negative_prompt() -> str:
    extra = NEGATIVE_PROMPT.strip().strip(",")
    return f"{HUMANLESS_NEGATIVE}, {extra}" if extra else HUMANLESS_NEGATIVE


def trim_to_budget(script: str) -> str:
    """Never let the narration exceed the duration's word cap."""
    words = script.split()
    if len(words) <= WORD_BUDGET:
        return script.strip()
    trimmed = " ".join(words[:WORD_BUDGET]).rstrip(" ,;:-")
    if not trimmed.endswith((".", "!", "?")):
        trimmed += "."
    log(f"Script trimmed to {WORD_BUDGET} words for the {DURATION}s limit")
    return trimmed


def ensure_script_length(script: str, minimum_words: int) -> str:
    """Keep narration dense enough to occupy the requested runtime."""
    if len(script.split()) >= minimum_words:
        return trim_to_budget(script)

    expanded = clean_script(
        cf_chat(
            "You expand narration for humanless nature and cosmic short films. "
            "Reply with narration sentences only.",
            f"Topic: {PROMPT}\nCurrent narration: {script}\n"
            f"Rewrite this as {minimum_words} to {WORD_BUDGET} flowing words. "
            "Keep the meaning, use no humans, labels, lists, or stage directions.",
            max_tokens=500,
        )
    )
    if len(expanded.split()) >= minimum_words:
        return trim_to_budget(expanded)

    # A provider can occasionally return an empty/very short answer. Preserve the
    # selected runtime with neutral scenery narration rather than a silent tail.
    pieces = [script or PROMPT]
    bridges = [
        "Across this vast scene, light and motion reveal details shaped quietly through time.",
        "Colors drift through the landscape while distant forms create depth, rhythm, and wonder.",
        "Every changing texture invites a closer look at the beauty held within this world.",
        "The view continues beyond the horizon, calm, immense, and alive with subtle movement.",
    ]
    index = 0
    while len(" ".join(pieces).split()) < minimum_words:
        pieces.append(bridges[index % len(bridges)])
        index += 1
    return trim_to_budget(" ".join(pieces))


def cf_chat(system: str, user: str, *, max_tokens: int = 700) -> str:
    """Runs a chat prompt through the first Workers AI model that answers."""
    body = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
    }
    for model in CF_LLM_MODELS:
        try:
            raw = cloudflare_run(model, body).json()
        except Exception as error:  # noqa: BLE001 - retired/unavailable model, try the next
            log(f"Script model {model} unavailable ({error}); trying the next one")
            continue

        # Workers AI response shapes: {"result":{"response":"..."}} or {"response":"..."}
        result = raw.get("result") if isinstance(raw, dict) else None
        if not isinstance(result, dict):
            result = raw if isinstance(raw, dict) else {}
        candidate = result.get("response")
        if candidate is None:
            candidate = result.get("result")
        if isinstance(candidate, dict):
            candidate = candidate.get("response") or candidate.get("text")
        text = "" if candidate is None else str(candidate).strip()
        if text and text.lower() != "none":
            return text
    return ""


def clean_script(text: str) -> str:
    """Strips list markers, labels and quotes an instruct model likes to add."""
    if not text:
        return ""
    # A JSON reply is still accepted, but plain prose is the expected shape now.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict) and parsed.get("script"):
                text = str(parsed["script"])
        except json.JSONDecodeError:
            pass
    lines = []
    for line in text.splitlines():
        line = line.strip().strip("`").strip()
        low = line.lower()
        if not line or low.startswith(("script:", "narration:", "here", "note:", "scene")):
            continue
        lines.append(line.lstrip("-*0123456789. ").strip('"'))
    return " ".join(lines).strip()


def parse_scenes(text: str) -> list[str]:
    """Reads one image prompt per line (or from a JSON array) out of the reply."""
    scenes: list[str] = []
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, list):
                scenes = [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            scenes = []
    if not scenes:
        for line in text.splitlines():
            line = line.strip().lstrip("-*0123456789. ").strip('"').strip()
            if len(line) > 12 and not line.lower().startswith(("here", "note", "scene prompts")):
                scenes.append(line)
    return scenes


SCENE_VARIATIONS = [
    "sweeping establishing wide shot",
    "close orbital detail shot",
    "dramatic low-angle vista",
    "glowing nebula backdrop with depth",
    "macro texture detail",
    "silhouetted horizon at golden light",
    "top-down aerial perspective",
    "distant scale shot with layered depth",
]


def write_script() -> tuple[str, list[str]]:
    """Returns (narration script, one image prompt per scene)."""
    log(f"Writing the {DURATION}s script with Cloudflare Workers AI", "Writing script", 12)
    lower = max(6, round(WORD_BUDGET * 0.8))

    script = clean_script(
        cf_chat(
            "You are a narrator for humanless nature and cosmic short films. "
            "Reply with the narration sentences only — no titles, labels, lists or notes.",
            f"Topic: {PROMPT}\n"
            f"Write flowing narration of between {lower} and {WORD_BUDGET} words so it reads "
            f"aloud in about {DURATION} seconds at 130 words per minute. "
            "No humans or characters, no stage directions, no hashtags.",
            max_tokens=500,
        )
    )
    if not script:
        log("Script model gave no usable text; narrating the prompt directly", None, None)
        script = PROMPT
    script = ensure_script_length(script, lower)
    log(f"Script ready ({len(script.split())} words / {WORD_BUDGET} max)")

    scenes = parse_scenes(
        cf_chat(
            "You write image-generation prompts. Reply with one prompt per line, nothing else.",
            f"Story: {script}\n"
            f"Write exactly {SCENE_COUNT} distinct image prompts in the '{IMAGE_STYLE}' style "
            "covering different moments of this story. Each prompt is one line, 15-30 words, "
            "showing nature, cosmic space, planets, oceans or micro-detail scenery only — "
            "never humans, faces, characters or text.\n"
            f"Avoid: {NEGATIVE_PROMPT or 'nothing in particular'}.",
            max_tokens=700,
        )
    )
    # Every fallback scene gets its own camera angle so the clip never repeats one frame.
    while len(scenes) < SCENE_COUNT:
        variation = SCENE_VARIATIONS[len(scenes) % len(SCENE_VARIATIONS)]
        scenes.append(f"{PROMPT}, {variation}, {IMAGE_STYLE}")
    return script, scenes[:SCENE_COUNT]


# --- Stage B: Pixazo AI images (Workers AI fallback) + Workers AI TTS -------
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
    raise RuntimeError(f"Provider response contained no media: {json.dumps(body)[:300]}")


def cloudflare_image(scene_prompt: str, path: str) -> None:
    """Fallback image generator so a Pixazo outage does not fail the whole render."""
    # FLUX.1 Schnell's current REST contract needs only a prompt. Avoid sending
    # dimensions or sampling fields shared by other image models: Workers AI
    # rejects unsupported fields with HTTP 400.
    prompt = image_prompt(scene_prompt)
    errors: list[str] = []
    for model in CF_IMAGE_MODELS:
        if "flux" in model:
            seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
            body = {"prompt": prompt, "seed": seed}
        else:
            body = {
                "prompt": prompt,
                "negative_prompt": image_negative_prompt(),
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
                    "prompt": image_prompt(scene_prompt),
                    "negative_prompt": image_negative_prompt(),
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
    """Narration comes from Cloudflare Workers AI TTS (Pixazo has no TTS API)."""
    log(
        f"Synthesising narration with Cloudflare Workers AI TTS ({VOICE_GENDER})",
        "Generating voice",
        52,
    )
    path = "voice.mp3"
    text = script.strip()[:1800]
    errors: list[str] = []
    for model in CF_TTS_MODELS:
        if "aura" in model:
            body = {"text": text, "speaker": CF_AURA_SPEAKER, "encoding": "mp3"}
        else:
            # MeloTTS takes plain text plus a language code and returns base64 mp3.
            body = {"prompt": text, "lang": env("CF_TTS_LANG", "en")}
        try:
            response = cloudflare_run(model, body, timeout=240)
            save_binary_or_b64(response, path, ("b64_json", "audio", "audio_base64"))
            return path
        except Exception as error:  # noqa: BLE001 - try the next TTS model
            errors.append(f"{model}: {error}")
    raise RuntimeError("no Workers AI TTS model succeeded — " + " | ".join(errors))


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


def ass_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    centis = int(round(seconds * 100))
    hours, centis = divmod(centis, 360_000)
    minutes, centis = divmod(centis, 6_000)
    secs, centis = divmod(centis, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


CAPTION_PRESETS = {
    # PrimaryColour / OutlineColour are ASS &HAABBGGRR values.
    "Yellow Pop-Up": {"primary": "&H0000E5FF", "outline": "&H00000000", "bold": -1, "border": 4},
    "Neon Glow": {"primary": "&H00FFFFFF", "outline": "&H00CC00FF", "bold": -1, "border": 5},
    "Minimalist White": {"primary": "&H00FFFFFF", "outline": "&H00000000", "bold": 0, "border": 3},
    "Monospace Subtitles": {"primary": "&H00FFFFFF", "outline": "&H00000000", "bold": 0, "border": 3},
}


def write_ass(script: str, total: float, path: str = "captions.ass") -> str:
    """Bottom-centre captions, 3-4 words per cue, at a real 44px on a 1080x1920 canvas.

    An ASS file is generated directly (instead of an SRT + force_style) because
    force_style sizes are relative to libass' default 384x288 script resolution,
    which is what previously blew the text up to full-screen height.
    """
    words = script.split()
    if not words:
        words = [PROMPT or "…"]
    per_cue = 4  # 3-4 words per frame keeps shorts captions readable
    cues = [words[i : i + per_cue] for i in range(0, len(words), per_cue)]
    weight = sum(len(" ".join(c)) for c in cues) or 1

    preset = CAPTION_PRESETS.get(CAPTION_STYLE, CAPTION_PRESETS["Minimalist White"])
    font = "DejaVu Sans Mono" if CAPTION_STYLE == "Monospace Subtitles" else "DejaVu Sans"
    font_size = max(22, round(HEIGHT * 44 / 1920))
    # Alignment 2 = bottom centre; the baseline sits at roughly 88% of the canvas.
    margin_v = max(24, round(HEIGHT * 0.10))
    margin_h = max(40, round(WIDTH * 0.08))

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        f"PlayResX: {WIDTH}\n"
        f"PlayResY: {HEIGHT}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Caption,{font},{font_size},{preset['primary']},{preset['primary']},"
        f"{preset['outline']},&H80000000,{preset['bold']},0,0,0,100,100,0,0,1,"
        f"{preset['border']},1,2,{margin_h},{margin_h},{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    clock = 0.0
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(header)
        for cue in cues:
            text = " ".join(cue).replace("\n", " ")
            span = max(0.5, total * (len(text) / weight))
            start, end = clock, min(total, clock + span)
            clock = end
            handle.write(
                f"Dialogue: 0,{ass_timestamp(start)},{ass_timestamp(end)},Caption,,0,0,0,,{text}\n"
            )
    log(f"Burning {len(cues)} caption cues ({CAPTION_STYLE})", "Rendering captions", 68)
    return path


def caption_filter(ass_path: str) -> str:
    """Burns the generated ASS captions (styling lives in the file itself)."""
    return f"subtitles={ass_path}"


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
    return f"zoompan=z='min(zoom+0.003,1.30)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={WIDTH}x{HEIGHT}:fps=30"


def atempo_filter(rate: float) -> str:
    """Build a legal FFmpeg atempo chain for any positive speed ratio."""
    rate = max(rate, 0.01)
    factors: list[float] = []
    while rate < 0.5:
        factors.append(0.5)
        rate /= 0.5
    while rate > 2.0:
        factors.append(2.0)
        rate /= 2.0
    factors.append(rate)
    return ",".join(f"atempo={factor:.6f}" for factor in factors)


def compose(scenes: list[str], voice: str, script: str) -> None:
    log("Composing final video with FFmpeg", "Rendering video", 78)
    narration = audio_duration(voice)

    # The requested duration is authoritative. Retiming the narration avoids a
    # long silent gap when a TTS provider speaks faster than expected.
    total = float(DURATION)
    narration_target = max(0.5, total - 0.25)
    tempo = narration / narration_target
    if abs(narration - narration_target) > 0.25:
        log(f"Fitting {narration:.1f}s narration to the selected {DURATION}s runtime")
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
        filters.append(caption_filter(write_ass(script, narration_target)))
    else:
        log("Captions disabled for this render", None, None)
    filters.append("format=yuv420p")

    command = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", "scenes.txt",
        "-i", voice,
        "-filter_complex",
        f"[0:v]{','.join(filters)}[v];[1:a]{atempo_filter(tempo)},apad,"
        f"atrim=0:{total:.3f},asetpts=N/SR/TB[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-b:v", VIDEO_BITRATE,
        "-c:a", "aac", "-b:a", "192k",
        "-r", "30", "-t", f"{total:.3f}", "-movflags", "+faststart",
        "out.mp4",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()[-6:]
        raise RuntimeError("FFmpeg failed: " + " | ".join(detail))


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
