# Video Agent — end-to-end pipeline architecture

Decoupled AI video generation across three systems:

| System | Owns |
| --- | --- |
| Lovable app (this repo) | UI, auth, credits, job row creation, dispatch, realtime display |
| Supabase (`uqyuwxztevkokzqldibh`) | Auth, `profiles`, `subscriptions`, `videos`, private `videos` storage bucket, Realtime |
| `tkdasofficial/video-agent` (GitHub) | All heavy AI work: Pixazo images, Cloudflare AI voice, FFmpeg composition, upload |

```text
UI form ──> startVideoRender (server fn) ──> subscriptions (credit) ──> videos row (pending)
                                   │
                                   └──> POST /repos/tkdasofficial/video-agent/dispatches
                                             event_type: video_agent_render
                                                         │
GitHub Actions runner (ubuntu-latest) <───────────────────┘
  status=processing → Pixazo scenes → Cloudflare voice → FFmpeg render
  → upload MP4 to Supabase Storage `videos/<user_id>/<video_id>.mp4`
  → videos.video_url + status=completed   (or status=failed + error)
                                                         │
Supabase Realtime (postgres_changes on public.videos) ────┘
  → terminal logger + video player + download in the UI
```

## Stage 1 — Trigger & queuing (UI → Supabase → GitHub)

Code: `src/routes/video-agent.tsx` (form + submit), `src/lib/video-agent.functions.ts`
(`startVideoRender`), `supabase/config/config.ts` (public config).

1. The form collects prompt, negative prompt, voice gender/persona/speed/pitch, image style,
   motion template, captions, aspect ratio, quality, bitrate.
2. `startVideoRender` runs server-side under `requireSupabaseAuth`, reads the caller's
   `subscriptions` row and spends **one** credit — `video_credits` first, otherwise one unit of
   `monthly_quota` via `credits_used`. No credits → the request is rejected before any work.
3. A `videos` row is inserted with `status: 'pending'`, `step: 'queued'`, `progress: 0`, and
   every user parameter; its `id` is the `video_id` for the whole pipeline.
4. A `repository_dispatch` (`event_type: video_agent_render`) is POSTed to
   `https://api.github.com/repos/tkdasofficial/video-agent/dispatches` with
   `client_payload: { video_id, prompt, negative_prompt, voice_gender, image_style, aspect_ratio, user_id }`.
   Dispatch failure marks the row `failed` and refunds the reserved credit.

The GitHub token lives in the server-side secret `GITHUB_PAT` (never in
`supabase/config/config.ts`, which ships to browsers — GitHub auto-revokes published tokens).
Repository owner/name/event constants do live in that config file.

## Stage 2 — Backend execution (GitHub Actions)

Reference files to copy into `tkdasofficial/video-agent`:

| From this repo | To engine repo |
| --- | --- |
| [`docs/video-agent/video-agent.yml`](./video-agent/video-agent.yml) | `.github/workflows/video-agent.yml` |
| [`docs/video-agent/requirements.txt`](./video-agent/requirements.txt) | `requirements.txt` (repo root) |
| [`docs/video-agent/render.py`](./video-agent/render.py) | `render.py` (repo root) |

Missing `requirements.txt` or `render.py` in the engine repo is what causes
`Could not open requirements file` / exit code 1 in the runner — the workflow now falls back to
installing `requests`, but `render.py` must exist for the render step.

Repository secrets required there: `PIXAZO_API_KEY`, `CLOUDFLARE_ACCOUNT_ID`,
`CLOUDFLARE_API_TOKEN`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`.

1. `on: repository_dispatch: types: [video_agent_render]` starts an `ubuntu-latest` runner.
2. The runner sets `status: 'processing'` and appends progress into `videos.logs` /
   `videos.step` / `videos.progress` as it goes (each write streams to the UI).
3. Pixazo → scene images from prompt + `image_style` (+ `negative_prompt`).
4. Cloudflare Workers AI → narration audio for `voice_gender`.
5. FFmpeg → composes images + audio + captions at the requested aspect ratio into an MP4.

## Stage 3 — Storage & delivery

1. The runner uploads the MP4 to the Supabase Storage bucket `videos` at
   `<user_id>/<video_id>.mp4` using the service role key.
2. The bucket is **private** (this workspace blocks public buckets), so the runner writes the
   object path — `<user_id>/<video_id>.mp4` — into `videos.video_url`; an absolute
   `https://` URL is also accepted and used verbatim.
3. `status → 'completed'`.
4. On any failure: `status → 'failed'` with the reason in `error` and detail appended to `logs`.

Storage access rules: signed-in users read only objects under their own `auth.uid()` folder; the
service role can write anywhere in the bucket.

## Stage 4 — UI sync & consumption

1. The page subscribes to `postgres_changes` UPDATEs on `public.videos` filtered by the active
   `video_id`, with a 6-second poll as a backgrounded-tab safety net.
2. The terminal logger prints each new `logs` entry / `step`; the status chip shows
   Queued (`pending`) → Rendering AI visuals & audio (`processing`) → Finished (`completed`) /
   Error (`failed`).
3. On `completed`, `getVideoPlaybackUrl` turns the stored object path into a 6-hour signed URL
   (or passes an absolute URL through), and the player plus download button use it.

## System boundaries

- Supabase public URL / anon key / ref id: `supabase/config/config.ts`.
- `GITHUB_PAT`: Lovable server secret only.
- Pixazo + Cloudflare keys: GitHub repository secrets in `tkdasofficial/video-agent` only — they
  never exist in this app or its database.
- Zero rendering in the browser or on the app server; FFmpeg and all AI calls run on the runner.
