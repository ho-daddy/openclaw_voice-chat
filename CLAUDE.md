# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the server

```bash
source .venv/bin/activate
python server.py
```

Open `http://localhost:8000` in a browser. For microphone access over HTTPS via Tailscale:

```bash
tailscale serve https / http://localhost:8000
```

## Setup (first time)

```bash
./setup.sh
# Then edit .env with your ElevenLabs API key and voice ID
```

Requires: NVIDIA GPU with CUDA, ffmpeg (`sudo apt install ffmpeg`), Node.js 22+ with the OpenClaw gateway running.

## systemd service

```bash
sudo cp voice-chat.service /etc/systemd/system/
sudo systemctl enable --now voice-chat
journalctl -u voice-chat -f
```

## Architecture

This is a minimal two-file project:

- **`server.py`** — FastAPI WebSocket server, all backend logic
- **`static/index.html`** — single-page frontend, all UI and audio logic

### Request pipeline (per voice turn)

```
Browser mic (Web Audio API)
  → VAD (RMS-based silence detection in JS)
    → WebM audio blob sent over WebSocket (binary)
      → faster-whisper STT (local GPU, CUDA float16)
        → `openclaw agent --session-id <id> --message <prefixed_text> --json` (subprocess)
          → ElevenLabs TTS per sentence (sentence-split streaming)
            → MP3 audio chunks sent back over WebSocket (binary)
              → Browser audio queue (sequential playback)
```

### Key design points

- **Lazy singletons**: `get_whisper_model()` and `get_eleven_client()` initialize on first use; the heavy Whisper model loads on the first audio message, not at startup.
- **Blocking ops in executor**: `_transcribe_sync` and `_tts_sync` run in `asyncio.run_in_executor` to avoid blocking the event loop.
- **openclaw agent is a subprocess**: The server never imports openclaw. It calls `openclaw agent --json` and parses `stdout`. Session continuity is maintained via `--session-id voice-<8hex>`.
- **Voice mode prefix**: Every user message is prepended with a Korean prompt instructing the agent to respond in spoken, markdown-free style. Configurable via `VOICE_MODE_PREFIX` env var.
- **TTS text cleaning**: `clean_for_tts()` strips markdown, emoji, URLs, and code blocks before sending to ElevenLabs. `split_sentences()` chunks the response for lower perceived latency.

## Environment variables

All configured in `.env` (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `ELEVENLABS_API_KEY` | (required) | ElevenLabs API key |
| `ELEVENLABS_VOICE_ID` | (required) | ElevenLabs voice ID |
| `ELEVENLABS_MODEL_ID` | `eleven_multilingual_v2` | TTS model |
| `ELEVENLABS_OUTPUT_FORMAT` | `mp3_22050_32` | Audio format |
| `WHISPER_MODEL` | `large-v3` | faster-whisper model size |
| `WHISPER_LANG` | `ko` | STT language |
| `OPENCLAW_TIMEOUT` | `120` | Agent call timeout (seconds) |
| `OPENCLAW_THINKING` | (unset) | Passes `--thinking` to openclaw if set |
| `VOICE_MODE_PREFIX` | Korean system prompt | Prefix injected before every user message |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
