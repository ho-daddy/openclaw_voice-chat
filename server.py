"""OpenClaw Voice Chat - FastAPI WebSocket server."""

import asyncio
import json
import os
import re
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# --- Lazy singletons ---

_whisper_model = None
_eleven_client = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(
            os.getenv("WHISPER_MODEL", "large-v3"),
            device="cuda",
            compute_type="float16",
        )
    return _whisper_model


def get_eleven_client():
    global _eleven_client
    if _eleven_client is None:
        from elevenlabs import ElevenLabs

        _eleven_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    return _eleven_client


# --- Helpers ---


def clean_for_tts(text: str) -> str:
    """Strip markdown formatting and emoji for cleaner TTS."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [link](url) -> link
    text = re.sub(r"[*_~`#]", "", text)  # markdown chars
    text = re.sub(
        r"[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff"
        r"\U0001f1e0-\U0001f1ff\U00002702-\U000027b0\U0001f900-\U0001f9ff"
        r"\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff\U00002600-\U000026ff"
        r"\U0000fe0f]+",
        "",
        text,
    )
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split into sentence-sized chunks for incremental TTS."""
    # Split on sentence-ending punctuation followed by space/newline
    parts = re.split(r"(?<=[.!?~])\s+|\n+", text)
    result = []
    buf = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        buf = f"{buf} {p}".strip() if buf else p
        # Flush when buffer has a sentence ending or is long enough
        if re.search(r"[.!?~]$", buf) or len(buf) > 120:
            result.append(buf)
            buf = ""
    if buf:
        result.append(buf)
    return result


# --- Core pipeline (sync, run in executor) ---


def _transcribe_sync(audio_path: str) -> str:
    model = get_whisper_model()
    segments, _ = model.transcribe(
        audio_path,
        language=os.getenv("WHISPER_LANG", "ko"),
        vad_filter=True,
    )
    return " ".join(seg.text for seg in segments).strip()


def _tts_sync(text: str) -> bytes:
    client = get_eleven_client()
    audio_iter = client.text_to_speech.convert(
        voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        text=text,
        model_id=os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"),
        output_format=os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_22050_32"),
    )
    return b"".join(audio_iter)


# --- Async wrappers ---


async def transcribe(audio_path: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _transcribe_sync, audio_path)


async def call_agent(message: str, session_id: str) -> str:
    cmd = [
        "openclaw", "agent",
        "--session-id", session_id,
        "--message", message,
        "--json",
        "--timeout", os.getenv("OPENCLAW_TIMEOUT", "120"),
    ]

    thinking = os.getenv("OPENCLAW_THINKING")
    if thinking:
        cmd.extend(["--thinking", thinking])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        err = stderr.decode().strip()
        raise RuntimeError(f"Agent error (rc={proc.returncode}): {err}")

    data = json.loads(stdout.decode())

    if data.get("status") != "ok":
        raise RuntimeError(f"Agent status: {data.get('status')} - {data.get('summary')}")

    return data["result"]["payloads"][0]["text"]


async def generate_tts(text: str) -> bytes:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _tts_sync, text)


# --- Routes ---


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = f"voice-{uuid.uuid4().hex[:8]}"

    await ws.send_json({
        "type": "state",
        "state": "ready",
        "sessionId": session_id,
    })

    try:
        while True:
            data = await ws.receive_bytes()

            # Save audio to temp file
            tmp = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
            try:
                tmp.write(data)
                tmp.close()

                # 1) STT
                await ws.send_json({"type": "state", "state": "transcribing"})
                text = await transcribe(tmp.name)

                if not text.strip():
                    await ws.send_json({"type": "state", "state": "listening"})
                    continue

                await ws.send_json({"type": "transcript", "text": text})

                # 2) Agent
                await ws.send_json({"type": "state", "state": "thinking"})
                response = await call_agent(text, session_id)
                await ws.send_json({"type": "response", "text": response})

                # 3) TTS - sentence by sentence
                await ws.send_json({"type": "state", "state": "speaking"})
                clean = clean_for_tts(response)
                sentences = split_sentences(clean)

                for sentence in sentences:
                    if not sentence:
                        continue
                    audio = await generate_tts(sentence)
                    # Send audio length header then audio bytes
                    await ws.send_json({"type": "audio_meta", "size": len(audio)})
                    await ws.send_bytes(audio)

                # Signal TTS complete
                await ws.send_json({"type": "tts_done"})
                await ws.send_json({"type": "state", "state": "listening"})

            finally:
                os.unlink(tmp.name)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
            await ws.send_json({"type": "state", "state": "listening"})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
