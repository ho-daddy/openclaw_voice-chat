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


def get_voice_prefix() -> str:
    """Build the voice-mode instruction prefix for the agent."""
    default = (
        "[음성 대화 모드] "
        "지금 사용자와 음성으로 대화하고 있습니다. 다음 규칙을 반드시 지켜주세요: "
        "1) 자연스러운 구어체(말투)로 답변하세요. "
        "2) 이모지, 마크다운 서식(**굵게**, *기울임*, `코드` 등), 특수기호를 절대 사용하지 마세요. "
        "3) 목록이나 번호 매기기 대신 자연스러운 문장으로 이어서 말하세요. "
        "4) 답변은 간결하게 해주세요. 길어도 3~4문장 이내로요. "
        "5) URL이나 코드 블록은 포함하지 마세요. "
        "사용자 메시지: "
    )
    return os.getenv("VOICE_MODE_PREFIX", default)


def clean_for_tts(text: str) -> str:
    """Strip any remaining markdown, emoji, and non-speech artifacts for TTS."""
    # Links: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Bare URLs
    text = re.sub(r"https?://\S+", "", text)
    # Markdown formatting chars
    text = re.sub(r"[*_~`#>]", "", text)
    # Bullet points and numbered lists (e.g. "- item", "1. item")
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)
    # Code blocks (```...```)
    text = re.sub(r"```[\s\S]*?```", "", text)
    # All emoji (comprehensive Unicode ranges)
    text = re.sub(
        r"[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff"
        r"\U0001f1e0-\U0001f1ff\U00002702-\U000027b0\U0001f900-\U0001f9ff"
        r"\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff\U00002600-\U000026ff"
        r"\U0000200d\U0000fe0f\U000020e3\U00003030\U000021a9-\U000021aa"
        r"\U0000231a-\U0000231b\U00002328\U000023cf\U000023e9-\U000023f3"
        r"\U000023f8-\U000023fa\U000024c2\U000025aa-\U000025ab\U000025b6"
        r"\U000025c0\U000025fb-\U000025fe\U00002934-\U00002935"
        r"\U00002b05-\U00002b07\U00002b1b-\U00002b1c\U00002b50\U00002b55"
        r"\U00003297\U00003299]+",
        "",
        text,
    )
    # Collapse multiple spaces/newlines
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"  +", " ", text)
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


# --- Response classification ---

# Patterns for text-only content
_RE_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
_RE_URL = re.compile(r"https?://\S+")
_RE_COMMAND = re.compile(r"^\s*[\$#]\s+.+", re.MULTILINE)
_RE_FILE_PATH = re.compile(r"(?:^|\s)(?:[~/][\w./-]+(?:\.\w+)?)", re.MULTILINE)
_RE_JSON_BLOCK = re.compile(r"\{[\s\S]{50,}?\}")


def classify_response(text: str) -> dict:
    """Classify agent response into speech/text/mixed mode.

    Returns dict with keys: mode, speech, text.
    - mode="speech": everything is speakable
    - mode="text": everything is text-only (code, URLs, etc.)
    - mode="mixed": some parts speech, some text-only
    """
    # Find all text-only segments
    text_only_spans = []
    for pattern in [_RE_CODE_BLOCK, _RE_URL, _RE_COMMAND, _RE_JSON_BLOCK]:
        for m in pattern.finditer(text):
            text_only_spans.append((m.start(), m.end(), m.group()))

    if not text_only_spans:
        return {"mode": "speech", "speech": text, "text": ""}

    # Merge overlapping spans
    text_only_spans.sort(key=lambda x: x[0])
    merged = [text_only_spans[0]]
    for start, end, content in text_only_spans[1:]:
        if start <= merged[-1][1]:
            prev_start, prev_end, prev_content = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end), text[prev_start:max(prev_end, end)])
        else:
            merged.append((start, end, content))

    # Extract speech parts (everything not in text-only spans)
    speech_parts = []
    text_parts = []
    pos = 0
    for start, end, content in merged:
        before = text[pos:start].strip()
        if before:
            speech_parts.append(before)
        text_parts.append(content)
        pos = end
    trailing = text[pos:].strip()
    if trailing:
        speech_parts.append(trailing)

    speech = " ".join(speech_parts).strip()
    text_content = "\n".join(text_parts).strip()

    if not speech:
        return {"mode": "text", "speech": "", "text": text}
    if not text_content:
        return {"mode": "speech", "speech": text, "text": ""}
    return {"mode": "mixed", "speech": speech, "text": text_content}


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


async def call_agent(message: str, session_id: str, use_voice_prefix: bool = True) -> str:
    prefixed = f"{get_voice_prefix()}{message}" if use_voice_prefix else message
    cmd = [
        "openclaw", "agent",
        "--session-id", session_id,
        "--message", prefixed,
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


async def _handle_agent_response(ws: WebSocket, response: str, input_mode: str):
    """Classify the agent response and send speech/text/mixed back to the client."""
    classified = classify_response(response)
    mode = classified["mode"]

    # Send the classified assistant message
    await ws.send_json({
        "type": "assistant_message",
        "mode": mode,
        "speech": classified["speech"],
        "text": classified["text"],
        "full": response,
    })

    # TTS for speech/mixed modes
    if mode in ("speech", "mixed"):
        await ws.send_json({"type": "state", "state": "speaking"})
        clean = clean_for_tts(classified["speech"])
        sentences = split_sentences(clean)

        for sentence in sentences:
            if not sentence:
                continue
            audio = await generate_tts(sentence)
            await ws.send_json({"type": "audio_meta", "size": len(audio)})
            await ws.send_bytes(audio)

        await ws.send_json({"type": "tts_done"})

    await ws.send_json({"type": "state", "state": "listening"})


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
            msg = await ws.receive()

            if msg.get("bytes"):
                # --- Voice input (binary audio) ---
                data = msg["bytes"]
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

                    await ws.send_json({"type": "transcript", "text": text, "mode": "speech"})

                    # 2) Agent (with voice prefix for spoken input)
                    await ws.send_json({"type": "state", "state": "thinking"})
                    response = await call_agent(text, session_id, use_voice_prefix=True)

                    # 3) Classify and respond
                    await _handle_agent_response(ws, response, "speech")

                finally:
                    os.unlink(tmp.name)

            elif msg.get("text"):
                # --- Text input (JSON message) ---
                payload = json.loads(msg["text"])

                if payload.get("type") == "user_message":
                    content = payload.get("content", "").strip()
                    if not content:
                        continue

                    # Echo user text back for display
                    await ws.send_json({"type": "transcript", "text": content, "mode": "text"})

                    # Agent (no voice prefix for text input)
                    await ws.send_json({"type": "state", "state": "thinking"})
                    response = await call_agent(content, session_id, use_voice_prefix=False)

                    # Classify and respond
                    await _handle_agent_response(ws, response, "text")

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
