# OpenClaw Voice Chat

OpenClaw 에이전트와 자연스러운 음성 대화를 할 수 있는 웹 기반 인터페이스입니다.

브라우저에서 마이크로 말하면, 자동으로 음성 인식(STT) → OpenClaw 에이전트 호출 → 음성 합성(TTS)을 거쳐 답변이 음성으로 재생됩니다. 답변이 끝나면 다시 듣기 모드로 전환되어 연속 대화가 가능합니다.

## 특징

- **자연스러운 대화 흐름**: VAD(음성 활동 감지)로 말이 끝나면 자동 전송, 답변 재생 후 자동 듣기모드
- **로컬 STT**: faster-whisper + GPU로 빠른 음성 인식 (~0.5초)
- **ElevenLabs TTS**: 원하는 목소리로 답변 재생, 문장별 스트리밍으로 체감 지연 최소화
- **세션 유지**: OpenClaw `--session-id`로 대화 맥락 유지
- **OpenClaw 코드 무수정**: CLI(`openclaw agent`)를 subprocess로 호출
- **HTTPS 지원**: Tailscale serve/Funnel로 폰에서도 안전하게 접속

## 요구사항

- **Ubuntu** (또는 Linux) + **NVIDIA GPU** (CUDA, faster-whisper용)
- **Python 3.10+**
- **Node.js 22+** + **OpenClaw** 게이트웨이 실행 중
- **ffmpeg** (`sudo apt install ffmpeg`)
- **ElevenLabs** API 키 + Voice ID
- (선택) **Tailscale** — 외부/폰에서 HTTPS 접속

## 설치

```bash
git clone https://github.com/ho-daddy/openclaw_voice-chat.git
cd openclaw_voice-chat

chmod +x setup.sh
./setup.sh
```

setup.sh가 자동으로:
1. Python 가상환경(`.venv`) 생성
2. 의존성 설치 (FastAPI, faster-whisper, ElevenLabs 등)
3. CUDA 및 ffmpeg 확인
4. `.env.example` → `.env` 복사

## 설정

`.env` 파일을 편집합니다:

```bash
nano .env
```

```env
# 필수
ELEVENLABS_API_KEY=your_api_key_here
ELEVENLABS_VOICE_ID=your_voice_id_here

# 선택 (기본값 있음)
ELEVENLABS_MODEL_ID=eleven_multilingual_v2
ELEVENLABS_OUTPUT_FORMAT=mp3_22050_32
WHISPER_MODEL=large-v3
WHISPER_LANG=ko
OPENCLAW_TIMEOUT=120
OPENCLAW_THINKING=low
HOST=0.0.0.0
PORT=8000
```

**ElevenLabs Voice ID 확인**: [ElevenLabs](https://elevenlabs.io) → Voice Lab → 원하는 목소리 선택 → Voice ID 복사

## 실행

### 수동 실행 (테스트)

```bash
cd ~/openclaw_voice-chat
source .venv/bin/activate
python server.py
```

브라우저에서 `http://localhost:8000` 접속 → **Start** 버튼 클릭

### 상시 실행 (systemd)

```bash
# 서비스 파일의 WorkingDirectory 경로를 확인 후 설치
sudo cp voice-chat.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now voice-chat

# 상태 확인
sudo systemctl status voice-chat

# 로그 보기
journalctl -u voice-chat -f
```

> `voice-chat.service`의 경로(`/home/hodaddy/voice-chat`)가 실제 설치 경로와 다르면 수정하세요.

### 폰에서 접속 (Tailscale)

```bash
# Tailscale이 설치되어 있다면
tailscale serve https / http://localhost:8000
```

폰에서 `https://<machine-name>.<tailnet>.ts.net` 으로 접속합니다.
Tailscale Funnel이 활성화되어 있으면 Tailscale 앱 없는 기기에서도 접근 가능합니다.

## 사용법

1. 브라우저에서 접속 → **Start** 버튼 클릭 (마이크 권한 허용)
2. 초록 orb → **듣기 모드** — 말을 시작하면 빨간색으로 변함
3. 빨간 orb → **녹음 중** — 1.2초 묵음 감지 시 자동 전송
4. 노란 orb → **처리 중** — STT → OpenClaw 에이전트 호출
5. 파란 orb → **말하는 중** — ElevenLabs TTS 재생
6. 재생 끝나면 자동으로 다시 듣기 모드

**Stop** 버튼으로 세션 종료.

## 아키텍처

```
[Browser]
  ├── Web Audio API (마이크 캡처)
  ├── VAD (음량 기반 묵음 감지)
  └── WebSocket
        ↕
[FastAPI Server (Ubuntu)]
  ├── faster-whisper (로컬 GPU STT)
  ├── openclaw agent --session-id <id> --message <text> --json
  └── ElevenLabs TTS API (문장별 스트리밍)
```

## 문제 해결

| 증상 | 해결 |
|------|------|
| 마이크가 작동하지 않음 | HTTPS 필요 — localhost 또는 Tailscale serve 사용 |
| STT가 느림 | CUDA 드라이버 확인, `nvidia-smi`로 GPU 상태 확인 |
| openclaw agent 오류 | OpenClaw 게이트웨이 실행 중인지 확인 (`openclaw channels status`) |
| ElevenLabs 오류 | `.env`의 API 키와 Voice ID 확인 |
| WebSocket 연결 끊김 | 자동 재연결됨 (2초 후), 서버 로그 확인 |

## 라이센스

MIT
