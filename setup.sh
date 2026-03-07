#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "=== OpenClaw Voice Chat Setup ==="

# 1. Python venv
if [ ! -d "$VENV_DIR" ]; then
  echo "[1/4] Creating Python virtual environment..."
  python3 -m venv "$VENV_DIR"
else
  echo "[1/4] Virtual environment exists, skipping."
fi

source "$VENV_DIR/bin/activate"

# 2. Install dependencies
echo "[2/4] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r "$SCRIPT_DIR/requirements.txt" -q

# 3. Check CUDA availability for faster-whisper
echo "[3/4] Checking CUDA..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  CUDA available: {torch.cuda.get_device_name(0)}')
else:
    print('  WARNING: CUDA not available. STT will be slow on CPU.')
" 2>/dev/null || echo "  (torch not installed, CUDA check skipped - faster-whisper will use its own CUDA bindings)"

# 4. Check ffmpeg (required by faster-whisper)
echo "[4/4] Checking ffmpeg..."
if command -v ffmpeg &>/dev/null; then
  echo "  ffmpeg found: $(ffmpeg -version 2>&1 | head -1)"
else
  echo "  WARNING: ffmpeg not found. Install with: sudo apt install ffmpeg"
fi

# Create .env if not exists
if [ ! -f "$SCRIPT_DIR/.env" ]; then
  cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
  echo ""
  echo ">>> .env file created from .env.example"
  echo ">>> Edit $SCRIPT_DIR/.env with your API keys before running."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your ElevenLabs API key and voice ID"
echo "  2. Run:  source .venv/bin/activate && python server.py"
echo "  3. Open: http://localhost:8000"
echo ""
echo "For HTTPS via Tailscale:"
echo "  tailscale serve https / http://localhost:8000"
echo "  Then open: https://$(hostname).your-tailnet.ts.net"
