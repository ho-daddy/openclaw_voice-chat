#!/bin/bash
# Tailscale Funnel setup script for voice-chat

# Wait for voice-chat service to be ready
sleep 5

# Reset any existing configuration
/usr/bin/tailscale serve reset

# Setup serve (internal proxy)
/usr/bin/tailscale serve --bg 8000

# Setup funnel (external access)
/usr/bin/tailscale funnel --bg 8000

echo "Tailscale Funnel setup complete"
