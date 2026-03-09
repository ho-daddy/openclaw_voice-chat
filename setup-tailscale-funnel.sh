#!/bin/bash
# Tailscale Funnel setup script for voice-chat

# Wait for voice-chat service to be ready
sleep 5

# Reset any existing configuration
/usr/bin/tailscale serve reset
/usr/bin/tailscale funnel reset

# Run funnel in foreground (includes serve, keeps process alive for systemd)
exec /usr/bin/tailscale funnel 8000
