#!/bin/bash

# Start virtual X display and run simulation with GUI
set -e

# Start Xvfb (virtual display)
export DISPLAY=:99
Xvfb $DISPLAY -screen 0 1920x1080x24 > /tmp/xvfb.log 2>&1 &
XVFB_PID=$!
echo "Started Xvfb on $DISPLAY (PID: $XVFB_PID)"

# Wait for display to start
sleep 2

# Run simulation
cd /workspaces/lsy_drone_racing
echo "Starting simulation..."
pixi run python scripts/sim.py "$@"

# Cleanup
kill $XVFB_PID 2>/dev/null || true
