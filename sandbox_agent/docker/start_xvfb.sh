#!/bin/bash
# Start Xvfb virtual display + VNC + noVNC if XVFB_ENABLED=true.
# This is sourced by the container entrypoint before Chainlit starts.

if [ "$XVFB_ENABLED" = "true" ]; then
    echo "[xvfb] Starting virtual display on :99 ..."
    Xvfb :99 -screen 0 1280x900x24 -ac &
    XVFB_PID=$!
    sleep 1

    if kill -0 $XVFB_PID 2>/dev/null; then
        echo "[xvfb] Display :99 ready (PID $XVFB_PID)"
        export DISPLAY=:99

        # VNC server on display :99 → port 5999
        echo "[vnc] Starting x11vnc on display :99 ..."
        x11vnc -display :99 -nopw -forever -shared -rfbport 5999 &
        sleep 1

        # noVNC web client → websockify proxy → VNC
        echo "[novnc] Starting noVNC on port 6080 ..."
        cd /opt/novnc/noVNC-1.4.0
        websockify --web=/opt/novnc/noVNC-1.4.0 \
            --daemon \
            6080 localhost:5999 &
        sleep 1
        echo "[novnc] Open http://localhost:6080 to watch the browser"
    else
        echo "[xvfb] WARNING: Xvfb failed to start"
    fi
fi
