# backend/app/main.py

import asyncio
import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.pinch_bridge import PinchBridge

load_dotenv()
PINCH_API_KEY = os.getenv("PINCH_API_KEY")

app = FastAPI(title="Medical Intake API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws/translate")
async def translate_ws(websocket: WebSocket):
    """
    WebSocket protocol:

    CLIENT → SERVER:
      1. First message (text): {"type": "config", "source": "ta-IN", "target": "en-US"}
      2. All subsequent messages (binary): raw PCM16 bytes at 16kHz mono

    SERVER → CLIENT:
      - binary frames : raw PCM16 bytes at 48kHz (Pinch output, play directly)
      - {"type": "partial", "text": "..."}  : partial transcript
      - {"type": "final",   "text": "..."}  : final transcript
      - {"type": "status",  "text": "..."}  : status messages
    """
    await websocket.accept()
    print("[WS] Client connected")

    bridge: PinchBridge | None = None

    # ── Step 1: wait for config message ──────────────────────────────────────
    try:
        raw    = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        config = json.loads(raw)
        assert config.get("type") == "config"
        source = config.get("source", "ta-IN")
        target = config.get("target", "en-US")
        print(f"[WS] Config received: {source} → {target}")
    except Exception as e:
        print(f"[WS] Config error: {e}")
        await websocket.close(code=4000)
        return

    # ── Step 2: start Pinch bridge ────────────────────────────────────────────
    bridge = PinchBridge(source, target, api_key=PINCH_API_KEY)
    await bridge.start()

    await websocket.send_text(json.dumps({
        "type": "status",
        "text": f"Connected: {source} → {target}. Speak now — audio starts after track ready (~2s)."
    }))

    # ── Step 3: four concurrent tasks ─────────────────────────────────────────

    async def receive_audio():
        """Browser mic audio → Pinch."""
        try:
            while True:
                data = await websocket.receive_bytes()
                await bridge.send_audio(data)
        except WebSocketDisconnect:
            pass

    async def forward_audio():
        """Pinch translated audio → browser (raw PCM16 at 48kHz)."""
        try:
            while True:
                pcm = await bridge.audio_output_queue.get()
                await websocket.send_bytes(pcm)
        except Exception as e:
            print(f"[WS] Audio forward error: {e}")

    async def forward_partials():
        """Pinch partial transcripts → browser."""
        try:
            while True:
                text = await bridge.partial_transcript_queue.get()
                await websocket.send_text(json.dumps({
                    "type": "partial",
                    "text": text,
                }))
        except Exception as e:
            print(f"[WS] Partial forward error: {e}")

    async def forward_finals():
        """Pinch final transcripts → browser."""
        try:
            while True:
                text = await bridge.final_transcript_queue.get()
                await websocket.send_text(json.dumps({
                    "type": "final",
                    "text": text,
                }))
        except Exception as e:
            print(f"[WS] Final forward error: {e}")

    # ── Run all four concurrently ──────────────────────────────────────────────
    try:
        await asyncio.gather(
            receive_audio(),
            forward_audio(),
            forward_partials(),
            forward_finals(),
        )
    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Session error: {e}")
    finally:
        if bridge:
            await bridge.close()
        print("[WS] Session cleaned up")