"""
Real-time Pinch voice translation test.
Speak Spanish → hear English translated back through your speakers.

Run: python3 test_pinch_realtime.py
Stop: Ctrl+C
"""

import asyncio
import sounddevice as sd
import numpy as np
import queue
import sys

from pinch import PinchClient
from pinch.session import SessionParams

# ── Config ────────────────────────────────────────────────────────────────────
SOURCE_LANG = "ta-IN"   # What you'll speak
TARGET_LANG = "en-US"   # What you'll hear
SAMPLE_RATE = 16000     # Required by Pinch
CHUNK_MS    = 20        # 20ms chunks — Pinch's preferred frame size
CHUNK_SIZE  = int(SAMPLE_RATE * CHUNK_MS / 1000)  # = 320 samples
# ──────────────────────────────────────────────────────────────────────────────

audio_input_queue: queue.Queue = queue.Queue()
audio_output_queue: asyncio.Queue = asyncio.Queue()


def mic_callback(indata, frames, time, status):
    """Called by sounddevice on every audio chunk from the mic."""
    if status:
        print(f"[mic] {status}", file=sys.stderr)
    # indata is float32, shape (CHUNK_SIZE, 1) — convert to PCM16 bytes
    mono = indata[:, 0]
    pcm16 = (mono * 32767).clip(-32768, 32767).astype(np.int16)
    audio_input_queue.put(pcm16.tobytes())


async def send_audio_loop(stream):
    """Pull PCM16 chunks from the queue and push to Pinch."""
    loop = asyncio.get_event_loop()
    while True:
        # Run blocking queue.get() in thread pool so we don't block the event loop
        pcm_bytes = await loop.run_in_executor(None, audio_input_queue.get)
        await stream.send_pcm16(pcm_bytes, sample_rate=SAMPLE_RATE, channels=1)



async def receive_events_loop(stream):
    """Pull transcript + audio events from Pinch and handle them."""
    async for event in stream.events():
        if event.type == "transcript":
            label = "[FINAL]" if event.is_final else "[partial]"
            print(f"{label} {event.text}")

        elif event.type == "audio":
            # Queue translated audio for playback
            if len(event.pcm16_bytes)>0:
                print("🔊 Received audio chunk size:", len(event.pcm16_bytes))
                break
            await audio_output_queue.put(event.pcm16_bytes)


async def playback_loop():
    """Play translated audio chunks as they arrive."""
    loop = asyncio.get_event_loop()
    while True:
        pcm_bytes = await audio_output_queue.get()
        # Convert PCM16 bytes back to float32 for sounddevice playback
        pcm16 = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767
        # Play non-blocking (run in executor to avoid blocking event loop)
        await loop.run_in_executor(
            None,
            lambda: sd.play(pcm16, samplerate=SAMPLE_RATE, blocking=True)
        )


async def main():
    client = PinchClient()  # Reads PINCH_API_KEY from env automatically

    print(f"\n🎙  Starting Pinch real-time translation")
    print(f"   Speak: {SOURCE_LANG}  →  Hear: {TARGET_LANG}")
    print(f"   Press Ctrl+C to stop\n")
    # print(client)
    import os
    print("KEY REPR:", repr(os.getenv("PINCH_API_KEY")))
    print("KEY LENGTH:", len(os.getenv("PINCH_API_KEY") or ""))
    print("CWD:", os.getcwd())
    from dotenv import load_dotenv
    import os

    load_dotenv()

    print("Loaded:", repr(os.getenv("PINCH_API_KEY")))
    # Create Pinch session
    session = client.create_session(
        SessionParams(
            source_language=SOURCE_LANG,
            target_language=TARGET_LANG,
        )
    )

    # Connect streaming transport
    stream = await client.connect_stream(session, audio_output_enabled=True)

    print("✅ Pinch stream connected. Start speaking...\n")

    # Open mic stream with sounddevice
    mic_stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SIZE,
        callback=mic_callback,
    )

    try:
        with mic_stream:
            # Run all three loops concurrently
            await asyncio.gather(
                send_audio_loop(stream),
                receive_events_loop(stream),
                playback_loop(),
            )
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        await stream.aclose()
        print("✅ Stream closed cleanly.")


if __name__ == "__main__":
    asyncio.run(main())