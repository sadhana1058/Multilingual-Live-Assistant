"""
Real-time Pinch voice translation — correct version.
Pinch outputs 48kHz audio. No upsampling needed.

Run: python3 trial2.py
Stop: Ctrl+C
"""

import asyncio
import queue
import sys

import numpy as np
import sounddevice as sd

from pinch import PinchClient
from pinch.session import SessionParams

# ── Config ────────────────────────────────────────────────────────────────────
SOURCE_LANG  = "ta-IN"
TARGET_LANG  = "en-US"
INPUT_RATE   = 16000          # mic input to Pinch — always 16kHz
CHUNK_SIZE   = 320            # 20ms at 16kHz

OUTPUT_RATE  = 48000          # Pinch OUTPUTS at 48kHz (confirmed from 0.5x symptom)
# No upsampling — Pinch audio is already at OUTPUT_RATE
# ──────────────────────────────────────────────────────────────────────────────

audio_input_queue:  queue.Queue   = queue.Queue()
audio_output_queue: asyncio.Queue = asyncio.Queue()
track_ready = False


def pcm16_to_float32(pcm16_bytes: bytes) -> np.ndarray:
    """Convert raw PCM16 bytes → float32. No resampling — Pinch is already 48kHz."""
    int16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
    return int16.astype(np.float32) / 32768.0


def mic_callback(indata, frames, time, status):
    if status:
        print(f"[mic] {status}", file=sys.stderr)
    mono  = indata[:, 0]
    pcm16 = (mono * 32767).clip(-32768, 32767).astype(np.int16)
    audio_input_queue.put(pcm16.tobytes())


async def send_audio_loop(stream):
    loop = asyncio.get_event_loop()
    while True:
        pcm_bytes = await loop.run_in_executor(None, audio_input_queue.get)
        await stream.send_pcm16(pcm_bytes, sample_rate=INPUT_RATE, channels=1)


async def receive_events_loop(stream):
    global track_ready
    async for event in stream.events():
        if event.type == "transcript":
            label = "[FINAL]  " if event.is_final else "[partial]"
            print(f"{label} {event.text}")

        elif event.type == "audio":
            int16     = np.frombuffer(event.pcm16_bytes, dtype=np.int16)
            has_audio = np.abs(int16).max() > 10

            if has_audio and not track_ready:
                # Print actual sample rate from event if available
                sr = getattr(event, 'sample_rate', 'unknown')
                print(f"[audio] ✅ Remote track active — event.sample_rate={sr}")
                track_ready = True

            await audio_output_queue.put(event.pcm16_bytes)


async def playback_loop():
    global track_ready
    loop    = asyncio.get_event_loop()
    waiting = 0

    with sd.OutputStream(
        samplerate=OUTPUT_RATE,   # 48000 — matches Pinch output
        channels=1,
        dtype="float32",
        blocksize=0,
    ) as out_stream:
        print(f"[audio] OutputStream open at {OUTPUT_RATE}Hz (no upsampling)")

        while True:
            pcm_bytes = await audio_output_queue.get()

            if not track_ready:
                waiting += 1
                if waiting % 100 == 0:
                    print(f"[audio] Waiting for track... ({waiting})")
                continue

            # Direct conversion — no resampling
            float32 = await loop.run_in_executor(None, pcm16_to_float32, pcm_bytes)
            await loop.run_in_executor(None, out_stream.write, float32)


async def main():
    client = PinchClient()

    print(f"\n🎙  Pinch real-time translation")
    print(f"   Speak : {SOURCE_LANG}  →  Hear : {TARGET_LANG}")
    print(f"   Ctrl+C to stop\n")

    session = client.create_session(
        SessionParams(
            source_language=SOURCE_LANG,
            target_language=TARGET_LANG,
        )
    )

    stream = await client.connect_stream(session, audio_output_enabled=True)
    print("⏳ Waiting for remote audio track... (speak now, ✅ appears when ready)\n")

    mic_stream = sd.InputStream(
        samplerate=INPUT_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SIZE,
        callback=mic_callback,
    )

    try:
        with mic_stream:
            await asyncio.gather(
                send_audio_loop(stream),
                receive_events_loop(stream),
                playback_loop(),
            )
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await stream.aclose()
        print("✅ Done.")


if __name__ == "__main__":
    asyncio.run(main())