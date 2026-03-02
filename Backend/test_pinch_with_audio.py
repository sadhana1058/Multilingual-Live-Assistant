"""
Real-time Pinch voice translation — fixed audio track subscription.
Speak Spanish → hear English translated back.

Run: python3 test_pinch_realtime.py
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
SOURCE_LANG = "es-ES"
TARGET_LANG = "en-US"
PINCH_RATE  = 16000
CHUNK_MS    = 20
CHUNK_SIZE  = int(PINCH_RATE * CHUNK_MS / 1000)   # 320 samples

_out_info   = sd.query_devices(kind="output")
OUTPUT_RATE = int(_out_info["default_samplerate"])  # 48000 on your Mac
UPSAMPLE    = OUTPUT_RATE // PINCH_RATE             # 3

print(f"[audio] {PINCH_RATE}Hz → {OUTPUT_RATE}Hz (×{UPSAMPLE})")

# Minimum amplitude to consider a chunk "real" audio vs zeros
# Pinch sends 0.000 chunks while track is not yet subscribed
SILENCE_THRESHOLD = 0.001
# ──────────────────────────────────────────────────────────────────────────────

audio_input_queue:  queue.Queue   = queue.Queue()
audio_output_queue: asyncio.Queue = asyncio.Queue()


def upsample_pcm16(pcm16_bytes: bytes) -> np.ndarray | None:
    """
    Convert Pinch 16kHz PCM16 → float32 at OUTPUT_RATE.
    Returns None if the chunk is silent (track not yet ready).
    """
    int16   = np.frombuffer(pcm16_bytes, dtype=np.int16)
    float32 = int16.astype(np.float32) / 32768.0

    # Skip zero chunks — these arrive before the remote track is subscribed
    if float32.max() < SILENCE_THRESHOLD:
        return None

    return np.repeat(float32, UPSAMPLE).astype(np.float32)


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
        await stream.send_pcm16(pcm_bytes, sample_rate=PINCH_RATE, channels=1)


async def receive_events_loop(stream):
    track_active = False
    async for event in stream.events():
        if event.type == "transcript":
            label = "[FINAL]  " if event.is_final else "[partial]"
            print(f"{label} {event.text}")

        elif event.type == "audio":
            # Check if this is the first real (non-zero) audio chunk
            int16 = np.frombuffer(event.pcm16_bytes, dtype=np.int16)
            max_val = np.abs(int16).max()

            if not track_active and max_val > 10:
                track_active = True
                print("[audio] ✅ Remote audio track active — translation audio flowing")

            await audio_output_queue.put(event.pcm16_bytes)


async def playback_loop():
    loop = asyncio.get_event_loop()
    skipped = 0

    with sd.OutputStream(
        samplerate=OUTPUT_RATE,
        channels=1,
        dtype="float32",
        blocksize=0,
    ) as out_stream:
        print(f"[audio] OutputStream open at {OUTPUT_RATE}Hz")
        while True:
            pcm_bytes = await audio_output_queue.get()
            float32   = await loop.run_in_executor(
                None, upsample_pcm16, pcm_bytes
            )

            if float32 is None:
                skipped += 1
                if skipped % 50 == 0:
                    print(f"[audio] Waiting for remote track... ({skipped} silent chunks skipped)")
                continue

            # Reset skip counter once real audio starts
            if skipped > 0:
                skipped = 0

            await loop.run_in_executor(None, out_stream.write, float32)


async def main():
    client = PinchClient()

    print(f"\n🎙  Pinch real-time translation")
    print(f"   Speak : {SOURCE_LANG}")
    print(f"   Hear  : {TARGET_LANG}")
    print(f"   Ctrl+C to stop\n")

    session = client.create_session(
        SessionParams(
            source_language=SOURCE_LANG,
            target_language=TARGET_LANG,
        )
    )

    stream = await client.connect_stream(session, audio_output_enabled=True)

    # ── KEY FIX: wait for LiveKit to negotiate the remote audio track ──────────
    # The remote translated audio track takes 1-3s to be subscribed after connect.
    # Without this wait, the first few seconds of translated audio are silent.
    print("⏳ Waiting for remote audio track to be ready...")
    await asyncio.sleep(2.0)
    print("✅ Connected. Start speaking...\n")
    # ──────────────────────────────────────────────────────────────────────────

    mic_stream = sd.InputStream(
        samplerate=PINCH_RATE,
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
        print("\n\nStopping...")
    finally:
        await stream.aclose()
        print("✅ Done.")


if __name__ == "__main__":
    asyncio.run(main())