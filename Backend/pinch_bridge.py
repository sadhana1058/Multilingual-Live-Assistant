# backend/app/pinch_bridge.py

import asyncio
import numpy as np
from pinch import PinchClient
from pinch.session import SessionParams


class PinchBridge:
    """
    Manages one Pinch streaming session.

    What we confirmed in Step 1:
    - Pinch outputs 48kHz PCM16 audio (not 16kHz)
    - No upsampling needed — convert PCM16 → float32 only
    - Remote audio track takes 1-3s to subscribe after connect
    - Zero chunks arrive until track is ready — skip them
    - Once track_ready=True it never resets — silence between words is valid

    Queues:
    - audio_output_queue  : bytes  — raw 48kHz PCM16 ready for browser
    - final_transcript_queue  : str — final transcripts for LangGraph (Step 3)
    - partial_transcript_queue : str — partials for UI display only
    """

    INPUT_RATE  = 16000   # mic input to Pinch
    OUTPUT_RATE = 48000   # Pinch output — confirmed from Step 1

    def __init__(self, source_language: str, target_language: str, api_key: str):
        self.source_language = source_language
        self.target_language = target_language
        self.api_key         = api_key

        self.client  = PinchClient(api_key=self.api_key)
        self.stream  = None
        self.session = None

        self._event_task: asyncio.Task | None = None
        self._closed     = False
        self._track_ready = False

        # Queues
        self.audio_output_queue:       asyncio.Queue[bytes] = asyncio.Queue()
        self.final_transcript_queue:   asyncio.Queue[str]   = asyncio.Queue()
        self.partial_transcript_queue: asyncio.Queue[str]   = asyncio.Queue()

    async def start(self):
        """Create Pinch session and connect stream."""
        self.session = self.client.create_session(
            SessionParams(
                source_language=self.source_language,
                target_language=self.target_language,
            )
        )
        self.stream = await self.client.connect_stream(
            self.session,
            audio_output_enabled=True,
        )
        self._event_task = asyncio.create_task(self._listen_events())
        print(f"[PinchBridge] Connected: {self.source_language} → {self.target_language}")

    async def _listen_events(self):
        """Background task: route Pinch events to queues."""
        try:
            async for event in self.stream.events():

                if event.type == "transcript":
                    if event.is_final:
                        await self.final_transcript_queue.put(event.text)
                        print(f"[Pinch FINAL] {event.text}")
                    else:
                        await self.partial_transcript_queue.put(event.text)

                elif event.type == "audio":
                    int16     = np.frombuffer(event.pcm16_bytes, dtype=np.int16)
                    has_audio = np.abs(int16).max() > 10

                    if has_audio and not self._track_ready:
                        self._track_ready = True
                        sr = getattr(event, "sample_rate", "unknown")
                        print(f"[PinchBridge] ✅ Track ready — sample_rate={sr}")

                    # Only queue audio once track is ready
                    # Queueing zeros before track is ready causes buffer desync
                    if self._track_ready:
                        await self.audio_output_queue.put(event.pcm16_bytes)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[PinchBridge] Event listener error: {e}")

    async def send_audio(self, pcm_bytes: bytes):
        """Forward 16kHz PCM16 chunk from browser to Pinch."""
        if not self._closed and self.stream:
            await self.stream.send_pcm16(
                pcm_bytes,
                sample_rate=self.INPUT_RATE,
                channels=1,
            )

    async def close(self):
        """Cleanly shut down the Pinch stream."""
        self._closed = True
        if self._event_task:
            self._event_task.cancel()
            try:
                await self._event_task
            except asyncio.CancelledError:
                pass
        if self.stream:
            await self.stream.aclose()
        print("[PinchBridge] Closed.")