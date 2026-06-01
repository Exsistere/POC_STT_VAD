import asyncio
import os
import sys
import queue
import numpy as np
import sounddevice as sd

from livekit import rtc
from dotenv import load_dotenv

load_dotenv(override=True)

SAMPLE_RATE = 24000 
NUM_CHANNELS = 1
BLOCK_MS = 20  # 20ms frames → 480 samples per frame


async def main():
    url = os.environ["LIVEKIT_URL"]
    api_key = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]

    # Generate a token programmatically using the Python SDK
    from livekit.api import AccessToken, VideoGrants
    token = (
        AccessToken(api_key, api_secret)
        .with_identity("user")
        .with_name("user")
        .with_grants(VideoGrants(room_join=True, room="my_room", can_publish=True))
        .to_jwt()
    )

    print(f"Connecting to {url} as 'user'...")
    room = rtc.Room()

    @room.on("connected")
    def on_connected():
        print(f"Connected to room '{room.name}'")

    @room.on("disconnected")
    def on_disconnected(reason):
        print(f"Disconnected: {reason}")

    await room.connect(url, token)

    # Create audio source and publish mic track
    source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    await room.local_participant.publish_track(track, opts)
    print("Mic track published — speak now! (Ctrl-C to stop)")

    # Capture mic via sounddevice and push frames into an async queue
    frame_queue: queue.Queue = queue.Queue()
    blocksize = int(SAMPLE_RATE * BLOCK_MS / 1000)  # 480 samples

    def sd_callback(indata: np.ndarray, frames: int, time_info, status):
        if status:
            print(f"sounddevice status: {status}", file=sys.stderr)
        # indata is float32 in [-1, 1] — convert to int16
        pcm = (indata * 32767).astype(np.int16).flatten()
        frame_queue.put_nowait(pcm)

    loop = asyncio.get_event_loop()

    async def push_frames():
        while True:
            # Run blocking queue.get in executor so we don't block the event loop
            pcm = await loop.run_in_executor(None, frame_queue.get)
            frame = rtc.AudioFrame(
                data=pcm.tobytes(),
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=len(pcm),
            )
            await source.capture_frame(frame)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=NUM_CHANNELS,
        dtype="float32",
        blocksize=blocksize,
        callback=sd_callback,
    ):
        try:
            await push_frames()
        except asyncio.CancelledError:
            pass

    await room.disconnect()
    print("Done.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")