import asyncio
import time
from datetime import datetime
from dotenv import load_dotenv
from livekit.agents import (
    AgentServer,
    AgentSession,
    Agent,
    JobContext,
    JobProcess,
    inference,
    room_io,
    TurnHandlingOptions,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
)
from livekit.plugins import silero, ai_coustics

load_dotenv(override=True)

# After VAD silence, wait this long for the final transcript before flushing anyway
FINAL_WAIT = 1.5


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


class SilentAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="")


server = AgentServer()


def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load(
            activation_threshold=0.65,
        )
    except Exception as e:
        print(f"Error loading VAD: {e}")


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",  # VAD silence is the EOU signal
        ),
    )

    _speech_active              = False
    _speech_index               = 0
    _last_interim               = ""
    _utterance_parts: list[str] = []
    _speech_start_time: float   = 0.0
    _first_final_time: float    = 0.0
    _vad_ended: bool            = False   # True once VAD fires silence
    _flushed: bool              = False
    _flush_task: asyncio.Task | None = None

    def _print_block_end():
        print(f"└─── SPEECH #{_speech_index} END   [{ts()}] {'─' * 30}\n")

    def _do_flush():
        nonlocal _utterance_parts, _last_interim, _flushed
        if _flushed:
            return
        if not _utterance_parts and not _last_interim:
            return  # silent/noise block — suppress entirely
        _flushed = True

        # Prefer confirmed finals; fall back to last interim only if no final arrived
        full = " ".join(_utterance_parts).strip() or _last_interim.strip()
        if full:
            e2e_ms = (_first_final_time - _speech_start_time) * 1000
            print(f"│  [{ts()}] EOU        VAD")
            print(f"│  [{ts()}] UTTERANCE  {full}")
            print(f"│  [{ts()}] E2E        {e2e_ms:.0f}ms (speech start → first final)")
            _print_block_end()
        _utterance_parts = []
        _last_interim    = ""

    async def _wait_for_final_then_flush():
        """
        Called after VAD fires. Waits FINAL_WAIT seconds for the in-flight
        Deepgram final to land before flushing. If a final arrives first,
        on_transcript cancels this task and flushes immediately.
        """
        await asyncio.sleep(FINAL_WAIT)
        _do_flush()

    @session.on("user_state_changed")
    def on_user_state(event: UserStateChangedEvent):
        nonlocal _speech_active, _speech_index, _last_interim, _utterance_parts
        nonlocal _speech_start_time, _flushed, _vad_ended, _flush_task
        nonlocal _first_final_time

        if event.new_state == "speaking":
            # Cancel any pending flush from previous block
            if _flush_task and not _flush_task.done():
                _flush_task.cancel()
            _do_flush()  # flush previous block if it hadn't flushed yet

            _speech_active     = True
            _speech_index     += 1
            _speech_start_time = time.monotonic()
            _first_final_time  = 0.0
            _flushed           = False
            _vad_ended         = False
            _last_interim      = ""
            _utterance_parts   = []
            print(f"\n┌─── SPEECH #{_speech_index} START [{ts()}] {'─' * 30}")

        elif event.new_state != "speaking" and _speech_active:
            # VAD detected silence = EOU, but final transcript may not have arrived yet
            _speech_active = False
            _vad_ended     = True
            # Start a timer — on_transcript will cancel it if the final arrives first
            _flush_task = asyncio.ensure_future(_wait_for_final_then_flush())

    @session.on("user_input_transcribed")
    def on_transcript(event: UserInputTranscribedEvent):
        nonlocal _last_interim, _utterance_parts, _first_final_time
        nonlocal _flush_task

        lang = getattr(event, "language", "?")
        if not event.is_final:
            _last_interim = event.transcript
            print(f"│  [{ts()}] INTERIM    [{lang}]  {event.transcript}")
            return

        # Accumulate finals
        if not _utterance_parts:
            _first_final_time = time.monotonic()
        _utterance_parts.append(event.transcript.strip())

        # If VAD already fired, cancel the safety timer and flush now that we
        # have the confirmed final transcript
        if _vad_ended:
            if _flush_task and not _flush_task.done():
                _flush_task.cancel()
            _do_flush()

    await session.start(
        agent=SilentAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                )
            ),
            audio_output=False,
            text_output=False,
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    from livekit.agents import cli
    cli.run_app(server)
