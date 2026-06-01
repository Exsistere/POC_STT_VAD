# save as test_vad.py and run: uv run python test_vad.py

import sounddevice as sd

import numpy as np

import time
 
SAMPLE_RATE = 16000

DURATION = 10  # seconds — speak normally during this
 
print("Recording for 10 seconds, speak normally...")

audio = sd.rec(

    int(DURATION * SAMPLE_RATE),

    samplerate=SAMPLE_RATE,

    channels=1,

    dtype='float32'

)

sd.wait()
 
# Print amplitude stats per second

print("\nAmplitude per second (RMS):")

for i in range(DURATION):

    chunk = audio[i*SAMPLE_RATE:(i+1)*SAMPLE_RATE]

    rms = float(np.sqrt(np.mean(chunk**2)))

    bar = '█' * int(rms * 200)

    print(f"  {i+1}s: {rms:.4f}  {bar}")
 
print(f"\nOverall max: {np.max(np.abs(audio)):.4f}")

print(f"Overall RMS: {float(np.sqrt(np.mean(audio**2))):.4f}")
 