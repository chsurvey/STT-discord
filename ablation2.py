import asyncio
import json
import os
import time

import edge_tts
import whisper
from faster_whisper import WhisperModel

from tts import playlist

ROOT_DIR = os.path.dirname(__file__)
AUDIO_PATH = os.path.join(ROOT_DIR, "ablation2_en.wav")
OUTPUT_PATH = os.path.join(ROOT_DIR, "ablation2_results.json")

MODEL_SIZE = "medium"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
LANGUAGE = "en"
BEAM_SIZE = 5

from pathlib import Path
async def generate_english_audio():
    text, voice = playlist[0]
    audio_path = Path(AUDIO_PATH)

    if audio_path.exists() and audio_path.is_file() and audio_path.stat().st_size > 0:
        print(f"Audio already exists, reusing: {audio_path}")
        return text, voice

    print("Generating English TTS audio...")
    communicate = edge_tts.Communicate(text, voice)

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    await communicate.save(str(audio_path))

    print(f"Saved audio: {audio_path}")
    return text, voice

def run_whisper(audio_path):
    print("Loading Whisper model...")
    start = time.perf_counter()
    model = whisper.load_model(MODEL_SIZE, device=DEVICE)
    load_s = time.perf_counter() - start

    print("Running Whisper transcription...")
    start = time.perf_counter()
    result = model.transcribe(
        audio_path, language=LANGUAGE, task="transcribe", beam_size=BEAM_SIZE
    )
    transcribe_s = time.perf_counter() - start
    text = result.get("text", "").strip()
    return {
        "name": "whisper",
        "model": MODEL_SIZE,
        "device": DEVICE,
        "load_s": load_s,
        "transcribe_s": transcribe_s,
        "text": text,
    }

def run_faster_whisper(audio_path):
    print("Loading Faster-Whisper model...")
    start = time.perf_counter()
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    load_s = time.perf_counter() - start

    print("Running Faster-Whisper transcription...")
    start = time.perf_counter()
    segments, info = model.transcribe(
        audio_path, beam_size=BEAM_SIZE, language=LANGUAGE
    )
    text = " ".join(segment.text for segment in segments).strip()
    transcribe_s = time.perf_counter() - start
    return {
        "name": "faster_whisper",
        "model": MODEL_SIZE,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "language": info.language,
        "load_s": load_s,
        "transcribe_s": transcribe_s,
        "text": text,
    }


async def main():
    text, voice = await generate_english_audio()
    print("Starting STT ablation...")

    results = {
        "config": {
            "audio_path": AUDIO_PATH,
            "text_prompt": text,
            "voice": voice,
            "model_size": MODEL_SIZE,
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
            "language": LANGUAGE,
            "beam_size": BEAM_SIZE,
        },
        "results": [],
    }

    results["results"].append(run_whisper(AUDIO_PATH))
    results["results"].append(run_faster_whisper(AUDIO_PATH))

    print(f"Writing results to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
