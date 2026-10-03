from faster_whisper import WhisperModel
from pathlib import Path
import json
import time


VIDEO = Path("recordings/2026-09-05 15-30-58.mp4")
OUTPUT = Path("transcripts/2026-09-05 15-30-58.json")


print("=" * 60)
print("CLIP FACTORY - WHISPER LARGE-V3")
print("=" * 60)

print("\nCargando Whisper Large-v3...")

load_start = time.time()

model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16",
)

print(
    f"Modelo cargado en "
    f"{time.time() - load_start:.2f} segundos."
)


print("\nTranscribiendo video...")
print(f"Archivo: {VIDEO}")

start_time = time.time()

segments, info = model.transcribe(
    str(VIDEO),

    beam_size=5,

    vad_filter=True,

    word_timestamps=True,

    condition_on_previous_text=False,
)


result = {
    "language": info.language,
    "language_probability": info.language_probability,
    "duration": info.duration,
    "segments": [],
    "full_text": "",
}


all_text = []


for segment in segments:

    words = []

    if segment.words:

        for word in segment.words:

            words.append(
                {
                    "start": round(word.start, 3),
                    "end": round(word.end, 3),
                    "word": word.word.strip(),
                    "probability": round(
                        word.probability,
                        4
                    ),
                }
            )

    segment_data = {
        "start": round(segment.start, 3),
        "end": round(segment.end, 3),
        "text": segment.text.strip(),
        "words": words,
    }

    result["segments"].append(segment_data)

    all_text.append(segment.text.strip())


result["full_text"] = " ".join(all_text)


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        result,
        file,
        ensure_ascii=False,
        indent=2
    )


elapsed = time.time() - start_time


print("\n" + "=" * 60)
print("TRANSCRIPCIÓN TERMINADA")
print("=" * 60)

print(f"Idioma: {result['language']}")

print(
    f"Probabilidad idioma: "
    f"{result['language_probability']:.4f}"
)

print(
    f"Duración: "
    f"{result['duration']:.2f} segundos"
)

print(
    f"Segmentos: "
    f"{len(result['segments'])}"
)

total_words = sum(
    len(segment["words"])
    for segment in result["segments"]
)

print(
    f"Palabras: "
    f"{total_words}"
)

print(
    f"Tiempo: "
    f"{elapsed:.2f} segundos"
)

print(
    f"Resultado: "
    f"{OUTPUT}"
)