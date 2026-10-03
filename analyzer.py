from pathlib import Path
import json
import ollama


TRANSCRIPT = Path("transcripts/2026-09-05 15-30-58.json")
OUTPUT = Path("candidates/2026-09-05 15-30-58.json")

MODEL = "qwen3:8b"

MIN_SCORE = 65
MIN_CLIP_DURATION = 15.0
MAX_CLIP_DURATION = 60.0
TARGET_CLIP_DURATION = 25.0


def load_transcript():

    with open(
        TRANSCRIPT,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def build_transcript(data):

    lines = []

    for segment in data["segments"]:

        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()

        if not text:
            continue

        lines.append(
            f"[{start:.2f}s - {end:.2f}s] {text}"
        )

    return "\n".join(lines)


def analyze_with_qwen(
    transcript,
    duration
):

    prompt = f"""
Eres un analizador profesional de contenido para
streams de Call of Duty Warzone.

Tu trabajo es detectar MOMENTOS INTERESANTES,
no generar clips artificialmente.

Duración total del video:

{duration:.2f} segundos


============================================================
REGLA PRINCIPAL
============================================================

Solo puedes seleccionar un momento si existe evidencia
clara en la transcripción.

NO inventes lo que está ocurriendo en pantalla.

La transcripción contiene solamente lo que dice el streamer.


============================================================
MOMENTOS QUE BUSCAMOS
============================================================

Busca:

- Eliminaciones mencionadas explícitamente.
- Headshots.
- Enemigos.
- Peleas.
- Victorias.
- Derrotas.
- Reacciones fuertes.
- Gritos.
- Sorpresas.
- Fails.
- Errores.
- Situaciones graciosas.
- Comentarios inesperados.
- Frases que tengan potencial para generar interés.
- Momentos de tensión.


============================================================
NO SELECCIONAR
============================================================

No selecciones:

- "hola"
- "adiós"
- saludos
- presentaciones
- contar números
- conversaciones normales
- frases aisladas sin contexto
- comentarios completamente normales
- silencios
- momentos que solamente parecen interesantes
  porque tú estás suponiendo qué ocurre en el juego


============================================================
MUY IMPORTANTE
============================================================

NO confundas una reacción con un evento.

Por ejemplo:

"hola"

NO es una reacción interesante.

"¡NOOOOO!"

tampoco demuestra por sí solo qué ocurrió.

"¡NOOO! Me mataron cuando iba a ganar"

sí contiene evidencia suficiente de un evento.


============================================================
SI NO HAY MOMENTOS
============================================================

Devuelve:

{{
    "moments": []
}}

No intentes llenar la lista.


============================================================
CANTIDAD
============================================================

Puedes devolver entre 0 y 10 momentos.

No es necesario devolver 10.

Devuelve únicamente los momentos que realmente tengan
potencial.


============================================================
TIMESTAMPS
============================================================

"moment_start" y "moment_end" deben señalar el momento
exacto donde aparece la evidencia en la transcripción.

No generes todavía el clip completo.


============================================================
SCORE
============================================================

90-100:
momento excepcional.

80-89:
momento claramente interesante.

70-79:
buen momento.

65-69:
interés moderado.

Menos de 65:
no devolver.


============================================================
CATEGORÍAS PERMITIDAS
============================================================

Usa solamente:

"jugada"
"reaccion"
"fail"
"gracioso"
"tension"
"sorpresa"
"victoria"
"derrota"
"comentario"


============================================================
FORMATO OBLIGATORIO
============================================================

Devuelve únicamente JSON.

{{
    "moments": [
        {{
            "moment_start": 125.2,
            "moment_end": 132.4,
            "score": 86,
            "category": "jugada",
            "evidence": "El streamer menciona que eliminó al enemigo con un headshot.",
            "reason": "La transcripción describe explícitamente una eliminación.",
            "title": "Headshot inesperado"
        }}
    ]
}}


============================================================
TRANSCRIPCIÓN
============================================================

{transcript}
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return response["message"]["content"]


def create_clip_window(
    moment_start,
    moment_end,
    source_duration
):

    moment_center = (
        moment_start + moment_end
    ) / 2

    clip_start = (
        moment_center
        - TARGET_CLIP_DURATION / 2
    )

    clip_end = (
        clip_start
        + TARGET_CLIP_DURATION
    )

    if clip_start < 0:

        clip_start = 0

        clip_end = min(
            TARGET_CLIP_DURATION,
            source_duration
        )

    if clip_end > source_duration:

        clip_end = source_duration

        clip_start = max(
            0,
            clip_end - TARGET_CLIP_DURATION
        )

    return (
        round(clip_start, 2),
        round(clip_end, 2)
    )


def validate_moments(
    result,
    source_duration
):

    valid = []

    for moment in result.get(
        "moments",
        []
    ):

        try:

            start = float(
                moment["moment_start"]
            )

            end = float(
                moment["moment_end"]
            )

            score = float(
                moment["score"]
            )

        except (
            KeyError,
            TypeError,
            ValueError
        ):

            continue

        if start < 0:
            continue

        if end <= start:
            continue

        if start >= source_duration:
            continue

        if end > source_duration:
            end = source_duration

        if score < MIN_SCORE:
            continue

        if score > 100:
            continue

        evidence = str(
            moment.get(
                "evidence",
                ""
            )
        ).strip()

        reason = str(
            moment.get(
                "reason",
                ""
            )
        ).strip()

        title = str(
            moment.get(
                "title",
                ""
            )
        ).strip()

        category = str(
            moment.get(
                "category",
                ""
            )
        ).strip().lower()

        if not evidence:
            continue

        if not reason:
            continue

        if not title:
            continue

        allowed_categories = {
            "jugada",
            "reaccion",
            "fail",
            "gracioso",
            "tension",
            "sorpresa",
            "victoria",
            "derrota",
            "comentario",
        }

        if category not in allowed_categories:
            continue

        clip_start, clip_end = create_clip_window(
            start,
            end,
            source_duration
        )

        duration = (
            clip_end - clip_start
        )

        if duration < MIN_CLIP_DURATION:
            continue

        if duration > MAX_CLIP_DURATION:
            continue

        valid.append(
            {
                "start": clip_start,
                "end": clip_end,
                "duration": round(
                    duration,
                    2
                ),

                "moment_start": round(
                    start,
                    2
                ),

                "moment_end": round(
                    end,
                    2
                ),

                "score": round(
                    score
                ),

                "category": category,

                "evidence": evidence,

                "reason": reason,

                "title": title,
            }
        )

    valid.sort(
        key=lambda clip: clip["score"],
        reverse=True
    )

    return {
        "clips": valid
    }


def remove_overlapping_clips(
    clips
):

    final = []

    for clip in clips:

        overlaps = False

        for existing in final:

            if (
                clip["start"]
                < existing["end"]
                and
                clip["end"]
                > existing["start"]
            ):

                overlaps = True
                break

        if not overlaps:

            final.append(clip)

    return final


def main():

    print("=" * 60)
    print("CLIP FACTORY - ANALIZADOR QWEN")
    print("=" * 60)

    print("\nCargando transcripción...")

    data = load_transcript()

    duration = float(
        data["duration"]
    )

    print(
        f"Idioma: {data['language']}"
    )

    print(
        f"Duración: {duration:.2f} segundos"
    )

    print(
        f"Segmentos: "
        f"{len(data['segments'])}"
    )

    transcript = build_transcript(
        data
    )

    print(
        "\nEnviando transcripción "
        "a Qwen3 8B..."
    )

    response = analyze_with_qwen(
        transcript,
        duration
    )

    print(
        "\nRespuesta de Qwen:"
    )

    print("-" * 60)

    print(response)

    print("-" * 60)

    try:

        result = json.loads(
            response
        )

    except json.JSONDecodeError:

        print(
            "\nERROR: Qwen no devolvió "
            "JSON válido."
        )

        return

    print(
        "\nValidando momentos..."
    )

    validated = validate_moments(
        result,
        duration
    )

    clips = remove_overlapping_clips(
        validated["clips"]
    )

    validated["clips"] = clips

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
            validated,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        "\n" + "=" * 60
    )

    print(
        "ANÁLISIS TERMINADO"
    )

    print(
        "=" * 60
    )

    print(
        f"Momentos detectados por Qwen: "
        f"{len(result.get('moments', []))}"
    )

    print(
        f"Clips válidos: "
        f"{len(clips)}"
    )

    for index, clip in enumerate(
        clips,
        start=1
    ):

        print(
            f"\n#{index} "
            f"{clip['start']:.2f}s → "
            f"{clip['end']:.2f}s"
        )

        print(
            f"Score: {clip['score']}"
        )

        print(
            f"Categoría: "
            f"{clip['category']}"
        )

        print(
            f"Título: "
            f"{clip['title']}"
        )

        print(
            f"Evidencia: "
            f"{clip['evidence']}"
        )

    print(
        f"\nResultado: {OUTPUT}"
    )


if __name__ == "__main__":
    main()