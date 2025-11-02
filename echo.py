import os
import sys
import json
from pathlib import Path
import fitz  # PyMuPDF
from crewai import Agent, Task, Process, Crew

os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["OPENAI_MODEL_NAME"] = "mistral"

BASE_DIR = Path(__file__).resolve().parent
output_path = BASE_DIR / "echo_output.json"


def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)


# ============ AGENTS (ECHO) ============
SettingAgent = Agent(
    role="Literary Setting Analyst",
    goal=(
        "Identify and describe locations and broader settings found in narrative or literary text."
    ),
    backstory=(
        "You specialise in reading prose and spotting where the scene is happening — specific places and the general type of place. "
        "You stay faithful to the text."
    ),
    verbose=True,
    allow_delegation=False,
)

AmbienceAgent = Agent(
    role="Ambient Detail Extractor",
    goal=(
        "Detect environmental or background elements mentioned in text that could correspond to audible ambience (weather, crowds, vehicles, natural sounds)."
    ),
    backstory=(
        "You focus on what the reader could reasonably hear in the scene, based only on what the text actually says."
    ),
    verbose=True,
    allow_delegation=False,
)

EmotionAgent = Agent(
    role="Narrative Emotion Profiler",
    goal=(
        "Identify clear emotional tones and, when obvious, story genres present in a passage of narrative text."
    ),
    backstory=(
        "You are good at noticing when a scene is tense, warm, fearful, mysterious, or romantic, but you do not make up emotions that aren't there."
    ),
    verbose=True,
    allow_delegation=False,
)

EchoFormatter = Agent(
    role="Literary JSON Formatter",
    goal=(
        "Format structured literary content into a clean JSON object for downstream use."
    ),
    backstory=(
        "You're a formatting specialist whose role is to convert structured literature content into clean, well-labelled JSON for a reading/sound system. "
        "You do not analyse or add interpretation — you just format."
    ),
    verbose=False,
    allow_delegation=False,
)

EchoStructureCorrector = Agent(
    role="Echo JSON Structure Corrector",
    goal="Ensure the final Echo output is a single valid JSON object without markdown or code fences.",
    backstory=(
        "You receive model output that should be JSON. You remove ```json ... ``` wrappers, extra text, and keep only the JSON object "
        "with the keys produced by upstream agents."
    ),
    verbose=False,
    allow_delegation=False,
)


def build_tasks(page_text: str):
    # Task 1: Extract setting (labelled plain text, no JSON)
    setting_task = Task(
        description=f"""Identify the setting in the text and express it as a generalised soundscape.

Output two labelled lines only:
setting_location: short tag (max 4 words) for the specific place, e.g. "hut_on_rock", "coastal_hut", "city_hotel_room", "forest_path", or "unknown".
setting_environment: broad ambience category (max 5 words) that could drive a soundscape, e.g. "stormy_coast", "rainy_city", "quiet_interior", "windy_seaside", "crowded_inn", or "unknown".

Prefer outdoor/ambiently rich descriptions over literal narrative sentences. Do not output long sentences or multiple locations. Pick the single dominant scene.

Text: {page_text}

Your answer must contain only the two labelled lines shown.
Any additional text, explanation, or commentary will make the answer invalid.
""",
        agent=SettingAgent,
        expected_output="setting_location: ...\nsetting_environment: ..."
    )

    # Task 2: Extract ambience (labelled plain text, no JSON)
    ambience_task = Task(
        description=f"""Extract background or environmental sounds implied or explicitly mentioned in the text. Output one labelled line:

ambient_sounds: comma-separated list of sounds (e.g. rain, wind, waves, city traffic, bar chatter). If none, write: ambient_sounds: 

Text: {page_text}

Your answer must contain only one line beginning with 'ambient_sounds:'.
Any additional text, sentences, or commentary make the answer invalid.
""",
        agent=AmbienceAgent,
        expected_output="ambient_sounds: ...",
    )

    # Task 3: Extract emotions / genre hints (labelled plain text, no JSON)
    emotion_task = Task(
        description=f"""Identify emotional tones and genre cues in the text. Output exactly two lines:

emotions: comma-separated emotions present, or empty if none.
genre_candidates: comma-separated genres present, or empty if none.

Include only emotions and genres supported by the text. Do not add commentary.

Text: {page_text}

Your answer must contain only the two labelled lines shown.
Do not include introductions, explanations, or commentary of any kind.
""",
        agent=EmotionAgent,
        context=[setting_task, ambience_task],
        expected_output="emotions: ...\ngenre_candidates: ...",
    )

    # Task 4: Convert labelled ECHO fields into final JSON
    structure_task = Task(
        description="""
Build a JSON object from the previous labelled outputs.

Output ONLY:

{
  "setting": {
    "location": "...",
    "environment": "..."
  },
  "ambient_sounds": [...],
  "emotions": [...],
  "genre_candidates": [...]
}

Rules:
- No extra text, no headings, no markdown, no code fences.
- If a field is missing, omit it.
- Convert comma-separated values to JSON arrays.
- Answer must start with '{' and end with '}'.
""",
        agent=EchoFormatter,
        context=[setting_task, ambience_task, emotion_task],
        expected_output=(
            "JSON object with keys: setting (with location, environment), ambient_sounds, emotions, genre_candidates. "
            "No markdown, no explanation."
        ),
    )

    echo_structure_task = Task(
        description="""
Ensure the previous output is ONLY a JSON object with these keys:
- setting (with location, environment)
- ambient_sounds (array)
- emotions (array)
- genre_candidates (array)

If it already is valid JSON, return it unchanged.
If there are code fences or text before/after, remove them.
Do NOT add explanations or extra text.
Output must start with '{' and end with '}'.
""",
        agent=EchoStructureCorrector,
        context=[structure_task],
        expected_output="""{
  "setting": {
    "location": "generic_interior",
    "environment": "neutral"
  },
  "ambient_sounds": [],
  "emotions": [],
  "genre_candidates": []
}"""
    )

    return setting_task, ambience_task, emotion_task, structure_task, echo_structure_task


def run_echo_on_text(text: str):
    setting_task, ambience_task, emotion_task, structure_task, echo_structure_task = build_tasks(text)

    crew = Crew(
        agents=[SettingAgent, AmbienceAgent, EmotionAgent, EchoFormatter, EchoStructureCorrector],
        tasks=[setting_task, ambience_task, emotion_task, structure_task, echo_structure_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    # crew.kickoff() returns a CrewOutput; get the raw string
    final_text = getattr(result, "raw", str(result))
    return final_text


if __name__ == "__main__":
    # usage:
    #   python Echo.py path/to/book.pdf
    # or:
    #   python Echo.py "some raw text to analyse"
    if len(sys.argv) < 2:
        print("❌ Please provide a PDF path or some raw text.")
        sys.exit(1)

    arg = sys.argv[1]

    # If it's a file
    if Path(arg).exists() and arg.lower().endswith(".pdf"):
        # PDF branch: extract pages, run echo per page, output list of page-level JSONs
        def extract_pages_from_pdf(pdf_path):
            doc = fitz.open(pdf_path)
            return [page.get_text() for page in doc]

        pages = extract_pages_from_pdf(arg)
        page_outputs = []
        last_setting = None
        last_ambient = None
        last_emotions = None
        last_genres = None

        for pidx, page_text in enumerate(pages, start=1):
            result = run_echo_on_text(page_text)
            try:
                parsed = json.loads(result)
            except json.JSONDecodeError:
                page_outputs.append({"page": pidx, "raw": result})
                continue

            # normalise missing sections
            setting = parsed.get("setting") or {}
            ambient = parsed.get("ambient_sounds") or []
            emotions = parsed.get("emotions") or []
            genres = parsed.get("genre_candidates") or []

            # if this page didn't give us a good setting but we have a previous one, carry it forward
            loc = setting.get("location") or ""
            env = setting.get("environment") or ""
            setting_too_generic = env in ("unknown", "neutral", "")

            if setting_too_generic and last_setting is not None:
                setting = last_setting
            else:
                # update last_setting only when this page actually had something concrete
                if env not in ("unknown", "neutral", ""):
                    last_setting = setting

            # carry forward ambience if empty
            if (not ambient) and (last_ambient is not None):
                ambient = last_ambient
            else:
                if ambient:
                    last_ambient = ambient

            # carry forward emotions if empty
            if (not emotions) and (last_emotions is not None):
                emotions = last_emotions
            else:
                if emotions:
                    last_emotions = emotions

            # carry forward genres if empty
            if (not genres) and (last_genres is not None):
                genres = last_genres
            else:
                if genres:
                    last_genres = genres

            parsed["setting"] = setting
            parsed["ambient_sounds"] = ambient
            parsed["emotions"] = emotions
            parsed["genre_candidates"] = genres
            parsed["page"] = pidx

            page_outputs.append(parsed)

        with open(output_path, "w") as f:
            json.dump(page_outputs, f, indent=2)
        print("✅ ECHO per-page analysis written to", output_path)
        print(json.dumps(page_outputs, indent=2))
    else:
        # treat it as raw text
        text = arg

        analysis_result = run_echo_on_text(text)

        try:
            parsed = json.loads(analysis_result)
            with open(output_path, "w") as f:
                json.dump(parsed, f, indent=2)
            print("✅ ECHO analysis written to", output_path)
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            # fall back to raw text
            with open(output_path, "w") as f:
                f.write(analysis_result)
            print("✅ ECHO analysis (raw) written to", output_path)
            print(analysis_result)