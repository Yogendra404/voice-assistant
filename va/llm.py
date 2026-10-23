"""LLM stage (Gemini) with short conversation memory."""
from google.genai import types
from . import config
SYSTEM_PROMPT = (
    "You are a friendly voice assistant. Your answer will be read aloud, so: "
    "keep it short (1-3 sentences), use plain spoken language, and never use "
    "markdown, bullet points, emojis or URLs. If the user's message seems garbled "
    "or doesn't make sense, politely ask them to repeat it instead of guessing."
)
MAX_TURNS = 10
def respond(user_text: str, history: list[dict] | None = None, unclear: bool = False) -> str:
    client = config.get_client()
    history = (history or [])[-MAX_TURNS * 2:]
    user_content = user_text
    if unclear:
        user_content = ("[Note: the speech recognizer was not confident; "
                        f"this transcript may be wrong] {user_text}")
    contents = []
    for m in history:
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=user_content)]))
    resp = client.models.generate_content(
        model=config.LLM_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.6,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response")
    return text
