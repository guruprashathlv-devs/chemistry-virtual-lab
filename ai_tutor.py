"""Optional AI tutor using an OpenAI-compatible Chat Completions endpoint.

Set CHEM_LAB_AI_API_KEY (and optionally CHEM_LAB_AI_BASE_URL / CHEM_LAB_AI_MODEL)
in environment variables or Streamlit secrets. If no key is configured, callers
should fall back to the built-in curated tutor so the app remains usable offline.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

SYSTEM_PROMPT = """You are ChemLab Tutor, a careful chemistry teaching assistant for undergraduate students.\n\nTeach step by step in clear language. Define symbols and units, show equations, and check units. For calculations, state assumptions and show the working. Distinguish educational simulations from real laboratory measurements. Never invent experimental data or claim a generated molecular conformer is experimentally verified. Do not provide instructions for dangerous chemical synthesis, explosives, poisons, or unsafe handling. Encourage following institutional lab safety procedures. If the question is ambiguous, ask a short clarifying question. Keep answers structured and exam-friendly."""


def is_configured() -> bool:
    key = os.getenv("CHEM_LAB_AI_API_KEY", "").strip()
    if not key:
        try:
            import streamlit as st
            key = str(st.secrets.get("CHEM_LAB_AI_API_KEY", "")).strip()
        except Exception:
            key = ""
    return bool(key)


def _secret(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default


def ask_tutor(question: str, topic: str = "General chemistry", context: str = "") -> str:
    """Call an OpenAI-compatible chat endpoint without exposing secrets to the UI."""
    api_key = _secret("CHEM_LAB_AI_API_KEY")
    if not api_key:
        raise RuntimeError("AI is not configured. Add CHEM_LAB_AI_API_KEY to Streamlit secrets or the environment.")
    base_url = _secret("CHEM_LAB_AI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = _secret("CHEM_LAB_AI_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "temperature": 0.25,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Topic: {topic}\nRelevant app context: {context or 'None supplied'}\nStudent question: {question}"},
        ],
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
        answer = data["choices"][0]["message"]["content"]
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("The AI provider returned an empty response.")
        return answer.strip()
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace")
        raise RuntimeError(f"AI provider returned HTTP {exc.code}. Check the API endpoint, model, and key. {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach the AI provider: {exc.reason}") from exc
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("The AI provider response was not in the expected chat-completions format.") from exc
