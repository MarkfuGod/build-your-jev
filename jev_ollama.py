"""Small client for the local Ollama HTTP API."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from jev_like import RequestError

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def ollama_post(path: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode()
    request = Request(
        OLLAMA_HOST + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            parsed = json.loads(response.read().decode())
    except HTTPError as error:
        detail = error.read().decode(errors="replace")
        try:
            message = json.loads(detail).get("error", detail)
        except json.JSONDecodeError:
            message = detail or error.reason
        raise RequestError(f"Ollama request failed: {message}") from error
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RequestError(
            "Ollama is not available. Start it, then pull deepseek-ocr:3b."
        ) from error
    if not isinstance(parsed, dict):
        raise RequestError("Ollama returned an unexpected response")
    return parsed
