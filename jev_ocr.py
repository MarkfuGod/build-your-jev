"""Turn an image into text with Ollama's deepseek-ocr:3b model."""

from __future__ import annotations

import base64
import binascii
import re
from pathlib import Path
from typing import Any

from jev_like import RequestError
from jev_ollama import ollama_post

OCR_MODEL = "deepseek-ocr:3b"
OCR_PROMPT = "Free OCR."
MAX_IMAGE_BYTES = 32 * 1024 * 1024
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
_SPECIAL_TOKEN = re.compile(r"<\|[^|<>]*\|>")
_ROLE_PREFIX = re.compile(r"^(?:system|user|assistant)\n")


def image_bytes(value: Any) -> bytes:
    """Accept a data URL, raw base64, or a local image path."""
    if not isinstance(value, str) or not value.strip():
        raise RequestError("image must be a data URL, base64 string, or local image path")
    text = value.strip()
    if text.startswith("data:"):
        header, separator, payload = text.partition(",")
        if not separator or not header.startswith("data:image/"):
            raise RequestError("image data URL must be an image")
        try:
            raw = base64.b64decode(payload, validate=True)
        except binascii.Error as error:
            raise RequestError("image data URL is not valid base64") from error
    else:
        path = Path(text).expanduser()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            if path.stat().st_size > MAX_IMAGE_BYTES:
                raise RequestError(f"image must be at most {MAX_IMAGE_BYTES} bytes")
            raw = path.read_bytes()
        else:
            try:
                raw = base64.b64decode(text, validate=True)
            except binascii.Error as error:
                raise RequestError("image must be a data URL, base64 string, or local image path") from error
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise RequestError(f"image must be 1-{MAX_IMAGE_BYTES} bytes")
    return raw


def clean_ocr_text(text: str) -> str:
    """Drop chat-template tokens this OCR model sometimes emits around the text."""
    text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text, flags=re.DOTALL)
    if _SPECIAL_TOKEN.search(text) and "<|ref|>" not in text:
        text = _SPECIAL_TOKEN.split(text)[-1]
    cleaned = _SPECIAL_TOKEN.sub("", text)
    return _ROLE_PREFIX.sub("", cleaned).strip()


def ocr_image(image: Any, *, prompt: str = OCR_PROMPT) -> str:
    """Read text from an image. The result is text, not a decision."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise RequestError("ocr prompt must be a nonempty string")
    encoded = base64.b64encode(image_bytes(image)).decode("ascii")
    payload = ollama_post(
        "/api/chat",
        {
            "model": OCR_MODEL,
            "stream": False,
            "keep_alive": "5m",
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": "user",
                    "content": prompt.strip(),
                    "images": [encoded],
                }
            ],
        },
        timeout=180,
    )
    message = payload.get("message")
    text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise RequestError("OCR returned no text")
    cleaned = clean_ocr_text(text)
    if not cleaned:
        raise RequestError("OCR returned no text")
    return cleaned


def merge_image_text(state: Any, image_text: str) -> Any:
    """Attach recognized text to the request state."""
    if state is None or state == "" or state == {} or state == []:
        return {"image_text": image_text}
    if isinstance(state, dict):
        merged = dict(state)
        merged["image_text"] = image_text
        return merged
    if isinstance(state, list):
        return {"items": state, "image_text": image_text}
    return {"context": state, "image_text": image_text}
