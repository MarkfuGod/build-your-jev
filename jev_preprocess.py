"""Image OCR that runs before the decision model."""

from __future__ import annotations

from typing import Any, Mapping

from jev_like import RequestError
from jev_ocr import OCR_MODEL, OCR_PROMPT, merge_image_text, ocr_image


def prepare_request(
    request: Mapping[str, Any],
    *,
    read_image=ocr_image,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Turn an image into text and attach that text to the state."""
    if not isinstance(request, Mapping):
        raise RequestError("request body must be an object")
    prepared = dict(request)
    meta: dict[str, Any] = {}
    if "image" in prepared and prepared.get("image") not in (None, ""):
        prompt = prepared.get("ocr_prompt", OCR_PROMPT)
        if not isinstance(prompt, str) or not prompt.strip():
            raise RequestError("ocr prompt must be a nonempty string")
        image_text = read_image(prepared["image"], prompt=prompt)
        prepared["state"] = merge_image_text(prepared.get("state"), image_text)
        meta["ocr_model"] = OCR_MODEL
        meta["image_text"] = image_text
    prepared.pop("image", None)
    prepared.pop("ocr_prompt", None)
    prepared.pop("retrieve", None)
    return prepared, meta
