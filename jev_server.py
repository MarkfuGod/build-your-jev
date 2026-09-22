#!/usr/bin/env python3
"""TypeSafe-shaped local HTTP server backed by the Jev-like MLX engine."""

from __future__ import annotations

import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import Request, urlopen

from jev_like import (
    MODEL_ALIAS,
    MODEL_CONTEXT_TOKENS,
    MODEL_REPO,
    MODEL_REVISION,
    PROBABILITY_STATUS,
    JevLikeEngine,
    RequestError,
)
from jev_ocr import OCR_MODEL, ocr_image
from jev_preprocess import prepare_request

MAX_REQUEST_BYTES = 32 * 1024 * 1024
PAGE = Path(__file__).with_name("youtube.html").read_text()
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}
_SHORT_HOSTS = {"youtu.be", "www.youtu.be"}

YOUTUBE_QUESTIONS = {
    "kind": {
        "type": "choice",
        "instructions": "What kind of video is this, based on its title and channel?",
        "criteria": {
            "music": "A song, official music video, or audio release",
            "tutorial": "A lesson, course, or how-to",
            "news": "News, reporting, or current events",
            "comedy": "Comedy, parody, or entertainment sketch",
            "vlog": "A personal clip or everyday life",
            "other": "None of the listed kinds",
        },
    },
    "educational": {
        "type": "noul",
        "instructions": "Is this video primarily trying to teach something?",
        "criteria": {
            "true": "The title presents instruction or explanation",
            "false": "The title presents entertainment, music, or a personal moment",
        },
    },
    "energy": {
        "type": "score",
        "instructions": "How energetic does the title sound?",
        "criteria": ["Calm", "Lively", "Intense"],
    },
}


def youtube_video_id(url: str) -> str:
    """Accept a public YouTube watch, short, embed, or youtu.be link."""
    if not isinstance(url, str) or not url.strip():
        raise RequestError("Paste a YouTube video link")
    parsed = urlsplit(url.strip())
    host = parsed.netloc.lower().split(":", 1)[0]
    if host in _SHORT_HOSTS:
        video = parsed.path.strip("/").split("/", 1)[0]
    elif host in _YOUTUBE_HOSTS:
        parts = [part for part in parsed.path.split("/") if part]
        if parsed.path == "/watch":
            video = parse_qs(parsed.query).get("v", [""])[0]
        elif parts and parts[0] in {"embed", "shorts", "live", "v"} and len(parts) >= 2:
            video = parts[1]
        else:
            raise RequestError("Use a link to one YouTube video")
    else:
        raise RequestError("Only YouTube video links are accepted")
    if not _VIDEO_ID.fullmatch(video):
        raise RequestError("That link does not contain a YouTube video id")
    return video


def fetch_youtube(video_id: str) -> dict[str, str]:
    """Read the public title and channel. The model does not watch the video."""
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    endpoint = "https://www.youtube.com/oembed?" + urlencode({"url": watch_url, "format": "json"})
    request = Request(endpoint, headers={"User-Agent": "diffusiongemma-jevlike/0.2"})
    try:
        with urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode())
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RequestError("YouTube did not return public details for this video") from error
    title = payload.get("title")
    channel = payload.get("author_name")
    thumbnail = payload.get("thumbnail_url")
    if not isinstance(title, str) or not title:
        raise RequestError("YouTube did not return a title for this video")
    return {
        "id": video_id,
        "title": title,
        "channel": channel if isinstance(channel, str) else "",
        "thumbnail": thumbnail if isinstance(thumbnail, str) else "",
        "watch_url": watch_url,
        "embed_url": f"https://www.youtube.com/embed/{video_id}",
    }


def evaluate_youtube(engine: JevLikeEngine, url: str) -> dict[str, Any]:
    video = fetch_youtube(youtube_video_id(url))
    decision = engine.evaluate(
        {
            "state": {
                "title": video["title"],
                "channel": video["channel"],
                "url": video["watch_url"],
            },
            "questions": YOUTUBE_QUESTIONS,
        }
    )
    return {"video": video, **decision}


class Handler(BaseHTTPRequestHandler):
    engine: JevLikeEngine
    api_key: str | None = None

    def _json(self, status: int, body: Any, **headers: str) -> None:
        payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Probability-Status", PROBABILITY_STATUS)
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)

    def _authorized(self) -> bool:
        if not self.api_key:
            return True
        return self.headers.get("Authorization") == f"Bearer {self.api_key}"

    def _html(self, status: int, body: str) -> None:
        payload = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path == "/":
            self._html(200, PAGE)
            return
        if path == "/health":
            self._json(200, {"status": "ok", "model_loaded": self.engine.is_loaded})
            return
        if path == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": MODEL_ALIAS,
                            "object": "model",
                            "owned_by": "local",
                            "source": MODEL_REPO,
                            "revision": MODEL_REVISION,
                        }
                    ],
                },
            )
            return
        self._json(404, {"error": {"message": "Not found", "type": "not_found"}})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path not in {"/v1/systemone", "/v1/ocr", "/youtube"}:
            self._json(404, {"error": {"message": "Not found", "type": "not_found"}})
            return
        if not self._authorized():
            self._json(401, {"error": {"message": "Invalid API key", "type": "authentication_error"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise RequestError(f"request body must be 1-{MAX_REQUEST_BYTES} bytes")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise RequestError("request body must be an object")
            if path == "/youtube":
                response = evaluate_youtube(self.engine, request.get("url"))
            elif path == "/v1/ocr":
                prompt = request.get("prompt")
                response = {
                    "model": OCR_MODEL,
                    "text": ocr_image(request.get("image"), **({} if prompt is None else {"prompt": prompt})),
                }
            else:
                cleaned, meta = prepare_request(request)
                response = self.engine.evaluate(cleaned)
                if meta:
                    response["preprocess"] = meta
        except (RequestError, json.JSONDecodeError, UnicodeDecodeError) as error:
            self._json(422, {"error": {"message": str(error), "type": "invalid_request_error"}})
            return
        except Exception as error:
            self.log_error("evaluation failed: %s", error)
            self._json(500, {"error": {"message": str(error), "type": "server_error"}})
            return
        self._json(200, response)

    def log_message(self, message: str, *args: Any) -> None:
        print(f"{self.address_string()} - {message % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model", help=f"Local directory or Hub id (default: local weights, then {MODEL_REPO})")
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Use this temperature for every question type. Default: fitted per-type values.",
    )
    parser.add_argument("--max-tokens", type=int, default=MODEL_CONTEXT_TOKENS)
    parser.add_argument("--cache-limit-mib", type=int, default=256)
    parser.add_argument("--eager", action="store_true", help="Load model before accepting requests")
    args = parser.parse_args()

    Handler.engine = JevLikeEngine(
        args.model,
        revision=args.revision,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        cache_limit_mib=args.cache_limit_mib,
    )
    Handler.api_key = os.environ.get("JEV_API_KEY")
    if args.eager:
        Handler.engine.load()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"YouTube click page: http://{args.host}:{args.port}/", flush=True)
    print(f"Jev-like local API: http://{args.host}:{args.port}/v1/systemone", flush=True)
    print(f"Model: {Handler.engine.model_source}", flush=True)
    fitted = " ".join(f"{kind}={value:g}" for kind, value in Handler.engine.temperatures.items())
    print(f"Temperature: {fitted}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
