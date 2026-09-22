#!/usr/bin/env python3
"""Local Jev-like typed decisions from Qwen option logits on Apple Silicon."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

MODEL_REPO = "mlx-community/Qwen3.5-4B-OptiQ-4bit"
MODEL_REVISION = "6cb5bdfd0bf15f484881fb9f1ab6d7c840fddde9"
MODEL_ALIAS = "local-qwen3.5-4b-jevlike"
LOCAL_MODEL = Path(__file__).resolve().parent / "models" / "qwen3.5-4b-optiq-4bit"
MODEL_CONTEXT_TOKENS = 262144

LETTERS = "ABCDEFGHIJKLMNOP"
MAX_CHOICES = len(LETTERS)
MAX_QUESTIONS = 128
# Reuse one state cache, but do not replicate it across every question at once.
SUFFIX_BATCH = 16
# Separates the shared state from the per-question suffix. The newline form
# cannot appear inside JSON, so every question shares one exact text prefix.
QUESTION_MARK = "\nQuestion:\n"
SYSTEM_PROMPT = (
    "Apply the supplied question and criteria to the supplied state. "
    "Choose exactly one listed option. Respond with only its uppercase letter, "
    "with no explanation or reasoning."
)
# Temperature scaling fit by negative log-likelihood on the 1000-question suite
# at raw temperature 1. Leave-one-scenario-out: noul and score improve; choice
# does not, so choice stays at 1. Argmax and a two-way 0.5 cut are unchanged.
FITTED_TEMPERATURE = {
    "noul": 0.6882,
    "choice": 1.0,
    "score": 2.9052,
}
PROBABILITY_STATUS = (
    "Temperature-scaled option probabilities. Noul uses 0.6882 and score uses "
    "2.9052, fitted on the local 1000-question suite; choice stays at 1. "
    "Refit before treating the values as confidence on a new workload."
)

JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]


class RequestError(ValueError):
    """Raised when a System One request does not match the supported contract."""


@dataclass(frozen=True)
class PreparedQuestion:
    name: str
    kind: str
    option_ids: tuple[str, ...]
    input_ids: tuple[int, ...]
    suffix_ids: tuple[int, ...]
    answer_token_ids: tuple[int, ...]
    prompt_sha256: str
    criteria: Any


def _json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=isinstance(value, dict),
        )
    except (TypeError, ValueError) as error:
        raise RequestError("State, instructions, and criteria must be finite JSON values") from error


def question_user_content(state: Any, instructions: Any, options: Any) -> str:
    """Place the state before a boundary so later questions can reuse its tokens."""
    return _json({"state": state}) + QUESTION_MARK + _json(
        {"instructions": instructions, "options": options}
    )


def render_chat(tokenizer: Any, user_content: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(prompt, str) or not prompt:
        raise RequestError("chat template did not return a prompt")
    return prompt


def shared_token_split(
    tokenizer: Any, state: Any, user_contents: Sequence[str]
) -> tuple[tuple[int, ...], list[tuple[int, ...]], list[str], list[list[int]]]:
    """Return one state prefix and the unmatched suffix of every full prompt.

    The prefix is shortened by at most a few tokens when the tokenizer would
    otherwise merge the boundary with the following question text.
    """
    if not user_contents:
        raise RequestError("at least one question is required")
    marker = _json({"state": state}) + QUESTION_MARK
    prompts = [render_chat(tokenizer, content) for content in user_contents]
    if any(prompt.count(marker) != 1 for prompt in prompts):
        raise RequestError("the shared state could not be located in every prompt")
    head = prompts[0][: prompts[0].index(marker) + len(marker)]
    if any(not prompt.startswith(head) for prompt in prompts):
        raise RequestError("questions do not share one state prefix")
    prefix_ids = list(tokenizer.encode(head, add_special_tokens=False))
    full_ids = [list(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    limit = len(prefix_ids)
    shrink = 0
    while limit > 0 and any(ids[:limit] != prefix_ids[:limit] for ids in full_ids):
        limit -= 1
        shrink += 1
        if shrink > 8:
            raise RequestError("state prefix tokens do not line up across questions")
    if limit < 1 or any(len(ids) <= limit for ids in full_ids):
        raise RequestError("each question needs text after the shared state")
    prefix = tuple(prefix_ids[:limit])
    suffixes = [tuple(ids[limit:]) for ids in full_ids]
    return prefix, suffixes, prompts, full_ids


def stable_softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Normalize finite option logits without arbitrary clipping."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise RequestError("temperature must be a finite number greater than zero")
    if len(logits) < 2 or any(not math.isfinite(float(value)) for value in logits):
        raise RequestError("At least two finite option logits are required")
    scaled = [float(value) / temperature for value in logits]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    total = sum(weights)
    return [weight / total for weight in weights]


def distribution_confidence(probabilities: Sequence[float]) -> float:
    """Map peak probability from chance=0 to certainty=1.

    This mirrors the simple Choice approximation in TypeSafe's public confidence
    documentation. It measures distribution concentration, not empirical accuracy.
    """
    count = len(probabilities)
    if count < 2:
        raise RequestError("At least two probabilities are required")
    peak = max(float(value) for value in probabilities)
    return max(0.0, min(1.0, (count * peak - 1.0) / (count - 1.0)))


def _validate_state(state: Any) -> None:
    if not isinstance(state, (str, dict, list)) or not state:
        raise RequestError("state must be a nonempty string, object, or array")
    _json(state)


def _validate_question_name(name: Any) -> str:
    if not isinstance(name, str) or not name:
        raise RequestError("question ids must be nonempty strings")
    return name


def _choice_options(criteria: Any) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
    if not isinstance(criteria, Mapping) or not 2 <= len(criteria) <= MAX_CHOICES:
        raise RequestError(f"choice criteria must contain 2-{MAX_CHOICES} named options")
    option_ids: list[str] = []
    rendered: list[dict[str, Any]] = []
    for index, (option_id, description) in enumerate(criteria.items()):
        if not isinstance(option_id, str) or not option_id:
            raise RequestError("choice option ids must be nonempty strings")
        _json(description)
        option_ids.append(option_id)
        rendered.append(
            {
                "letter": LETTERS[index],
                "value": option_id,
                "description": description,
            }
        )
    return tuple(option_ids), tuple(rendered)


def _score_options(criteria: Any) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
    if (
        not isinstance(criteria, Sequence)
        or isinstance(criteria, (str, bytes))
        or not 2 <= len(criteria) <= 10
    ):
        raise RequestError("score criteria must contain 2-10 ordered levels")
    rendered: list[dict[str, Any]] = []
    for index, description in enumerate(criteria):
        _json(description)
        rendered.append(
            {
                "letter": LETTERS[index],
                "value": str(index),
                "description": description,
            }
        )
    option_ids = tuple(str(index) for index in range(len(criteria)))
    return option_ids, tuple(rendered)


def _noul_options(criteria: Any) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
    if criteria is None:
        criteria = {}
    if not isinstance(criteria, Mapping) or any(key not in {"true", "false"} for key in criteria):
        raise RequestError("noul criteria may only describe true and false")
    true_description = criteria.get("true", "The answer is yes or true")
    false_description = criteria.get("false", "The answer is no or false")
    _json(true_description)
    _json(false_description)
    return (
        ("true", "false"),
        (
            {"letter": "A", "value": "true", "description": true_description},
            {"letter": "B", "value": "false", "description": false_description},
        ),
    )


def normalize_questions(questions: Any) -> list[dict[str, Any]]:
    """Validate the public Choice, Score, and Noul request shapes."""
    if not isinstance(questions, Mapping) or not questions:
        raise RequestError("questions must be a nonempty object")
    if len(questions) > MAX_QUESTIONS:
        raise RequestError(f"questions may contain at most {MAX_QUESTIONS} entries")
    normalized: list[dict[str, Any]] = []
    for raw_name, raw_question in questions.items():
        name = _validate_question_name(raw_name)
        if not isinstance(raw_question, Mapping):
            raise RequestError(f"question {name!r} must be an object")
        kind = raw_question.get("type")
        instructions = raw_question.get("instructions")
        _json(instructions)
        criteria = raw_question.get("criteria")
        if kind == "choice":
            option_ids, options = _choice_options(criteria)
            default_instructions = "Which option best applies?"
        elif kind == "score":
            option_ids, options = _score_options(criteria)
            default_instructions = "Which ordered level best applies?"
        elif kind == "noul":
            option_ids, options = _noul_options(criteria)
            default_instructions = "Is this true?"
        else:
            raise RequestError(f"question {name!r} has unsupported type {kind!r}")
        normalized.append(
            {
                "name": name,
                "kind": kind,
                "instructions": default_instructions if instructions is None else instructions,
                "criteria": criteria,
                "option_ids": option_ids,
                "options": options,
            }
        )
    return normalized


def _answer(kind: str, option_ids: Sequence[str], criteria: Any, probabilities: Sequence[float]) -> dict[str, Any]:
    mapped = {option_id: float(probability) for option_id, probability in zip(option_ids, probabilities)}
    if kind == "noul":
        return {"type": "noul", "noul": mapped["true"]}
    confidence = distribution_confidence(probabilities)
    if kind == "choice":
        winner = option_ids[max(range(len(probabilities)), key=probabilities.__getitem__)]
        return {
            "type": "choice",
            "choice": winner,
            "probabilities": mapped,
            "confidence": confidence,
        }
    score = sum(index * probability for index, probability in enumerate(probabilities))
    return {
        "type": "score",
        "score": score,
        "legend": {str(index): description for index, description in enumerate(criteria)},
        "probabilities": mapped,
        "confidence": confidence,
    }


class JevLikeEngine:
    """Score every question against one prefilled state on a pinned Qwen3.5-4B checkpoint."""

    def __init__(
        self,
        model: str | Path | None = None,
        *,
        revision: str = MODEL_REVISION,
        temperature: float | None = None,
        max_tokens: int = MODEL_CONTEXT_TOKENS,
        cache_limit_mib: int = 256,
    ) -> None:
        if not 1 <= max_tokens <= MODEL_CONTEXT_TOKENS:
            raise RequestError(f"max_tokens must be 1-{MODEL_CONTEXT_TOKENS}")
        if cache_limit_mib < 0:
            raise RequestError("cache_limit_mib must be nonnegative")
        self.temperatures = dict(FITTED_TEMPERATURE)
        if temperature is not None:
            stable_softmax([0.0, 0.0], temperature)
            self.temperatures = {kind: float(temperature) for kind in FITTED_TEMPERATURE}
        if model is None:
            model = LOCAL_MODEL if (LOCAL_MODEL / "config.json").is_file() else MODEL_REPO
        self.model_source = str(model)
        self.revision = revision
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_limit_mib = cache_limit_mib
        self.model = None
        self.tokenizer = None
        self._load_seconds = 0.0
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.is_loaded:
            return
        import mlx.core as mx
        from mlx_lm import load

        mx.set_default_device(mx.gpu)
        mx.set_cache_limit(self.cache_limit_mib * 1024 * 1024)
        started = time.perf_counter()
        local = Path(self.model_source).is_dir()
        self.model, self.tokenizer = load(
            self.model_source,
            revision=None if local else self.revision,
            tokenizer_config={"trust_remote_code": False},
        )
        self.model.eval()
        mx.eval(self.model.parameters())
        mx.synchronize()
        self._load_seconds = time.perf_counter() - started

    def _answer_token_ids(self, prompt: str, count: int, input_ids: list[int]) -> tuple[int, ...]:
        assert self.tokenizer is not None
        slots: list[int] = []
        for letter in LETTERS[:count]:
            encoded = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(encoded) != 1 or self.tokenizer.decode(encoded) != letter:
                raise RequestError(f"answer slot {letter!r} is not one exact round-trip token")
            if self.tokenizer.encode(prompt + letter, add_special_tokens=False) != input_ids + encoded:
                raise RequestError(f"answer boundary changes tokenization for slot {letter!r}")
            slots.append(encoded[0])
        if len(slots) != len(set(slots)):
            raise RequestError("answer-slot token ids collide")
        return tuple(slots)

    def _prepare_batch(
        self, state: Any, questions: Sequence[dict[str, Any]]
    ) -> tuple[tuple[int, ...], list[PreparedQuestion]]:
        assert self.tokenizer is not None
        contents = [
            question_user_content(state, question["instructions"], question["options"])
            for question in questions
        ]
        prefix, suffixes, prompts, full_ids = shared_token_split(self.tokenizer, state, contents)
        prepared: list[PreparedQuestion] = []
        for question, prompt, input_ids, suffix in zip(questions, prompts, full_ids, suffixes):
            if len(input_ids) > self.max_tokens:
                raise RequestError(
                    f"question {question['name']!r} uses {len(input_ids)} tokens; "
                    f"limit is {self.max_tokens} and inputs are never truncated"
                )
            slots = self._answer_token_ids(prompt, len(question["option_ids"]), input_ids)
            prepared.append(
                PreparedQuestion(
                    name=question["name"],
                    kind=question["kind"],
                    option_ids=question["option_ids"],
                    input_ids=tuple(input_ids),
                    suffix_ids=suffix,
                    answer_token_ids=slots,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    criteria=question["criteria"],
                )
            )
        return prefix, prepared

    def _project_slots(self, hidden: Any, positions: Sequence[int], prepared: Sequence[PreparedQuestion]) -> list[Any]:
        import mlx.core as mx

        language_model = getattr(self.model, "language_model", self.model)
        if not hasattr(language_model, "model"):
            raise RuntimeError("Expected the native MLX-LM Qwen3.5 text model")
        last_hidden = hidden[mx.arange(len(prepared)), mx.array(positions)]
        if language_model.args.tie_word_embeddings:
            vocabulary = language_model.model.embed_tokens.as_linear(last_hidden)
        else:
            vocabulary = language_model.lm_head(last_hidden)
        return [
            vocabulary[index, mx.array(item.answer_token_ids)].astype(mx.float32)
            for index, item in enumerate(prepared)
        ]

    def _option_logits(
        self, prefix: Sequence[int], prepared: Sequence[PreparedQuestion]
    ) -> tuple[list[list[float]], dict[str, float]]:
        """Prefill the state once, then score every question suffix in one batch."""
        import mlx.core as mx

        assert self.model is not None and self.tokenizer is not None
        if not hasattr(self.model, "make_cache"):
            raise RuntimeError("Expected the native MLX-LM Qwen3.5 text model")
        language_model = getattr(self.model, "language_model", self.model)
        text_model = getattr(language_model, "model", None)
        if text_model is None:
            raise RuntimeError("Expected the native MLX-LM Qwen3.5 text model")

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise RequestError("tokenizer has neither a padding nor an EOS token")

        mx.synchronize()
        started = time.perf_counter()
        cache = self.model.make_cache()
        hidden = text_model(mx.array([list(prefix)]), cache=cache)
        mx.eval(hidden, [entry.state for entry in cache])
        mx.synchronize()
        prefill_seconds = time.perf_counter() - started

        logits: list[list[float]] = []
        suffix_seconds = 0.0
        for start in range(0, len(prepared), SUFFIX_BATCH):
            chunk = prepared[start : start + SUFFIX_BATCH]
            selected, elapsed = self._score_suffix_chunk(text_model, cache, pad_id, chunk)
            logits.extend(selected)
            suffix_seconds += elapsed
        return logits, {
            "prefill_seconds": prefill_seconds,
            "suffix_seconds": suffix_seconds,
        }

    def _score_suffix_chunk(self, text_model: Any, cache: Any, pad_id: int, prepared: Sequence[PreparedQuestion]) -> tuple[list[list[float]], float]:
        """Score a slice of questions against an already-prefilled state cache."""
        import mlx.core as mx

        lengths = [len(item.suffix_ids) for item in prepared]
        width = max(lengths)
        mx.synchronize()
        started = time.perf_counter()
        branches = [entry.merge([entry] * len(prepared)) for entry in cache]
        right_padding = [width - length for length in lengths]
        for entry in branches:
            entry.prepare(lengths=lengths, right_padding=right_padding)
        mx.eval([entry.state for entry in branches])
        tokens = mx.array(
            [list(item.suffix_ids) + [pad_id] * (width - len(item.suffix_ids)) for item in prepared]
        )
        hidden = text_model(tokens, cache=branches)
        selected = self._project_slots(hidden, [length - 1 for length in lengths], prepared)
        mx.eval(selected)
        mx.synchronize()
        return [values.tolist() for values in selected], time.perf_counter() - started

    def _independent_option_logits(self, prepared: Sequence[PreparedQuestion]) -> list[list[float]]:
        """Score each full prompt from scratch. Used to check the shared prefix."""
        import mlx.core as mx

        assert self.model is not None and self.tokenizer is not None
        language_model = getattr(self.model, "language_model", self.model)
        lengths = [len(item.input_ids) for item in prepared]
        width = max(lengths)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        tokens = mx.array(
            [list(item.input_ids) + [pad_id] * (width - len(item.input_ids)) for item in prepared]
        )
        hidden = language_model.model(tokens)
        selected = self._project_slots(hidden, [length - 1 for length in lengths], prepared)
        mx.eval(selected)
        mx.synchronize()
        return [values.tolist() for values in selected]

    def evaluate(self, request: Mapping[str, Any], *, include_debug: bool = False) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise RequestError("request body must be an object")
        state = request.get("state")
        _validate_state(state)
        normalized = normalize_questions(request.get("questions"))

        with self._lock:
            self.load()
            started = time.perf_counter()
            prefix, prepared = self._prepare_batch(state, normalized)
            logits, timing = self._option_logits(prefix, prepared)
            elapsed = time.perf_counter() - started

        suffix_tokens = sum(len(item.suffix_ids) for item in prepared)
        answers: dict[str, Any] = {}
        debug: dict[str, Any] = {}
        for item, values in zip(prepared, logits):
            probabilities = stable_softmax(values, self.temperatures[item.kind])
            answers[item.name] = _answer(item.kind, item.option_ids, item.criteria, probabilities)
            if include_debug:
                debug[item.name] = {
                    "option_ids": list(item.option_ids),
                    "option_logits": values,
                    "answer_token_ids": list(item.answer_token_ids),
                    "input_tokens": len(item.input_ids),
                    "suffix_tokens": len(item.suffix_ids),
                    "prompt_sha256": item.prompt_sha256,
                }

        response: dict[str, Any] = {
            "model": MODEL_ALIAS,
            "temperature": dict(self.temperatures),
            "answers": answers,
            "usage": {
                "input_tokens": len(prefix) + suffix_tokens,
                "prefix_tokens": len(prefix),
                "suffix_tokens": suffix_tokens,
                "output_tokens": 0,
            },
        }
        if include_debug:
            response["_debug"] = {
                "questions": debug,
                "temperature": dict(self.temperatures),
                "probability_status": PROBABILITY_STATUS,
                "model_source": self.model_source,
                "model_revision": self.revision,
                "load_seconds": self._load_seconds,
                "evaluation_seconds": elapsed,
                "prefill_seconds": timing["prefill_seconds"],
                "suffix_seconds": timing["suffix_seconds"],
                "readout": (
                    "one state prefill, then last-position logits of each question "
                    "suffix restricted to declared single-token answer slots"
                ),
            }
        return response


DEMO_REQUEST: dict[str, Any] = {
    "model": "jev-latest",
    "state": "Help! My payouts have been failing for 3 days.",
    "questions": {
        "is_urgent": {
            "type": "noul",
            "instructions": "Does this message convey urgency?",
            "criteria": {
                "true": "Explicitly time-sensitive or prolonged impact",
                "false": "No urgency or ongoing impact expressed",
            },
        },
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "billing": "Payments, invoicing, refunds, or payouts",
                "technical": "Bugs, outages, or integrations",
                "sales": "Pricing, upgrades, or new accounts",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
    },
}


def _read_request(path: str | None) -> dict[str, Any]:
    if path is None:
        return DEMO_REQUEST
    if path == "-":
        import sys

        return json.load(sys.stdin)
    with Path(path).open() as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="System One request JSON; use - for stdin (default: built-in demo)")
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
    parser.add_argument("--debug", action="store_true", help="Include raw option logits and provenance")
    args = parser.parse_args()

    engine = JevLikeEngine(
        args.model,
        revision=args.revision,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        cache_limit_mib=args.cache_limit_mib,
    )
    response = engine.evaluate(_read_request(args.input), include_debug=args.debug)
    print(json.dumps(response, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
