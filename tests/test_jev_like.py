from __future__ import annotations

import base64
import math
import os
import unittest

from jev_like import (
    FITTED_TEMPERATURE,
    LOCAL_MODEL,
    MAX_QUESTIONS,
    JevLikeEngine,
    RequestError,
    _answer,
    distribution_confidence,
    normalize_questions,
    shared_token_split,
    stable_softmax,
)
from jev_ocr import clean_ocr_text, image_bytes, merge_image_text
from jev_preprocess import prepare_request
from jev_server import youtube_video_id

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class ProbabilityTests(unittest.TestCase):
    def test_stable_softmax_handles_large_logits(self) -> None:
        probabilities = stable_softmax([10_000.0, 9_999.0, -10_000.0])
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertGreater(probabilities[0], probabilities[1])
        self.assertEqual(probabilities[2], 0.0)

    def test_default_temperature_is_fitted_per_question_type(self) -> None:
        engine = JevLikeEngine()
        self.assertEqual(engine.temperatures, dict(FITTED_TEMPERATURE))
        self.assertLess(engine.temperatures["noul"], 1.0)
        self.assertEqual(engine.temperatures["choice"], 1.0)
        self.assertGreater(engine.temperatures["score"], 1.0)

    def test_temperature_override_applies_to_every_question_type(self) -> None:
        engine = JevLikeEngine(temperature=1.0)
        self.assertEqual(set(engine.temperatures.values()), {1.0})

    def test_temperature_changes_confidence_not_argmax(self) -> None:
        cold = stable_softmax([3.0, 1.0], temperature=0.5)
        warm = stable_softmax([3.0, 1.0], temperature=2.0)
        self.assertEqual(cold.index(max(cold)), warm.index(max(warm)))
        self.assertGreater(max(cold), max(warm))

    def test_invalid_temperature_is_rejected(self) -> None:
        for value in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(RequestError):
                stable_softmax([0.0, 1.0], value)

    def test_distribution_confidence_has_chance_and_certainty_anchors(self) -> None:
        self.assertAlmostEqual(distribution_confidence([0.5, 0.5]), 0.0)
        self.assertAlmostEqual(distribution_confidence([1 / 3, 1 / 3, 1 / 3]), 0.0)
        self.assertAlmostEqual(distribution_confidence([1.0, 0.0, 0.0]), 1.0)


class ContractTests(unittest.TestCase):
    def test_all_three_question_types_normalize(self) -> None:
        questions = normalize_questions(
            {
                "route": {
                    "type": "choice",
                    "instructions": "Where?",
                    "criteria": {"billing": "Money", "support": "Help"},
                },
                "severity": {
                    "type": "score",
                    "instructions": "How severe?",
                    "criteria": ["low", "medium", "high"],
                },
                "urgent": {
                    "type": "noul",
                    "instructions": "Urgent?",
                    "criteria": {"true": "Urgent", "false": "Not urgent"},
                },
            }
        )
        self.assertEqual([question["kind"] for question in questions], ["choice", "score", "noul"])
        self.assertEqual(questions[0]["option_ids"], ("billing", "support"))
        self.assertEqual(questions[1]["option_ids"], ("0", "1", "2"))
        self.assertEqual(questions[2]["option_ids"], ("true", "false"))

    def test_choice_answer_is_typed(self) -> None:
        answer = _answer("choice", ("billing", "technical"), {}, [0.8, 0.2])
        self.assertEqual(answer["choice"], "billing")
        self.assertAlmostEqual(answer["confidence"], 0.6)
        self.assertEqual(answer["probabilities"], {"billing": 0.8, "technical": 0.2})

    def test_score_returns_probability_weighted_value(self) -> None:
        answer = _answer("score", ("0", "1", "2"), ["low", "mid", "high"], [0.1, 0.2, 0.7])
        self.assertAlmostEqual(answer["score"], 1.6)
        self.assertEqual(answer["legend"], {"0": "low", "1": "mid", "2": "high"})

    def test_noul_returns_true_probability(self) -> None:
        answer = _answer("noul", ("true", "false"), {}, [0.73, 0.27])
        self.assertEqual(answer, {"type": "noul", "noul": 0.73})

    def test_invalid_cardinalities_are_rejected(self) -> None:
        with self.assertRaises(RequestError):
            normalize_questions({"x": {"type": "choice", "criteria": {"only": None}}})
        with self.assertRaises(RequestError):
            normalize_questions({"x": {"type": "score", "criteria": ["only"]}})
        with self.assertRaises(RequestError):
            normalize_questions(
                {
                    str(index): {"type": "noul", "instructions": "True?"}
                    for index in range(MAX_QUESTIONS + 1)
                }
            )

    def test_youtube_links_resolve_to_one_video(self) -> None:
        self.assertEqual(youtube_video_id("https://youtu.be/dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(
            youtube_video_id("https://www.youtube.com/shorts/_uQrJ0TkZlc?feature=share"),
            "_uQrJ0TkZlc",
        )
        with self.assertRaises(RequestError):
            youtube_video_id("https://example.com/watch?v=dQw4w9WgXcQ")


class SharedPrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (LOCAL_MODEL / "tokenizer.json").is_file():
            raise unittest.SkipTest("local tokenizer is not downloaded")
        from mlx_lm.utils import load_tokenizer

        cls.tokenizer = load_tokenizer(str(LOCAL_MODEL))

    def test_state_is_one_prefix_and_each_question_is_a_suffix(self) -> None:
        from jev_like import question_user_content

        state = {"ticket": "My payouts have been failing for 3 days.", "locale": "zh"}
        contents = [
            question_user_content(state, "Which team?", [{"letter": "A", "value": "billing", "description": "Money"}]),
            question_user_content(
                state,
                "Is this urgent?",
                [
                    {"letter": "A", "value": "true", "description": "Yes"},
                    {"letter": "B", "value": "false", "description": "No"},
                ],
            ),
        ]
        prefix, suffixes, prompts, full_ids = shared_token_split(self.tokenizer, state, contents)
        self.assertGreater(len(prefix), 10)
        self.assertEqual(len(suffixes), 2)
        for suffix, prompt, full in zip(suffixes, prompts, full_ids):
            self.assertGreater(len(suffix), 0)
            self.assertEqual(list(prefix) + list(suffix), full)
            self.assertLess(len(prefix), len(full))
            self.assertIn("payouts", prompt)
        self.assertNotEqual(suffixes[0], suffixes[1])


class ImageTests(unittest.TestCase):
    def test_ocr_cleanup_keeps_the_recognized_line(self) -> None:
        self.assertEqual(clean_ocr_text("\\*<|im_end|>Invoice 42 due Friday"), "Invoice 42 due Friday")
        self.assertEqual(
            clean_ocr_text(" by \n<|user|>user\nInvoice 42 due Friday"),
            "Invoice 42 due Friday",
        )
        self.assertEqual(clean_ocr_text("<|ref|>Total 8<|/ref|><|det|>[[1,2]]<|/det|>"), "Total 8")

    def test_image_text_is_added_to_state_without_calling_a_model(self) -> None:
        encoded = "data:image/png;base64," + base64.b64encode(_PNG).decode()
        self.assertEqual(image_bytes(encoded), _PNG)
        merged = merge_image_text("existing note", "Invoice 42")
        self.assertEqual(merged, {"context": "existing note", "image_text": "Invoice 42"})
        prepared, meta = prepare_request(
            {
                "image": encoded,
                "state": "existing note",
                "questions": {"urgent": {"type": "noul"}},
            },
            read_image=lambda image, prompt: "Invoice 42",
        )
        self.assertNotIn("image", prepared)
        self.assertEqual(prepared["state"]["image_text"], "Invoice 42")
        self.assertEqual(meta["image_text"], "Invoice 42")
        self.assertEqual(prepared["questions"]["urgent"]["type"], "noul")


@unittest.skipUnless(os.environ.get("JEV_RUN_MODEL") == "1", "set JEV_RUN_MODEL=1 to score the local model")
class SharedPrefixAgreesTests(unittest.TestCase):
    def test_shared_prefix_matches_independent_prompts(self) -> None:
        from jev_like import DEMO_REQUEST, JevLikeEngine

        engine = JevLikeEngine()
        engine.load()
        questions = normalize_questions(DEMO_REQUEST["questions"])
        prefix, prepared = engine._prepare_batch(DEMO_REQUEST["state"], questions)
        shared, _ = engine._option_logits(prefix, prepared)
        fresh = engine._independent_option_logits(prepared)
        for left, right in zip(shared, fresh):
            self.assertEqual(left.index(max(left)), right.index(max(right)))
            for shared_logit, fresh_logit in zip(left, right):
                # The checkpoint computes in bfloat16. One ulp around these
                # logits is 0.125, so the two paths can differ by that rounding.
                self.assertLess(abs(shared_logit - fresh_logit), 0.25)


if __name__ == "__main__":
    unittest.main()
