# Build Your Jev

English · [中文](README.md)

A local Jev-style semantic judge for Apple Silicon, built on the public Qwen3.5-4B checkpoint.

The input is one `state` plus a set of typed questions. The output is a `choice`, `score`, or `noul` probability. The model reads the material once and reads the logits of the allowed answers directly. It does not write an explanation.

```text
state + typed questions
        ↓
prefill the shared state prefix once
        ↓
append each question as a short suffix
        ↓
read the next-token logits of letters A/B/C…
        ↓
temperature scaling and softmax, per question type
        ↓
Choice / Score / Noul JSON
```

The weights stay frozen. The change is at inference and readout: the language model's next-token distribution is reduced to a set of answers declared in advance.

## Base model

| Item | Value |
| --- | --- |
| Base model | `Qwen/Qwen3.5-4B` (2026) |
| Local checkpoint | `mlx-community/Qwen3.5-4B-OptiQ-4bit` |
| Revision | `6cb5bdfd0bf15f484881fb9f1ab6d7c840fddde9` |
| Quantization | MLX OptiQ mixed 4/8-bit |
| Weights | 3,269,669,552 bytes, about 3.3 GB |
| Context | 262,144 tokens |
| Runtime | Apple Silicon GPU, MLX |
| License | Apache-2.0 |

A 4B model can stay resident on a laptop, handles Chinese, has a long context, and is small enough to serve after quantization. The source, revision, and weight SHA-256 are pinned in `jev_model.lock.json`.

MLX-LM is pinned to commit `a63e24c389382619eb6d9af656e3b46024be217a`, which includes the Qwen3.5 recurrent q/k RMSNorm implementation.

## Install

```bash
uv sync
uv run hf download mlx-community/Qwen3.5-4B-OptiQ-4bit \
  chat_template.jinja config.json generation_config.json kv_config.json \
  model.safetensors model.safetensors.index.json optiq_metadata.json \
  tokenizer.json tokenizer_config.json \
  --revision 6cb5bdfd0bf15f484881fb9f1ab6d7c840fddde9 \
  --local-dir models/qwen3.5-4b-optiq-4bit
```

Image input also needs Ollama and the OCR model:

```bash
ollama pull deepseek-ocr:3b
```

## Three question types, one letter each

The raw model output used here is the logit of the next token. Every legal answer is one fixed, single-token uppercase letter:

- `choice`: the first option is `A`, the second is `B`, through `P`, at most 16 options
- `score`: level 0 is `A`, level 1 is `B`, at most 10 levels
- `noul`: `A = true`, `B = false`

A choice question reaches the model in this form:

```text
A = billing: payments, refunds, and invoices
B = technical: outages, APIs, and program errors
```

Each letter has to pass four checks: it encodes as exactly one token; decoding that token returns the same letter; appending it to the prompt does not let BPE merge across the boundary; and every letter has a distinct token id. The logit then lines up with the option.

Question ids are JSON keys in the response. They are not sent to the model.

## The state is read once

Every question is rendered with the same shape:

```text
system instruction
{"state": ...}
Question:
{"instructions": ..., "options": ...}
```

The tokens before `Question:` are identical for every question. The program renders each full prompt, finds the common prefix, and, if the tokenizer merged tokens at the boundary, walks the prefix back by at most 8 tokens until the prefixes match exactly. It then:

1. prefills the shared prefix once;
2. keeps the model cache;
3. branches that cache and scores each question's short suffix.

Up to 16 suffixes run together, and one request holds up to 128 questions. Chunking keeps a long state cache from being copied once per question. A 2,000-token document plus 100 short questions costs about one read of the document plus 100 short suffixes. The questions share a read-only cache and do not see one another's answers.

## Read the logits directly

After a suffix, the readout takes the hidden state of that question's last real token. This checkpoint ties the embeddings, so the input embedding matrix is used as a linear map to vocabulary logits. Only the legal letter positions are kept:

```text
full vocabulary logits → [logit_A, logit_B, logit_C, ...]
```

There is no sampling and no JSON generation. The answer is fixed when those logits are read.

## From logits to probabilities

Legal candidates go through a stable softmax:

```text
p_i = exp(logit_i / T - max_logit) / Σ exp(logit_j / T - max_logit)
```

Subtracting the maximum avoids overflow in the exponential. Temperature `T` changes how sharp the distribution is. The option with the largest logit stays the same.

### Noul

Noul returns the probability of "yes":

```json
{ "type": "noul", "noul": 0.91 }
```

A rule can use it directly:

```python
if answer["noul"] >= 0.8:
    go_to_spam()
```

### Choice

Choice returns the option with the highest probability and keeps the full distribution:

```json
{
  "type": "choice",
  "choice": "billing",
  "probabilities": { "billing": 0.82, "technical": 0.18 },
  "confidence": 0.64
}
```

### Score

An ordered scale also returns the expected level index:

```text
score = Σ (level index × probability of that level)
```

The probabilities `[0.1, 0.6, 0.3]` produce a continuous score of `1.2`. That number is the expected position on the three-level scale.

### Confidence

For Choice and Score:

```text
confidence = (n × p_max - 1) / (n - 1)
```

A uniform distribution maps to 0. A probability of 1 on one option maps to 1. This describes how concentrated the distribution is.

## Fit the temperature on 1,000 questions

Raw softmax values still need labeled data. The suite has 10 scenarios of 100 questions: 400 Noul, 400 Choice, and 200 Score.

Temperature is fit by minimizing negative log-likelihood on the temperature-1 results, checked by leaving out one whole scenario at a time. The deployed values are:

| Type | Temperature | Held-out scenarios |
| --- | --- | --- |
| Noul | 0.6882 | negative log-likelihood 0.149 → 0.139; confidence error 7.0% → 5.6% |
| Choice | 1.0 | about 1.03 on the fitting set; held-out scenarios did not improve, so it stays at 1 |
| Score | 2.9052 | negative log-likelihood 1.171 → 1.000; confidence error 26.1% → 16.5% |

Calibration moves the probability scale. The winning Choice or Score option stays the same, and the Noul 0.5 boundary stays the same. The fit is in `examples/calibration.json`. A new workload should be refit on its own labels.

The same 1,000 questions, rerun after calibration:

- Overall 832/1000, 83.2%
- Noul 380/400, 95.0%
- Choice 362/400, 90.5%
- Score exact level 90/200, 45.0%
- Score within one level 176/200, 88.0%
- About 194 seconds

Exact accuracy follows the winning option, so temperature scaling changes how close the reported probabilities are to the measured accuracy.

On the 400 yes/no questions, the calibrated Noul gates are:

| Gate | Called "yes" | False positives | Precision |
| --- | --- | --- | --- |
| ≥ 0.5 | 176 | 5 | 97.2% |
| ≥ 0.8 | 160 | 2 | 98.8% |
| ≥ 0.9 | 151 | 1 | 99.3% |

## Images become text first

The OptiQ checkpoint makes the text decision. An image is read by `deepseek-ocr:3b`, written to `state.image_text`, and then scored by the decision model. OCR answers what the picture says. Qwen answers what to decide from that text. The decision path does not use an embedding model.

## API

```bash
uv run python jev_server.py --host 127.0.0.1 --port 8081 --eager
```

- `POST /v1/systemone`: Choice, Score, and Noul
- `POST /v1/ocr`: OCR only
- `POST /youtube`: a decision from the public title and channel
- `GET /health`
- `GET /v1/models`

Request:

```json
{
  "state": "The material to judge",
  "questions": {
    "suspicious": {
      "type": "noul",
      "instructions": "Is this message suspicious?"
    }
  }
}
```

```bash
curl http://127.0.0.1:8081/v1/systemone \
  -H 'Content-Type: application/json' \
  --data @examples/jev_request.json
```

If `JEV_API_KEY` is set, requests need `Authorization: Bearer <key>`. `--temperature 1` returns the raw softmax for every question type.

Open <http://127.0.0.1:8081/>, click a YouTube video, and read the decision made from its title and channel.

## Code

- `jev_like.py`: question types, shared prefix, cache branches, logits, softmax, calibration
- `jev_server.py`: HTTP API
- `jev_preprocess.py`: places OCR text on the state
- `jev_ocr.py`: calls `deepseek-ocr:3b` and cleans its output
- `examples/scenario_suite.py`: the 10 × 100 suite
- `examples/run_scenario_suite.py`: runs and grades the suite
- `examples/calibration.json`: the fitted temperatures
- `tests/test_jev_like.py`: contract, numerics, OCR, and the shared prefix

```bash
uv run python -m unittest tests.test_jev_like
uv run python examples/run_scenario_suite.py
```

Compare the shared prefix with a full forward pass per question:

```bash
JEV_RUN_MODEL=1 uv run python -m unittest tests.test_jev_like
```
