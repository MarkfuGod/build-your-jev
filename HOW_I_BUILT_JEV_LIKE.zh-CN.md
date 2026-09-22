# Build Your Jev

在 Apple Silicon 上，用公开的 Qwen3.5-4B 做一个本地 Jev 式语义判断器。

输入是一份 `state` 和一组带类型的问题。输出是 `choice`、`score` 或 `noul` 的概率。模型一次读完材料，直接读取候选答案的 logits，不生成解释文字。

```text
state + typed questions
        ↓
共享 state 前缀只计算一次
        ↓
每道题补自己的短后缀
        ↓
读取候选字母 A/B/C… 的下一 token logits
        ↓
按题型做温度校准和 softmax
        ↓
Choice / Score / Noul JSON
```

权重保持冻结。改造发生在推理和输出层：把语言模型的下一 token 分布，收成一组事先声明好的答案。

## 基础模型

| 项目 | 值 |
| --- | --- |
| 基础模型 | `Qwen/Qwen3.5-4B`（2026） |
| 本地检查点 | `mlx-community/Qwen3.5-4B-OptiQ-4bit` |
| 版本 | `6cb5bdfd0bf15f484881fb9f1ab6d7c840fddde9` |
| 量化 | MLX OptiQ mixed 4/8-bit |
| 权重 | 3,269,669,552 bytes，约 3.3 GB |
| 上下文 | 262,144 tokens |
| 运行 | Apple Silicon GPU，MLX |
| 协议 | Apache-2.0 |

4B 可以常驻本机，中文够用，上下文够长，量化后的体积也适合做成服务。来源、版本和权重 SHA-256 写在 `jev_model.lock.json`。

MLX-LM 固定在提交 `a63e24c389382619eb6d9af656e3b46024be217a`，其中包含 Qwen3.5 recurrent q/k RMSNorm 的实现。

## 安装

```bash
uv sync
uv run hf download mlx-community/Qwen3.5-4B-OptiQ-4bit \
  chat_template.jinja config.json generation_config.json kv_config.json \
  model.safetensors model.safetensors.index.json optiq_metadata.json \
  tokenizer.json tokenizer_config.json \
  --revision 6cb5bdfd0bf15f484881fb9f1ab6d7c840fddde9 \
  --local-dir models/qwen3.5-4b-optiq-4bit
```

读图还需要 Ollama 和 OCR 模型：

```bash
ollama pull deepseek-ocr:3b
```

## 三种题，都变成一个字母

语言模型直接给出的是下一个 token 的 logit。每个合法答案对应一个确定的单 token 大写字母：

- `choice`：第一个选项是 `A`，第二个是 `B`，依次到 `P`，最多 16 个选项
- `score`：第 0 档是 `A`，第 1 档是 `B`，最多 10 档
- `noul`：`A = true`，`B = false`

一道选择题在进入模型前会变成：

```text
A = billing：付款、退款和账单
B = technical：故障、接口和程序错误
```

每个字母都要满足四件事：单独编码后正好一个 token；编码再解码仍是同一个字母；接到 prompt 末尾时不会被 BPE 合并；不同字母的 token id 互不相同。这样读到的 logit 和选项一一对应。

问题 ID 只作为返回 JSON 的 key，不送进模型。

## State 只读一次

每道题都按同一结构渲染：

```text
system instruction
{"state": ...}
Question:
{"instructions": ..., "options": ...}
```

`Question:` 之前的 token 对所有题相同。程序先渲染完整 prompt，找出共同前缀；如果分词器在边界处合并了 token，就最多向前退 8 个 token，直到前缀能精确对齐。然后：

1. 共享前缀只做一次 prefill；
2. 留下模型 cache；
3. 从这份 cache 分出分支，计算每道题自己的短后缀。

一次最多并行 16 个后缀，一个请求最多 128 道题。分块是为了控制长 state cache 的复制。一份 2,000-token 材料加 100 道短问题，计算量接近「材料一次，加上 100 段短问题」。各题共享只读 cache，彼此看不到对方的答案。

## 直接读 logits

后缀算完后，取该题最后一个真实 token 的 hidden state。这个检查点使用 tied embeddings，所以用输入 embedding 矩阵做线性投影，得到词表 logits，再只保留合法字母的位置：

```text
完整词表 logits → [logit_A, logit_B, logit_C, ...]
```

这里没有采样，也没有生成 JSON。答案在读出 logits 时就已经确定。

## 从 logits 到概率

合法候选用稳定 softmax：

```text
p_i = exp(logit_i / T - max_logit) / Σ exp(logit_j / T - max_logit)
```

先减去最大值，避免指数溢出。温度 `T` 改变分布的尖锐程度，最大 logit 对应的选项保持不变。

### Noul

返回「是」的概率：

```json
{ "type": "noul", "noul": 0.91 }
```

业务规则可以直接写成：

```python
if answer["noul"] >= 0.8:
    go_to_spam()
```

### Choice

返回概率最大的选项，并保留全部分布：

```json
{
  "type": "choice",
  "choice": "billing",
  "probabilities": { "billing": 0.82, "technical": 0.18 },
  "confidence": 0.64
}
```

### Score

有序档位同时返回期望位置：

```text
score = Σ（档位下标 × 该档概率）
```

`[0.1, 0.6, 0.3]` 的连续分数是 `1.2`。这是三档概率的期望位置。

### Confidence

Choice 和 Score 的 `confidence` 是：

```text
confidence = (n × p_max - 1) / (n - 1)
```

均匀分布对应 0，某个选项概率为 1 对应 1。它描述分布有多集中。

## 用 1,000 题校准温度

原始 softmax 还要放到标注数据上对齐。测试集是 10 个情景、每个 100 题：400 道 Noul、400 道 Choice、200 道 Score。

在温度 1 的结果上最小化负对数似然，并用「每次留出一个完整情景」检查。部署值：

| 题型 | 温度 | 留出情景上的结果 |
| --- | --- | --- |
| Noul | 0.6882 | 负对数似然 0.149 → 0.139，置信度误差 7.0% → 5.6% |
| Choice | 1.0 | 样本内约 1.03，留出情景没有更好，保持 1 |
| Score | 2.9052 | 负对数似然 1.171 → 1.000，置信度误差 26.1% → 16.5% |

校准移动的是概率刻度。Choice 和 Score 的最大项不变，Noul 的 0.5 分界也不变。数据在 `examples/calibration.json`。换一套真实业务时，用那套标注重新拟合。

校准后重跑同一套 1,000 题：

- 总体 832/1000，83.2%
- Noul 380/400，95.0%
- Choice 362/400，90.5%
- Score 精确命中 90/200，45.0%
- Score 相邻一档以内 176/200，88.0%
- 用时约 194 秒

分类正确率由最大项决定，所以温度校准改善的是概率和实测正确率的接近程度。

在这 400 道是非题上，校准后的 Noul 门槛是：

| 门槛 | 判成「是」 | 误报 | 精确率 |
| --- | --- | --- | --- |
| ≥ 0.5 | 176 | 5 | 97.2% |
| ≥ 0.8 | 160 | 2 | 98.8% |
| ≥ 0.9 | 151 | 1 | 99.3% |

## 图片先变成文字

当前 OptiQ 检查点负责文字判断。图片先由 `deepseek-ocr:3b` 读成文字，写入 `state.image_text`，再交给判断模型。OCR 回答「图里写了什么」，Qwen 回答「据此如何判断」。决策路径不使用 embedding。

## 接口

```bash
uv run python jev_server.py --host 127.0.0.1 --port 8081 --eager
```

- `POST /v1/systemone`：Choice、Score、Noul
- `POST /v1/ocr`：只做 OCR
- `POST /youtube`：用公开标题和频道做判断
- `GET /health`
- `GET /v1/models`

请求：

```json
{
  "state": "这里是一份材料",
  "questions": {
    "suspicious": {
      "type": "noul",
      "instructions": "这条消息是否可疑？"
    }
  }
}
```

```bash
curl http://127.0.0.1:8081/v1/systemone \
  -H 'Content-Type: application/json' \
  --data @examples/jev_request.json
```

设置 `JEV_API_KEY` 后，请求需要 `Authorization: Bearer <key>`。`--temperature 1` 会让三种题都回到未校准的原始 softmax。

打开 <http://127.0.0.1:8081/> 可以点一个 YouTube 视频，看标题和频道对应的判断。

## 代码

- `jev_like.py`：题型、共享前缀、cache 分支、logit、softmax、校准
- `jev_server.py`：HTTP API
- `jev_preprocess.py`：把 OCR 文本放进 state
- `jev_ocr.py`：调用并清理 `deepseek-ocr:3b`
- `examples/scenario_suite.py`：10 × 100 测试集
- `examples/run_scenario_suite.py`：跑分
- `examples/calibration.json`：温度拟合
- `tests/test_jev_like.py`：契约、数值、OCR 和共享前缀

```bash
uv run python -m unittest tests.test_jev_like
uv run python examples/run_scenario_suite.py
```

共享前缀和逐题完整计算的模型级对照：

```bash
JEV_RUN_MODEL=1 uv run python -m unittest tests.test_jev_like
```
