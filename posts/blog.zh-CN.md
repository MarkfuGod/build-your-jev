# 不生成文字，直接读 logits：我是怎样把 Qwen3.5-4B 做成 Jev 式语义判断器的

大语言模型最常见的使用方式是生成文字：给它一个 prompt，让它继续写，最后再从回复里解析 JSON。

但很多自动化任务根本不需要文本生成。

比如：

- 这条消息是否可疑？
- 这个工单应该分给哪个部门？
- 这件事属于低、中、高哪个风险档位？
- 当“可疑”的概率超过 80% 时，是否应该进入垃圾箱？

这些任务的答案空间在调用模型之前就已经确定了。真正需要的是语义判断和概率，而不是一段自然语言。

基于这个想法，我做了 **Build Your Jev**：一个运行在 Apple Silicon 上的本地 Jev 式语义判断器。它接收一份 `state` 和一组 typed questions，直接返回 `Noul`、`Choice` 或 `Score` 的结构化概率。

这篇文章记录整个实现过程。

## 目标：把语言模型变成判断器

我想要的接口很简单：

```json
{
  "state": "用户收到一条要求提供银行卡密码的短信",
  "questions": {
    "suspicious": {
      "type": "noul",
      "instructions": "这条短信是否可疑？"
    }
  }
}
```

模型不需要回答“这条短信可能是诈骗，因为……”，只要返回：

```json
{
  "type": "noul",
  "noul": 0.91
}
```

业务程序随后可以使用普通规则：

```python
if answer["noul"] >= 0.8:
    go_to_spam()
```

为了实现这一点，我没有重新训练一个 4B 模型。基础权重保持冻结，改造集中在三个地方：

1. 如何把不同类型的答案变成模型可以稳定打分的 token；
2. 如何让很多问题共享同一份材料的计算结果；
3. 如何直接读取候选答案的 logits，而不是生成文字。

## 为什么选择 Qwen3.5-4B

基础模型使用 `Qwen/Qwen3.5-4B`，本地运行的是 `mlx-community/Qwen3.5-4B-OptiQ-4bit`。

这个选择主要考虑四点：

- 4B 规模可以在 Apple Silicon 上常驻；
- 中文语义能力足够覆盖日常自动化判断；
- 262,144-token 上下文可以容纳较长的 state；
- OptiQ mixed 4/8-bit 权重约 3.3GB，适合本地服务。

模型 revision 和权重 SHA-256 都被锁定。这样以后重新下载时，可以确认实验使用的是同一份权重。

推理框架使用 MLX。MLX-LM 固定在一个包含 Qwen3.5 recurrent q/k RMSNorm 实现的版本，避免不同实现造成 logits 偏差。

## 第一步：把答案映射成单 token 字母

语言模型在每个位置都会输出“下一个 token”的完整词表 logits。

如果直接让模型生成业务选项，可能遇到大小写、空格、解释文字和 JSON 格式等问题。我的做法是把所有合法答案先映射成大写字母：

- Choice：第一项映射到 `A`，第二项映射到 `B`，依次到 `P`
- Score：第 0 档映射到 `A`，第 1 档映射到 `B`
- Noul：`A = true`，`B = false`

例如一个工单路由问题：

```json
{
  "type": "choice",
  "instructions": "应该交给哪个部门？",
  "criteria": {
    "billing": "付款、退款和账单",
    "technical": "故障、接口和程序错误"
  }
}
```

进入 prompt 后，选项会被写成：

```text
A = billing：付款、退款和账单
B = technical：故障、接口和程序错误
```

这里有一个容易忽略的细节：字母必须真的是稳定的单 token。

代码会检查：

1. `A` 单独编码后是否正好得到一个 token；
2. 这个 token 解码后是否仍然是 `A`；
3. 把 `A` 接在 prompt 末尾时，分词器是否会跨边界合并；
4. 不同字母是否拥有不同 token id。

只有全部通过，才能保证后面读取的 `logit_A` 真的代表业务选项 A。

问题 ID 不会送进模型。它只是响应 JSON 的 key，避免随机的业务 ID 影响语义判断。

## 第二步：让 state 只计算一次

如果一份 2,000-token 的邮件后面有 100 道问题，最直接的写法是构造 100 个完整 prompt。

这样模型会重复读取同一封邮件 100 次。

我把 prompt 拆成两部分：

```text
共享部分：
system instruction
{"state": ...}
Question:

每题独立部分：
{"instructions": ..., "options": ...}
```

程序先渲染所有完整 prompt，再在 token 层找出精确相同的前缀。之所以必须在 token 层检查，是因为 BPE tokenizer 可能在 state 和 question 的边界处发生合并。

如果发现边界 token 不一致，前缀最多向前缩短 8 个 token，直到：

```text
prefix_tokens + suffix_tokens == full_prompt_tokens
```

然后共享前缀只进行一次 prefill，得到模型 cache。每道题从这份 cache 分出自己的分支，只计算短后缀。

为了控制 cache 复制的峰值，每批最多处理 16 个问题后缀；一个请求最多接收 128 道题。

这样计算量从：

```text
100 ×（state + question）
```

变成更接近：

```text
state 一次 + 100 个 question suffix
```

每道题共享的是只读 state cache。它们不会看到其他题的答案，也不会按题目顺序互相影响。

## 第三步：跳过文本生成，直接读取候选 logits

每个问题后缀完成 forward 后，我取该题最后一个真实 token 对应的 hidden state。

这个检查点使用 tied embeddings，所以可以直接用输入 embedding 矩阵的线性投影得到完整词表 logits：

```text
last hidden state
        ↓
vocabulary projection
        ↓
完整词表 logits
```

接下来不做采样，只收集允许的字母位置：

```text
[logit_A, logit_B, logit_C, ...]
```

到这里，模型已经完成判断。

没有生成 token，没有自然语言解释，没有 JSON repair，也没有“生成后再解析”的第二套逻辑。

## 第四步：把 logits 变成概率

候选 logits 使用稳定 softmax：

```text
p_i = exp(logit_i / T - max_logit) / Σ exp(logit_j / T - max_logit)
```

减去最大 logit 可以防止指数计算溢出。这里不需要随意 clamp logits，因为 clamp 会改变候选之间的相对距离。

温度 `T` 控制概率分布的尖锐程度，但不会改变最大 logit 对应的选项。

## 三种输出类型

### Noul：二元条件

Noul 只返回 `true` 的概率：

```json
{
  "type": "noul",
  "noul": 0.91
}
```

它最适合进入 `if`：

```python
if suspicious_probability >= 0.8:
    go_to_spam()
```

### Choice：多个动作或分类

Choice 返回获胜选项和完整分布：

```json
{
  "type": "choice",
  "choice": "billing",
  "probabilities": {
    "billing": 0.82,
    "technical": 0.18
  },
  "confidence": 0.64
}
```

当系统需要在“放行、垃圾箱、人工复核”之间选择时，Choice 比多个互相独立的 if 更自然。

### Score：有序档位

Score 的选项带有顺序。除了每一档的概率，还返回档位下标的期望值：

```text
score = Σ（档位下标 × 该档概率）
```

如果低、中、高三档的概率为 `[0.1, 0.6, 0.3]`：

```text
score = 0 × 0.1 + 1 × 0.6 + 2 × 0.3 = 1.2
```

这里的 1.2 是概率分布在量表上的期望位置。

## 第五步：分别校准三种题的温度

原始 softmax 的数值不能自动当作现实正确率。

我制作了 10 个情景、每个 100 题的测试集：

- 400 道 Noul
- 400 道 Choice
- 200 道 Score

先保存温度 1 的原始概率，再通过最小化负对数似然拟合温度。验证方式是每次留出一个完整情景，用另外九个情景拟合，再测试被留出的情景。

最终部署温度：

- Noul：`0.6882`
- Choice：`1.0`
- Score：`2.9052`

Choice 在全部样本上的最优值约为 1.03，但没有改善留出情景，所以保持 1。

Noul 需要更尖的分布，Score 则需要明显更平的分布。这也是为什么不能简单给所有题共用一个温度。

校准只改变概率刻度。Choice 和 Score 的最大项不变，Noul 的 0.5 分界也不变。

## 1,000 题的结果

移除 embedding、加入分题型校准后，我重新运行了整套测试：

- 总体正确：832/1000，83.2%
- Noul：380/400，95.0%
- Choice：362/400，90.5%
- Score 精确命中：90/200，45.0%
- Score 相邻一档以内：176/200，88.0%
- 完整运行时间：约 194 秒

Score 的 45% 看起来很低，但它衡量的是必须命中同一个档位。200 道题中，86 道只差相邻一档。

分析错误后发现，风险题最容易出现语义方向不一致。例如问题想问的是“如果误开卧室灯，后果有多严重”，标准答案标记的是错误动作的严重程度；模型有时回答的是“当前卧室灯已经关着，所以目前风险不高”。

这说明 Score 的量表必须同时写清楚：

- 评分对象是什么；
- 评分的是当前状态还是假设动作；
- 每个档位的业务含义是什么。

如果真正需要的是“是否满足条件”，应该使用 Noul，而不是把二元问题硬塞进三档 Score。

## 一个 80% 的 if 是否可用

在这 400 道 Noul 测试题上，校准后的门槛表现为：

- `p >= 0.5`：176 次触发，5 次误报，精确率 97.2%
- `p >= 0.8`：160 次触发，2 次误报，精确率 98.8%
- `p >= 0.9`：151 次触发，1 次误报，精确率 99.3%

所以 `p >= 0.8` 可以成为一个简单的自动化门槛，但它是这套测试分布上的结果。换成真实短信、邮件或工单后，需要重新收集标注并校准门槛。

## 图片输入：先 OCR，再判断

当前决策模型走文字路径。图片先通过 `deepseek-ocr:3b` 提取文字，再把结果写入 `state.image_text`。

```text
image
  ↓
DeepSeek OCR
  ↓
recognized text
  ↓
state
  ↓
typed decision
```

OCR 解决“图里写了什么”，判断模型解决“根据这些文字该做什么”。当前决策路径不需要 embedding model。

## 封装成一个本地接口

服务提供：

- `POST /v1/systemone`：执行 Noul、Choice 和 Score
- `POST /v1/ocr`：只做 OCR
- `GET /health`：检查服务状态
- `GET /v1/models`：查看当前模型

启动：

```bash
uv run python jev_server.py --host 127.0.0.1 --port 8081 --eager
```

调用：

```bash
curl http://127.0.0.1:8081/v1/systemone \
  -H 'Content-Type: application/json' \
  --data @examples/jev_request.json
```

服务会返回答案、分题型温度和 token 使用量。设置 `JEV_API_KEY` 后，可以要求 Bearer Token。

## 我从这个实验里学到的三件事

第一，很多 AI 自动化需求本质上是有限答案集合上的语义判断。生成自然语言只是其中一种实现方式，不一定是最合适的实现方式。

第二，共享 state 比单纯并发更重要。如果每道题都复制完整材料，即使请求并发，模型仍然在重复做大量相同计算。

第三，softmax 数字不等于真实置信度。概率要在自己的数据上检查；不同题型甚至可能需要完全不同的温度。

Build Your Jev 最终形成了一条很短的判断链：

```text
读一次材料 → 计算短问题 → 读取候选 logits → 校准概率 → 进入普通业务规则
```

对于路由、过滤、告警、审核和自动化条件判断，这种方式值得作为文本生成之外的另一种选择。

---

项目名称：**Build Your Jev**

发布时可在这里加入项目仓库链接。
