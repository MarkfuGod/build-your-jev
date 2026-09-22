"""Ten scenario categories with 100 graded questions each.

The decision request sent to the model contains only state, instructions, and
criteria. Expected answers stay here for scoring.
"""

from __future__ import annotations

from typing import Any

LEVELS = ("不急", "今天处理", "马上处理")
RISK = ("低", "中", "高")


def opts(*pairs: tuple[str, str]) -> dict[str, str]:
    return dict(pairs)


def pack(
    tag: str,
    noul_q: str,
    noul_yes: bool,
    flip_q: str,
    flip_yes: bool,
    choice_q: str,
    choice_gold: str,
    choice_opts: dict[str, str],
    choice2_q: str,
    choice2_gold: str,
    choice2_opts: dict[str, str],
    score_q: str,
    score_at: int,
    levels: tuple[str, ...] = LEVELS,
) -> dict[str, Any]:
    return {
        "tag": tag,
        "noul": (noul_q, noul_yes),
        "flip": (flip_q, flip_yes),
        "choice": (choice_q, choice_gold, choice_opts),
        "choice2": (choice2_q, choice2_gold, choice2_opts),
        "score": (score_q, score_at, levels),
    }


def items_from(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(facts) != 20:
        raise AssertionError(f"expected 20 facts, got {len(facts)}")
    items: list[dict[str, Any]] = []
    for fact in facts:
        tag = fact["tag"]
        if tag not in {"lookup", "judgment"}:
            raise AssertionError(tag)
        noul_q, noul_yes = fact["noul"]
        flip_q, flip_yes = fact["flip"]
        choice_q, choice_gold, choice_opts = fact["choice"]
        choice2_q, choice2_gold, choice2_opts = fact["choice2"]
        score_q, score_at, levels = fact["score"]
        for gold, options in ((choice_gold, choice_opts), (choice2_gold, choice2_opts)):
            if gold not in options or len(options) < 2:
                raise AssertionError(f"bad choice {gold} in {options}")
        if not 0 <= score_at < len(levels):
            raise AssertionError(score_q)
        items.append(
            {
                "tag": tag,
                "type": "noul",
                "instructions": noul_q,
                "expected": bool(noul_yes),
                "criteria": {"true": "概况或记录支持这个说法", "false": "概况或记录不支持这个说法"},
            }
        )
        items.append(
            {
                "tag": tag,
                "type": "noul",
                "instructions": flip_q,
                "expected": bool(flip_yes),
                "criteria": {"true": "概况或记录支持这个说法", "false": "概况或记录不支持这个说法"},
            }
        )
        items.append(
            {
                "tag": tag,
                "type": "choice",
                "instructions": choice_q,
                "expected": choice_gold,
                "criteria": choice_opts,
            }
        )
        items.append(
            {
                "tag": tag,
                "type": "choice",
                "instructions": choice2_q,
                "expected": choice2_gold,
                "criteria": choice2_opts,
            }
        )
        items.append(
            {
                "tag": tag,
                "type": "score",
                "instructions": score_q,
                "expected": score_at,
                "criteria": list(levels),
            }
        )
    if len(items) != 100:
        raise AssertionError(len(items))
    seen: set[str] = set()
    for index, item in enumerate(items, start=1):
        item["id"] = f"q{index:03d}"
        if item["instructions"] in seen:
            raise AssertionError(f"duplicate instruction: {item['instructions']}")
        seen.add(item["instructions"])
    return items


def scenario(sid: str, name: str, state: dict[str, Any], facts: list[dict[str, Any]]) -> dict[str, Any]:
    text = state["概况"]
    for needle in state.get("_must", []):
        if needle not in text:
            raise AssertionError(f"{sid} overview missing {needle}")
    public_state = {key: value for key, value in state.items() if key != "_must"}
    return {"id": sid, "name": name, "state": public_state, "items": items_from(facts)}


def request_for(case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    questions = {
        item["id"]: {
            "type": item["type"],
            "instructions": item["instructions"],
            "criteria": item["criteria"],
        }
        for item in case["items"]
    }
    gold = {
        item["id"]: {"type": item["type"], "tag": item["tag"], "expected": item["expected"], "instructions": item["instructions"]}
        for item in case["items"]
    }
    return {"state": case["state"], "questions": questions}, gold


TEAMS = opts(
    ("finance", "财务：付款、发票、账单、报销"),
    ("engineering", "工程：生产故障、部署、缺陷"),
    ("security", "安全：密码、卡号、钓鱼、账号盗用"),
    ("sales", "销售：新购、升级、续费谈判"),
)
YESNO = opts(("yes", "按规则应该做"), ("no", "按规则不应该做"), ("later", "先拖到下周再看"))


def support() -> dict[str, Any]:
    state = {
        "概况": (
            "工单 T-8841，客户林晓，企业版，年付 8600 元，12 天后续费。邮件渠道，已等待 30 小时。"
            "连续 3 笔付款失败，原因 card_declined，已自动重试 2 次，上限 3 次。生产报错数是 0。"
            "客户要求开发票，发票还没开。客户没有要求降级，也没有要求退款。账号状态正常。"
            "情绪焦急。人在上海。这是大客户。正文没有密码，也没有完整卡号。近 30 天只有这 1 张工单。"
            "财务现在在线。工程值班是赵晨。状态页还没有更新。"
            "规则：付款失败且生产报错为 0 时交给财务，不要交给工程。不得因付款失败自动降级。"
            "没有退款请求就不要走退款。大客户等待超过 24 小时，今天必须处理。还能再自动重试 1 次。"
            "正文出现密码或完整卡号才升级到安全组。状态页只在生产故障时更新，这次不用。"
        ),
        "记录": {
            "工单号": "T-8841",
            "套餐": "企业版",
            "年付元": 8600,
            "续费剩余天数": 12,
            "已等待小时": 30,
            "连续失败付款": 3,
            "已重试次数": 2,
            "重试上限": 3,
            "失败原因": "card_declined",
            "生产报错数": 0,
            "要求开发票": True,
            "发票已开": False,
            "要求降级": False,
            "要求退款": False,
            "账号状态": "正常",
            "情绪": "焦急",
            "城市": "上海",
            "大客户": True,
            "正文含密码": False,
            "正文含完整卡号": False,
            "近30天工单数": 1,
            "财务在线": True,
            "状态页已更新": False,
        },
        "_must": ["T-8841", "30 小时", "card_declined", "生产报错数是 0", "不要交给工程"],
    }
    facts = [
        pack("lookup", "已经发生付款失败了吗？", True, "这些付款都成功了吗？", False, "付款失败的原因是哪一类？", "card", opts(("card", "银行拒付 card_declined"), ("timeout", "网关超时"), ("stock", "库存不足")), "失败笔数落在哪一档？", "three", opts(("none", "没有失败"), ("one", "只有 1 笔"), ("three", "连续 3 笔")), "只看付款失败的积压，今天要不要处理？", 1),
        pack("lookup", "生产环境有报错吗？", False, "生产环境是正常的吗？", True, "按规则这张工单交给谁？", "finance", TEAMS, "需要把赵晨叫来处理生产故障吗？", "no", YESNO, "这次生产故障的紧急度是多少？", 0),
        pack("lookup", "客户在等一张还没开的发票吗？", True, "发票已经开好了吗？", False, "发票这件事应该怎么做？", "issue", opts(("issue", "开具尚未开出的发票"), ("skip", "客户没要发票"), ("void", "作废已经开出的发票")), "发票问题交给哪个团队？", "finance", TEAMS, "发票未开并且客户在等，排期怎么定？", 1),
        pack("judgment", "客户要求降级了吗？", False, "可以因为付款失败自动降级吗？", False, "账号套餐应该怎么处理？", "keep", opts(("keep", "保持企业版，不降级"), ("downgrade", "自动降到免费版"), ("cancel", "直接注销")), "降级会违反哪条规则？", "forbidden", opts(("forbidden", "规则禁止因付款失败自动降级"), ("required", "规则要求立刻降级"), ("optional", "规则没提降级")), "如果现在自动降级，风险有多高？", 2, RISK),
        pack("lookup", "客户要求退款了吗？", False, "应该直接进入退款流程吗？", False, "退款流程怎么处理？", "skip", opts(("skip", "没有退款请求，不要走退款"), ("full", "全额退回 8600 元"), ("partial", "退回最近一笔")), "退款交给谁才符合规则？", "nobody", opts(("nobody", "先不要交给任何人去退款"), ("finance", "财务立刻退款"), ("sales", "销售审批退款")), "退款这件事有多急？", 0),
        pack("lookup", "林晓是大客户吗？", True, "这个客户可以按普通个人用户排队吗？", False, "客户等级是哪一档？", "vip", opts(("vip", "大客户"), ("standard", "普通客户"), ("trial", "试用客户")), "大客户等待超过 24 小时后，规则要求哪天处理？", "today", opts(("today", "今天必须处理"), ("week", "本周任意一天"), ("next", "等到续费前一天")), "按大客户超时规则，紧急度是多少？", 1),
        pack("lookup", "这张工单已经等了超过 24 小时吗？", True, "这张工单是 1 小时内新开的吗？", False, "等待时间落在哪一档？", "over", opts(("fresh", "不到 1 小时"), ("same", "刚好 2 小时"), ("over", "已等待 30 小时，超过 24 小时")), "等待时间够不够触发今天处理？", "yes", YESNO, "只看等待时长，排期有多急？", 1),
        pack("judgment", "自动重试还有剩余次数吗？", True, "自动重试已经达到上限了吗？", False, "付款下一步应该怎么做？", "retry", opts(("retry", "已重试 2 次，上限 3 次，再自动试 1 次"), ("stop", "停止自动重试"), ("downgrade", "改为降级账号")), "重试之后如果仍然失败，再交给谁？", "finance", TEAMS, "再试一次付款的风险有多高？", 1, RISK),
        pack("lookup", "正文里有密码吗？", False, "需要因为正文内容升级到安全组吗？", False, "安全组现在要不要接手？", "no", opts(("no", "没有密码或完整卡号，不升级"), ("yes", "立刻交给安全组"), ("later", "等客户再回一封再升级")), "账号要不要马上冻结？", "no", YESNO, "这封正文的安全紧急度是多少？", 0),
        pack("lookup", "正文里有完整卡号吗？", False, "应该把卡号记进工单备注吗？", False, "支付凭证应该怎么记录？", "code", opts(("code", "只保留失败码 card_declined，不记卡号"), ("pan", "记下完整卡号"), ("photo", "让客户拍照银行卡")), "这件事属于安全事件吗？", "no", YESNO, "卡号泄露风险有多高？", 0, RISK),
        pack("lookup", "账号目前是正常状态吗？", True, "账号已经被封禁了吗？", False, "账号状态应该怎么处理？", "keep", opts(("keep", "保持正常，不要解封或封禁"), ("unlock", "帮客户解封"), ("lock", "立刻封禁")), "需要销售去谈解封吗？", "no", YESNO, "账号状态问题有多急？", 0),
        pack("lookup", "财务现在在线吗？", True, "财务今天不在，必须等到明天吗？", False, "现在可以把工单转给谁？", "finance", TEAMS, "转交前要不要先等一个工作日？", "no", YESNO, "因为财务在线，处理可以排到哪一档？", 1),
        pack("judgment", "状态页已经更新了这条记录吗？", False, "这次应该更新状态页吗？", False, "状态页怎么处理？", "skip", opts(("skip", "不是生产故障，不更新状态页"), ("outage", "写成全站故障"), ("maintenance", "写成计划维护")), "状态页文案交给工程写吗？", "no", YESNO, "状态页这件事有多急？", 0),
        pack("lookup", "客户在上海吗？", True, "客户在北京吗？", False, "按哪个城市的工作时间处理？", "shanghai", opts(("shanghai", "上海"), ("beijing", "北京"), ("london", "伦敦")), "需要把工单改成英文时区吗？", "no", YESNO, "时区问题有多急？", 0),
        pack("lookup", "客户买的是企业版吗？", True, "应该按免费版流程处理吗？", False, "套餐是哪一种？", "enterprise", opts(("enterprise", "企业版，年付 8600 元"), ("free", "免费版"), ("trial", "7 天试用")), "续费窗口属于哪一档？", "soon", opts(("soon", "还剩 12 天，不到 30 天"), ("far", "还有半年"), ("expired", "已经过期")), "续费临近但还没到期，排期怎么定？", 1),
        pack("lookup", "近 30 天这是客户唯一的一张工单吗？", True, "这是近 30 天的反复投诉吗？", False, "投诉模式属于哪一种？", "single", opts(("single", "30 天内只有 1 张工单"), ("repeat", "30 天内超过 5 张"), ("abuse", "疑似恶意刷单")), "要升级到投诉处理组吗？", "no", YESNO, "重复投诉的风险有多高？", 0, RISK),
        pack("lookup", "客户的情绪是焦急吗？", True, "客户的语气是平静的吗？", False, "回复语气应该怎么选？", "calm", opts(("calm", "先承认等待，语气稳，不刺激"), ("joke", "用玩笑带过"), ("blame", "指出是客户的银行问题并结束")), "可以先冷处理到下周吗？", "no", YESNO, "按焦急情绪，回复有多急？", 1),
        pack("lookup", "这张工单来自邮件吗？", True, "需要给客户回电话吗？", False, "回复渠道用哪个？", "email", opts(("email", "沿用邮件"), ("phone", "打电话"), ("sms", "发短信要验证码")), "要不要改成上门处理？", "no", YESNO, "更换渠道这件事有多急？", 0),
        pack("judgment", "续费是在 30 天内吗？", True, "服务今天就会因为没续费而中断吗？", False, "续费风险怎么处理？", "watch", opts(("watch", "还剩 12 天，今天先处理付款和发票，不断约"), ("cut", "今天就停服"), ("ignore", "续费与这张工单无关，完全不用看")), "续费谈判要马上交给销售吗？", "no", YESNO, "续费中断的紧急度是多少？", 1),
        pack("judgment", "按书面规则，这张工单应该今天处理吗？", True, "按书面规则，这张工单应该交给工程吗？", False, "最终路由是哪一个？", "finance", TEAMS, "自动化允许执行的下一步是哪个？", "retry", opts(("retry", "再自动重试 1 次付款，并让财务开发票，不降级"), ("downgrade", "降级并退款"), ("page", "更新状态页并呼叫工程值班")), "综合这张工单，处理有多急？", 1),
    ]
    return scenario("support", "企业客服工单", state, facts)


def home() -> dict[str, Any]:
    state = {
        "概况": (
            "现在是周二 22:40。用户说 23:10 到家。客厅太暗，需要有光。卧室灯不要打开。"
            "客房里有一位客人，客房灯保持客人自己控制，不要远程打开或关闭。"
            "外面在下雨，客厅窗是开着的。空调目前 26 度，用户没有提温度。"
            "门锁电量 18%，低于 20% 的提醒线。摄像头正常。没有烟雾报警。"
            "猫在家，不要启动全屋清洁机器人。洗衣机里有湿衣服，但用户没说要烘干。"
            "规则：到家前只处理用户点名的灯。下雨且窗开着时，到家场景要包含关窗提醒，不要自动锁死窗户。"
            "门锁低于 20% 发提醒，不要远程开锁。没有烟雾报警就不要通知消防。清洁机器人在宠物在家时禁止启动。"
        ),
        "记录": {
            "到达": "23:10",
            "客厅要光": True,
            "卧室灯": "关",
            "客房有人": True,
            "下雨": True,
            "客厅窗开着": True,
            "空调设定": 26,
            "用户提到空调": False,
            "门锁电量": 18,
            "摄像头正常": True,
            "烟雾报警": False,
            "宠物在家": True,
            "湿衣服": True,
            "用户要求烘干": False,
        },
        "_must": ["23:10", "卧室灯不要打开", "门锁电量 18%", "猫在家"],
    }
    facts = [
        pack("lookup", "用户是在 23 点以后到家吗？", True, "用户下午就到家吗？", False, "到家时间落在哪一档？", "late", opts(("late", "23:10，晚于 23 点"), ("evening", "19 点"), ("noon", "中午")), "需要按清晨场景执行吗？", "no", YESNO, "时间本身有多急？", 1),
        pack("lookup", "客厅需要增加光线吗？", True, "用户希望客厅保持全黑吗？", False, "客厅灯怎么设置？", "on", opts(("on", "打开或调亮，因为用户说太暗"), ("off", "保持关闭"), ("color", "改成彩色闪烁")), "客厅可以完全不处理吗？", "no", YESNO, "客厅灯光这件事有多急？", 1),
        pack("judgment", "卧室灯应该打开吗？", False, "用户禁止打开卧室灯吗？", True, "卧室灯怎么设置？", "off", opts(("off", "保持关闭"), ("on", "打开"), ("dim", "调到夜灯")), "可以把卧室灯当成客厅的替代吗？", "no", YESNO, "误开卧室灯的风险有多高？", 2, RISK),
        pack("judgment", "客房里有客人吗？", True, "可以远程把客房灯打开吗？", False, "客房灯怎么处理？", "leave", opts(("leave", "保持客人自己控制，不远程开关"), ("on", "远程打开"), ("off", "远程关掉")), "客房要纳入到家场景吗？", "no", YESNO, "打扰客人的风险有多高？", 2, RISK),
        pack("lookup", "外面在下雨吗？", True, "现在是晴天吗？", False, "天气属于哪一种？", "rain", opts(("rain", "下雨"), ("clear", "晴天"), ("snow", "下雪")), "天气需要纳入到家提醒吗？", "yes", YESNO, "天气相关提醒有多急？", 1),
        pack("judgment", "客厅窗开着并且外面在下雨吗？", True, "应该自动把窗户锁死吗？", False, "窗户怎么处理？", "remind", opts(("remind", "提醒关窗，不自动锁死"), ("lock", "自动锁死窗户"), ("ignore", "不提窗户")), "开着的窗会不会淋雨？", "yes", YESNO, "不提醒关窗的风险有多高？", 2, RISK),
        pack("lookup", "用户提到要改空调温度吗？", False, "空调现在是 26 度吗？", True, "空调怎么处理？", "keep", opts(("keep", "保持 26 度，用户没提"), ("heat", "改成 30 度"), ("off", "关掉")), "可以因为下雨就把空调改成制热吗？", "no", YESNO, "改空调的必要度有多高？", 0),
        pack("judgment", "门锁电量低于 20% 吗？", True, "应该远程替用户开锁吗？", False, "门锁怎么处理？", "remind", opts(("remind", "电量 18%，发提醒，不远程开锁"), ("unlock", "远程开锁"), ("ignore", "电量正常，不用管")), "门锁提醒今天要发吗？", "yes", YESNO, "门锁低电量有多急？", 1),
        pack("lookup", "摄像头是正常的吗？", True, "需要报修摄像头吗？", False, "摄像头怎么处理？", "keep", opts(("keep", "正常，不报修"), ("repair", "立刻报修"), ("cover", "转到卧室")), "摄像头要因为客人在家就关掉吗？", "no", YESNO, "摄像头维修有多急？", 0),
        pack("lookup", "现在有烟雾报警吗？", False, "应该通知消防吗？", False, "烟雾报警怎么处理？", "skip", opts(("skip", "没有报警，不通知消防"), ("call", "拨打消防"), ("test", "做一次报警测试")), "安全告警的紧急度是多少？", "none", opts(("none", "没有告警"), ("smoke", "烟雾"), ("gas", "燃气")), "消防通知有多急？", 0),
        pack("judgment", "宠物在家吗？", True, "可以启动全屋清洁机器人吗？", False, "清洁机器人怎么处理？", "forbid", opts(("forbid", "猫在家，禁止启动"), ("run", "立刻全屋清扫"), ("mop", "只拖地")), "清洁要排进到家场景吗？", "no", YESNO, "误启动清洁机器人的风险有多高？", 2, RISK),
        pack("lookup", "洗衣机里有湿衣服吗？", True, "用户要求现在烘干吗？", False, "湿衣服怎么处理？", "wait", opts(("wait", "有湿衣服，但用户没说烘干，先不动"), ("dry", "立刻烘干"), ("wash", "再洗一遍")), "烘干应该自动开始吗？", "no", YESNO, "湿衣服这件事有多急？", 0),
        pack("lookup", "到家场景应该在 23:10 前准备好吗？", True, "可以等到午夜以后再执行吗？", False, "场景触发时间怎么定？", "before", opts(("before", "23:10 到家前"), ("midnight", "午夜"), ("morning", "明早")), "现在离到家还有大约 30 分钟，要不要开始？", "yes", YESNO, "场景准备有多急？", 1),
        pack("judgment", "用户点名要处理的是客厅而不是卧室吗？", True, "规则允许顺手打开所有灯吗？", False, "灯的范围怎么定？", "named", opts(("named", "只处理点名的客厅，卧室保持关闭"), ("all", "打开所有灯"), ("none", "所有灯都不动")), "扩大到全屋灯光符合规则吗？", "no", YESNO, "扩大灯光范围的风险有多高？", 2, RISK),
        pack("lookup", "今天是周二晚上吗？", True, "今天是周末白天吗？", False, "日期时段属于哪一种？", "weeknight", opts(("weeknight", "周二夜间"), ("weekend", "周末白天"), ("monday", "周一早晨")), "要按周末派对场景执行吗？", "no", YESNO, "搞错日期场景的风险有多高？", 1, RISK),
        pack("lookup", "用户原话里包含不要打开卧室灯吗？", True, "用户原话要求打开卧室灯吗？", False, "遇到明确禁止项应该怎么做？", "obey", opts(("obey", "遵守不要打开卧室灯"), ("invert", "理解成相反的意思"), ("ask", "先打开再问")), "禁止项可以当成建议忽略吗？", "no", YESNO, "违反明确禁止的风险有多高？", 2, RISK),
        pack("judgment", "这次到家需要关窗提醒吗？", True, "这次到家需要通知消防吗？", False, "到家提醒应该包含哪一项？", "window", opts(("window", "下雨且窗开着，提醒关窗"), ("fire", "通知消防"), ("robot", "启动清洁机器人")), "提醒里应该包含远程开锁吗？", "no", YESNO, "关窗提醒有多急？", 1),
        pack("lookup", "摄像头正常并且没有烟雾报警吗？", True, "家里正在发生火警吗？", False, "家庭安全状态属于哪一档？", "calm", opts(("calm", "摄像头正常，无烟雾报警"), ("fire", "火警"), ("breakin", "入侵报警")), "需要把这次当成紧急安全事件吗？", "no", YESNO, "家庭安全事件的紧急度是多少？", 0),
        pack("judgment", "门锁提醒和开锁是两件事吗？", True, "低电量就应该远程开锁吗？", False, "低电量门锁的正确动作是哪个？", "notify", opts(("notify", "只提醒电量 18%"), ("unlock", "远程开锁"), ("remove", "拆掉门锁")), "可以把门锁提醒拖到下周吗？", "no", YESNO, "门锁低电量的紧急度是多少？", 1),
        pack("judgment", "到家自动化应该只开客厅灯、提醒关窗并提醒门锁低电量吗？", True, "到家自动化应该打开卧室、客房、清洁机器人和远程开锁吗？", False, "最终到家动作选哪一套？", "narrow", opts(("narrow", "客厅给光，卧室不动，提醒关窗和门锁低电量，不碰客房、机器人、开锁"), ("wide", "全屋开灯并清扫、开锁"), ("nothing", "什么都不做")), "最终方案包含远程开锁吗？", "no", YESNO, "这套到家动作有多急？", 1),
    ]
    return scenario("home", "晚归智能家居", state, facts)


def expense() -> dict[str, Any]:
    state = {
        "概况": (
            "员工周宁，周四从上海去北京见客户。高铁二等座 553 元，有发票。酒店 980 元，有发票，公司上限 800 元。"
            "出租车 186 元，发票丢了。客户晚餐 420 元，餐饮上限 300 元，有发票，同席有客户 2 人和同事 1 人。"
            "没有购买礼品。行程没有改签。返程是周五晚上，还没出发。项目号 P-17，客户是北辰制造。"
            "规则：超住宿上限的部分自付，不超过上限的部分报销。无发票的交通不能报销。餐饮超限只报销到上限。"
            "有客户同席的晚餐可以报销到上限，不算私人聚餐。礼品必须事先审批，没有审批就不能报。未发生的返程先不报。"
        ),
        "记录": {
            "住宿费用": 980,
            "住宿上限": 800,
            "出租费用": 186,
            "出租有发票": False,
            "晚餐费用": 420,
            "餐饮上限": 300,
            "晚餐有客户": True,
            "礼品": False,
            "返程已发生": False,
        },
        "_must": ["980 元", "上限 800 元", "发票丢了", "420 元", "餐饮上限 300 元"],
    }
    hotel_over = 180
    facts = [
        pack("lookup", "这趟出差的出发地是上海吗？", True, "这趟出差的出发地是广州吗？", False, "行程方向是哪一个？", "sh_bj", opts(("sh_bj", "上海去北京"), ("bj_sh", "北京去上海"), ("local", "同城")), "高铁座位是哪一档？", "second", opts(("second", "二等座 553 元"), ("first", "一等座"), ("flight", "飞机头等舱")), "交通票本身有多急？", 0),
        pack("lookup", "高铁票有发票吗？", True, "高铁票缺少发票吗？", False, "高铁票能不能报销？", "yes", opts(("yes", "二等座且有发票，可以报销"), ("no", "缺少发票"), ("upgrade", "改报成头等舱")), "高铁金额要不要自付？", "no", YESNO, "高铁报销的风险有多高？", 0, RISK),
        pack("lookup", "酒店价格超过公司上限了吗？", True, "酒店费用在 800 元上限以内吗？", False, "酒店超限多少？", "over180", opts(("over180", "980 减 800，超了 180 元"), ("under", "没有超限"), ("double", "超了 980 元")), "酒店发票齐全吗？", "yes", YESNO, "酒店票据风险有多高？", 1, RISK),
        pack("judgment", "超限的 180 元应该由员工自付吗？", True, "980 元可以全部报销吗？", False, "酒店报销金额应该是多少？", "cap", opts(("cap", "报销 800 元，自付 180 元"), ("all", "报销 980 元"), ("none", "酒店全部不报")), "超限部分可以让客户支付吗？", "no", YESNO, "酒店超限这件事有多急？", 1),
        pack("lookup", "出租车有发票吗？", False, "出租车发票丢了吗？", True, "186 元出租车能不能报销？", "no", opts(("no", "无发票的交通不能报销"), ("yes", "可以凭记忆报销"), ("half", "报销一半")), "可以事后补一张空白发票吗？", "no", YESNO, "无票交通的合规风险有多高？", 2, RISK),
        pack("judgment", "出租车费用应该从报销单里去掉吗？", True, "出租车可以和酒店打包报销吗？", False, "出租车这一项怎么处理？", "drop", opts(("drop", "删除 186 元，因为没有发票"), ("keep", "保留并报销"), ("fine", "改成罚款")), "需要财务通融无票出租吗？", "no", YESNO, "处理无票出租有多急？", 1),
        pack("lookup", "晚餐有客户同席吗？", True, "这是纯私人聚餐吗？", False, "晚餐性质属于哪一种？", "client", opts(("client", "客户 2 人和同事 1 人同席"), ("private", "私人朋友"), ("solo", "一个人")), "晚餐有发票吗？", "yes", YESNO, "晚餐性质判断有多急？", 0),
        pack("judgment", "晚餐可以报销到餐饮上限吗？", True, "420 元晚餐可以全额报销吗？", False, "晚餐报销金额应该是多少？", "cap", opts(("cap", "有客户同席，报销到上限 300 元"), ("all", "报销 420 元"), ("none", "有客户就不能报")), "超出的 120 元谁承担？", "self", opts(("self", "员工自付"), ("company", "公司全报"), ("client", "向客户收取")), "餐饮超限的合规风险有多高？", 1, RISK),
        pack("lookup", "这次买了礼品吗？", False, "有事先审批的礼品吗？", False, "礼品项怎么处理？", "none", opts(("none", "没有礼品，不报销礼品"), ("gift", "报销一份礼品"), ("cash", "折成现金")), "可以补一个礼品审批吗？", "no", YESNO, "礼品合规风险有多高？", 0, RISK),
        pack("lookup", "返程已经发生了吗？", False, "周五晚上的返程现在就能报销吗？", False, "返程怎么处理？", "wait", opts(("wait", "还没出发，先不报"), ("claim", "先按预估报销"), ("cancel", "取消返程")), "返程高铁现在有票吗？", "unknown", opts(("unknown", "概况没说已经买好返程票"), ("bought", "已经买好"), ("missed", "已经误车")), "返程报销有多急？", 0),
        pack("lookup", "项目号是 P-17 吗？", True, "这笔费用属于个人项目吗？", False, "费用记到哪个项目？", "p17", opts(("p17", "P-17 北辰制造"), ("personal", "个人"), ("p99", "P-99")), "客户名称是北辰制造吗？", "yes", YESNO, "项目归集有多急？", 0),
        pack("lookup", "行程有改签吗？", False, "需要报销改签费吗？", False, "改签费怎么处理？", "none", opts(("none", "没有改签，不报改签费"), ("fee", "报销改签费"), ("fine", "报销退票费")), "去程高铁需要改票吗？", "no", YESNO, "改签问题有多急？", 0),
        pack("judgment", "可报销的交通只包括有发票的高铁吗？", True, "出租车也应该算进可报销交通吗？", False, "交通合计应该包含哪些？", "train", opts(("train", "只含高铁 553 元"), ("both", "高铁加出租车"), ("none", "交通都不报")), "交通票据缺的是哪一项？", "taxi", opts(("taxi", "出租车发票丢失"), ("train", "高铁发票丢失"), ("none", "都不缺")), "交通合规风险有多高？", 1, RISK),
        pack("judgment", "公司为住宿和晚餐合计最多承担 1100 元吗？", True, "公司要承担 980 加 420 的全部吗？", False, "住宿加晚餐的公司承担额是多少？", "sum", opts(("sum", "住宿 800 加晚餐 300，共 1100 元"), ("gross", "980 加 420"), ("zero", "都不承担")), "员工自付部分包含酒店超限和晚餐超限吗？", "yes", YESNO, "金额算错的风险有多高？", 1, RISK),
        pack("lookup", "这是周四出发的北京客户拜访吗？", True, "这是周末旅游吗？", False, "出差类型属于哪一种？", "client", opts(("client", "周四见客户北辰制造"), ("tour", "旅游"), ("training", "内部培训")), "可以按旅游补贴处理吗？", "no", YESNO, "出差类型判断有多急？", 0),
        pack("lookup", "酒店有发票吗？", True, "酒店和出租车都缺发票吗？", False, "发票齐全的费用是哪些？", "hotel_meal_train", opts(("hotel_meal_train", "酒店、晚餐和高铁有发票"), ("taxi", "只有出租车有发票"), ("none", "都没有发票")), "缺发票的只有出租车吗？", "yes", YESNO, "发票审核有多急？", 1),
        pack("judgment", "晚餐因为有客户同席就可以超过 300 元上限全额报销吗？", False, "有客户同席只是让晚餐能够报销到上限吗？", True, "客户同席改变的是哪一条？", "eligible", opts(("eligible", "使晚餐可报，但仍受 300 元上限约束"), ("uncap", "取消上限"), ("ban", "有客户反而不能报")), "同席人数够不够支持公务晚餐？", "yes", YESNO, "把同席理解错的风险有多高？", 1, RISK),
        pack("lookup", "员工名字是周宁吗？", True, "报销单属于林晓吗？", False, "报销人是谁？", "zhou", opts(("zhou", "周宁"), ("lin", "林晓"), ("zhao", "赵晨")), "需要改成客户名下的报销吗？", "no", YESNO, "报销人核对有多急？", 0),
        pack("judgment", "未发生的返程不应该出现在本次报销里吗？", True, "可以把预计返程票先垫进本次吗？", False, "本次报销的时间范围怎么定？", "done", opts(("done", "只报已经发生且有票的去程、住宿和晚餐"), ("future", "加上未发生的返程"), ("all", "加上礼品和出租车")), "财务要不要等周五晚上再整单提交？", "no", YESNO, "提前报销返程的风险有多高？", 1, RISK),
        pack("judgment", "按规则，本次可报销的是高铁、住宿 800 元和晚餐 300 元吗？", True, "按规则，出租车、酒店超限、晚餐超限和返程都能报吗？", False, "最终报销处理选哪一套？", "strict", opts(("strict", "报高铁 553、住宿 800、晚餐 300；出租车、超限和未发生返程不报"), ("loose", "全部实报实销"), ("reject", "整单拒绝")), "未发生的返程包含在这套报销里吗？", "no", YESNO, "这张报销单有多急？", 1),
    ]
    del hotel_over
    return scenario("expense", "出差报销", state, facts)


def commute() -> dict[str, Any]:
    state = {
        "概况": (
            "周三早上。用户 8:10 还在家。9:00 有一场必须出席的客户会，迟到需要提前说明。"
            "平时路线 A 要 35 分钟，今天路线 A 有事故，导航预计 70 分钟。路线 B 没有事故，预计 50 分钟。"
            "外面在下雨。用户有伞。公司有淋浴，但用户没有带替换衣服。会议是视频加现场，用户必须到会议室。"
            "规则：要在 9:00 前到。8:10 出发走 B 预计 9:00 到，没有余量。走 A 会迟到约 20 分钟。"
            "下雨不是留在家里的理由。没有替换衣服就不要改骑自行车。若选 A，必须现在给客户发迟到说明。"
        ),
        "记录": {
            "出发时刻": "8:10",
            "会议": "9:00",
            "路线A分钟": 70,
            "路线B分钟": 50,
            "下雨": True,
            "有伞": True,
            "有替换衣服": False,
            "必须到场": True,
        },
        "_must": ["8:10", "9:00", "70 分钟", "50 分钟", "没有带替换衣服"],
    }
    facts = [
        pack("lookup", "会议是 9:00 吗？", True, "会议是下午吗？", False, "会议约束属于哪一种？", "hard", opts(("hard", "9:00 必须出席"), ("optional", "可参加可不参加"), ("done", "会议已结束")), "迟到需要说明吗？", "yes", YESNO, "会议迟到的紧急度是多少？", 2),
        pack("lookup", "用户现在还在家吗？", True, "用户已经在路上了吗？", False, "当前状态是哪一种？", "home", opts(("home", "8:10 还在家"), ("transit", "已在地铁上"), ("office", "已到公司")), "还可以选择路线吗？", "yes", YESNO, "出发决策有多急？", 2),
        pack("lookup", "路线 A 今天有事故吗？", True, "路线 A 今天是平时的 35 分钟吗？", False, "路线 A 今天要多久？", "slow", opts(("slow", "事故，导航 70 分钟"), ("normal", "35 分钟"), ("closed", "完全封闭")), "路线 A 能在 9:00 前到吗？", "no", YESNO, "误选路线 A 的风险有多高？", 2, RISK),
        pack("lookup", "路线 B 没有事故吗？", True, "路线 B 比路线 A 更慢吗？", False, "路线 B 预计多久？", "fifty", opts(("fifty", "50 分钟"), ("seventy", "70 分钟"), ("twenty", "20 分钟")), "路线 B 比 A 省时间吗？", "yes", YESNO, "比较两条路线有多急？", 2),
        pack("judgment", "8:10 走路线 B 会卡着 9:00 到吗？", True, "8:10 走路线 B 还有 15 分钟余量吗？", False, "路线 B 的时间余量属于哪一档？", "none", opts(("none", "50 分钟刚好到 9:00，没有余量"), ("spare", "还能提前 15 分钟"), ("late", "会迟到 20 分钟")), "走 B 还需要发迟到说明吗？", "no", YESNO, "走 B 仍要马上出发吗？", 2),
        pack("judgment", "走路线 A 会迟到大约 20 分钟吗？", True, "走路线 A 可以不通知客户吗？", False, "如果仍选路线 A，必须做什么？", "notice", opts(("notice", "现在发迟到说明"), ("silent", "什么都不说"), ("cancel", "取消客户会")), "路线 A 符合 9:00 前到达的规则吗？", "no", YESNO, "选 A 又不通知的风险有多高？", 2, RISK),
        pack("lookup", "外面在下雨吗？", True, "下雨就可以留在家里开会吗？", False, "下雨对出发的影响是哪一种？", "go", opts(("go", "下雨不是留在家里的理由"), ("stay", "下雨就改成在家视频"), ("wait", "等雨停再走")), "用户有伞吗？", "yes", YESNO, "天气本身有多急？", 1),
        pack("judgment", "用户没带替换衣服，所以不该改骑自行车吗？", True, "公司有淋浴就可以建议现在骑车吗？", False, "骑车方案怎么处理？", "skip", opts(("skip", "没有替换衣服，不要改骑自行车"), ("ride", "立刻骑车"), ("run", "跑步去")), "淋浴用得上吗？", "no", YESNO, "错误建议骑车的风险有多高？", 1, RISK),
        pack("lookup", "用户必须到会议室吗？", True, "这场会可以全程只开视频吗？", False, "出席方式必须是哪一种？", "room", opts(("room", "必须到会议室"), ("remote", "在家视频即可"), ("delegate", "让同事代替")), "人在路上视频算出席吗？", "no", YESNO, "到场要求有多硬？", 2),
        pack("lookup", "平时路线 A 是 35 分钟吗？", True, "今天还能按平时 35 分钟估算吗？", False, "今天估算应该用哪个数？", "live", opts(("live", "用今天的导航，不用平时 35 分钟"), ("usual", "仍用 35 分钟"), ("zero", "忽略通勤时间")), "导航和历史哪个优先？", "nav", opts(("nav", "今天的导航"), ("history", "历史平均"), ("guess", "凭感觉")), "用错通勤时间的风险有多高？", 2, RISK),
        pack("judgment", "现在就该出发吗？", True, "可以再等 20 分钟再走吗？", False, "出发动作选哪个？", "now", opts(("now", "马上走路线 B"), ("wait", "再等 20 分钟"), ("stay", "留在家里")), "再等 20 分钟还会准时吗？", "no", YESNO, "出发有多急？", 2),
        pack("lookup", "今天是周三早上吗？", True, "今天是周日吗？", False, "这是工作日通勤吗？", "yes", YESNO, "要按周末出行规划吗？", "no", YESNO, "日期判断有多急？", 0),
        pack("lookup", "客户会迟到需要提前说明吗？", True, "内部站会迟到也可以不说吗？", False, "通知对象是谁？", "client", opts(("client", "客户"), ("nobody", "不用通知"), ("family", "只告诉家人")), "说明应该在出发前发吗？", "only_if_late", opts(("only_if_late", "只有会迟到才发"), ("always", "准时也要发迟到说明"), ("never", "永远不发")), "通知判断有多急？", 1),
        pack("judgment", "路线 B 是今天能赶上 9:00 的路线吗？", True, "路线 A 是今天能赶上 9:00 的路线吗？", False, "应该选哪条路线？", "b", opts(("b", "路线 B，50 分钟"), ("a", "路线 A，70 分钟"), ("walk", "步行")), "选路时要不要忽略事故？", "no", YESNO, "选错路的风险有多高？", 2, RISK),
        pack("lookup", "用户有伞，所以下雨仍可步行接驳吗？", True, "没有伞就不能出门吗？", False, "雨具状态是哪一种？", "ready", opts(("ready", "有伞"), ("none", "没有雨具"), ("coat", "只有雨衣没有伞")), "缺的是雨具还是替换衣服？", "clothes", opts(("clothes", "缺替换衣服，不缺伞"), ("umbrella", "缺伞"), ("both", "都缺")), "雨具问题有多急？", 0),
        pack("lookup", "会议同时有视频和现场吗？", True, "有视频就表示人不用到场吗？", False, "视频在这场会里起什么作用？", "extra", opts(("extra", "有视频，但人仍必须到会议室"), ("replace", "视频可以替代到场"), ("none", "没有视频")), "可以只打开家里的视频吗？", "no", YESNO, "误解视频会议的风险有多高？", 2, RISK),
        pack("judgment", "8:10 加 50 分钟等于 9:00 吗？", True, "8:10 加 70 分钟仍早于 9:00 吗？", False, "两条路线的到达时间怎么算？", "exact", opts(("exact", "B 约 9:00 到，A 约 9:20 到"), ("reverse", "A 更早"), ("same", "两条一样")), "计算时要从 8:10 起算吗？", "yes", YESNO, "算错到达时间的风险有多高？", 2, RISK),
        pack("lookup", "公司有淋浴吗？", True, "有淋浴就足够支持骑车方案吗？", False, "淋浴信息怎么用？", "not_enough", opts(("not_enough", "有淋浴但没有替换衣服，仍不建议骑车"), ("enough", "有淋浴就可以骑车"), ("irrelevant", "概况没提淋浴")), "衣服是限制项吗？", "yes", YESNO, "装备判断有多急？", 0),
        pack("judgment", "若选择路线 B，现在不用发迟到说明吗？", True, "若选择路线 B，还可以在家停留一会儿吗？", False, "路线 B 的配套动作是哪个？", "leave", opts(("leave", "马上出发，不发迟到说明"), ("notice", "先发迟到说明再慢慢走"), ("wait", "等事故结束走 A")), "路线 B 有余量缓冲吗？", "no", YESNO, "路线 B 的执行有多急？", 2),
        pack("judgment", "按规则现在应该走路线 B 并立刻出发吗？", True, "按规则现在应该走路线 A 或留在家里吗？", False, "最终通勤决定是哪一个？", "b", opts(("b", "立刻走路线 B，不发迟到说明，不骑车"), ("a", "走路线 A 并什么都不说"), ("home", "留在家里用视频")), "最终决定包含骑车吗？", "no", YESNO, "这个通勤决定有多急？", 2),
    ]
    return scenario("commute", "雨天通勤", state, facts)


def clinic() -> dict[str, Any]:
    state = {
        "概况": (
            "父亲周五 14:00 在市一医院复查，要带上周二的验血报告。母亲不会开车。姐姐人在外地，周五回不来。"
            "用户自己有车，周五 14:00 前没有会。医院停车位紧张，院内停车场已满的公告发在今天早上。"
            "父亲能自己走路，不需要轮椅。复查不是急诊。报告在用户家里的桌子上，还没放进包。"
            "规则：姐姐不在本市就不能安排姐姐开车。母亲不会开车就不能让母亲单独送。用户有空且有车，应由用户送。"
            "停车位紧张时改停地铁站换乘，不要把车停在消防通道。报告没装包之前，出门清单必须包含报告。不是急诊就不要走急救通道。"
        ),
        "记录": {
            "时间": "周五 14:00",
            "医院": "市一医院",
            "母亲会开车": False,
            "姐姐在本市": False,
            "用户有车": True,
            "用户有空": True,
            "院内车位紧张": True,
            "需要轮椅": False,
            "急诊": False,
            "报告已装包": False,
        },
        "_must": ["周五 14:00", "姐姐人在外地", "母亲不会开车", "报告在用户家里"],
    }
    facts = [
        pack("lookup", "复查时间是周五 14:00 吗？", True, "这是今晚的急诊吗？", False, "就诊类型属于哪一种？", "followup", opts(("followup", "周五 14:00 复查"), ("er", "急诊"), ("surgery", "立刻手术")), "可以改到晚上吗？", "no", YESNO, "时间协调有多急？", 1),
        pack("lookup", "要带周二的验血报告吗？", True, "报告已经装进包里了吗？", False, "报告现在在哪里？", "desk", opts(("desk", "在家里的桌子上，还没装包"), ("bag", "已经在包里"), ("hospital", "医院已经有了，不用带")), "出门清单要包含报告吗？", "yes", YESNO, "忘带报告的风险有多高？", 2, RISK),
        pack("lookup", "母亲会开车吗？", False, "可以让母亲单独开车送父亲吗？", False, "母亲适合担任司机吗？", "no", opts(("no", "不会开车，不能单独送"), ("yes", "可以开车"), ("taxi", "母亲负责叫救护车")), "母亲可以坐在车上陪同吗？", "yes", YESNO, "让母亲开车的风险有多高？", 2, RISK),
        pack("lookup", "姐姐周五人在外地吗？", True, "姐姐周五能开车送到医院吗？", False, "姐姐能不能当司机？", "no", opts(("no", "人在外地，周五回不来"), ("yes", "就在本市"), ("maybe", "可能下午到")), "应该把接送任务发给姐姐吗？", "no", YESNO, "依赖姐姐的风险有多高？", 2, RISK),
        pack("lookup", "用户自己有车吗？", True, "用户周五 14:00 前有会吗？", False, "用户有没有条件开车？", "yes", opts(("yes", "有车，而且 14:00 前没有会"), ("busy", "有会走不开"), ("nocar", "没有车")), "用户是不是规则里的司机？", "yes", YESNO, "确认司机有多急？", 1),
        pack("judgment", "按规则应该由用户送父亲吗？", True, "按规则应该由姐姐或母亲开车吗？", False, "司机应该选谁？", "user", opts(("user", "用户自己开车"), ("sister", "姐姐"), ("mother", "母亲")), "需要再找第三方代驾吗？", "no", YESNO, "司机选错的风险有多高？", 2, RISK),
        pack("lookup", "院内停车场很紧张吗？", True, "今天早上有车位已满的公告吗？", True, "停车预期属于哪一种？", "tight", opts(("tight", "院内车位紧张"), ("easy", "车位充足"), ("free", "医院免费代客泊车")), "可以直接开进院内停好吗？", "no", YESNO, "停车问题有多急？", 1),
        pack("judgment", "应该改停地铁站再换乘吗？", True, "可以把车停在消防通道吗？", False, "停车方案选哪个？", "metro", opts(("metro", "停地铁站换乘，不占消防通道"), ("hydrant", "停消防通道"), ("circle", "在院门口违停等人")), "停车方案要写进出门清单吗？", "yes", YESNO, "违停的风险有多高？", 2, RISK),
        pack("lookup", "父亲能自己走路吗？", True, "需要准备轮椅吗？", False, "行动协助属于哪一档？", "walk", opts(("walk", "能自己走，不用轮椅"), ("wheelchair", "需要轮椅"), ("stretcher", "需要担架")), "要预约无障碍车辆吗？", "no", YESNO, "行动协助有多急？", 0),
        pack("lookup", "这次是急诊吗？", False, "应该走急救通道吗？", False, "入院路线怎么选？", "clinic", opts(("clinic", "走复查普通入口"), ("er", "走急救通道"), ("inpatient", "直接去住院部")), "需要叫救护车吗？", "no", YESNO, "急救误判的风险有多高？", 2, RISK),
        pack("lookup", "医院是市一医院吗？", True, "复查在社区诊所吗？", False, "目的地是哪里？", "city1", opts(("city1", "市一医院"), ("clinic", "社区诊所"), ("home", "医生上门")), "导航目的地要设成市一医院吗？", "yes", YESNO, "目的地确认有多急？", 0),
        pack("judgment", "报告还在桌上，所以出发前必须先装包吗？", True, "人到了医院再让邻居送报告来得及吗？", False, "出发前第一件物品准备是什么？", "report", opts(("report", "把验血报告放进包"), ("wheelchair", "先找轮椅"), ("gift", "买水果")), "可以只记得带医保卡、不带报告吗？", "no", YESNO, "报告准备有多急？", 2),
        pack("lookup", "复查日是周五吗？", True, "复查日是周一凌晨吗？", False, "日程应该写在哪一天？", "friday", opts(("friday", "周五 14:00"), ("monday", "周一凌晨"), ("daily", "每天")), "需要请周五下午的假吗？", "yes", YESNO, "请假这件事有多急？", 1),
        pack("lookup", "用户 14:00 前没有会议冲突吗？", True, "用户要同时出席公司会和医院吗？", False, "日程冲突属于哪一种？", "none", opts(("none", "14:00 前没有会，可以去送"), ("clash", "和董事会冲突"), ("unknown", "概况没提用户日程")), "需要改会议时间吗？", "no", YESNO, "日程冲突的风险有多高？", 0, RISK),
        pack("judgment", "姐姐在外地是排除姐姐开车的充分理由吗？", True, "姐姐可以远程把车开到医院吗？", False, "对姐姐的安排应该是哪一种？", "inform", opts(("inform", "可以告知病情，但不安排她周五开车"), ("drive", "让她周五开车"), ("ticket", "给她订周五上午的紧急机票当司机")), "把任务留给姐姐符合规则吗？", "no", YESNO, "错派姐姐的风险有多高？", 2, RISK),
        pack("lookup", "父亲复查要用户陪同开车，不是父亲一个人打车吗？", True, "父亲已经自己叫好车了吗？", False, "交通责任在谁？", "user", opts(("user", "用户开车送"), ("father", "父亲自己打车"), ("sister", "姐姐负责")), "需要再确认父亲是否独自前往吗？", "no", YESNO, "交通责任确认有多急？", 1),
        pack("judgment", "停车紧张不改变司机人选吗？", True, "因为停车紧张就改成母亲开车吗？", False, "停车和司机的关系是哪一种？", "separate", opts(("separate", "司机仍是用户，只改变停车地点"), ("switch", "改由母亲开"), ("cancel", "取消复查")), "车位公告是今天早上发的吗？", "yes", YESNO, "把两件事混在一起的风险有多高？", 1, RISK),
        pack("lookup", "消防通道禁止停车吗？", True, "公告允许临时停在消防通道吗？", False, "哪些停车地点是明确禁止的？", "hydrant", opts(("hydrant", "消防通道"), ("metro", "地铁站停车场"), ("lot", "正规停车场")), "禁止项可以因为父亲行动慢就打破吗？", "no", YESNO, "违停风险有多高？", 2, RISK),
        pack("judgment", "出门前的必要动作包含装报告和设定地铁站停车吗？", True, "出门前应该先找轮椅和走急救通道吗？", False, "出发清单选哪一套？", "list", opts(("list", "装上验血报告，用户开车，停地铁站换乘"), ("er", "轮椅、急救通道、母亲开车"), ("none", "什么都不用准备")), "清单要在周五中午前确认吗？", "yes", YESNO, "出发准备有多急？", 1),
        pack("judgment", "按规则，用户开车、不派姐姐和母亲、不停消防通道、带上报告吗？", True, "按规则，这是急诊并且可以由姐姐开车吗？", False, "最终就医安排选哪一个？", "user", opts(("user", "用户送，带报告，地铁站停车，普通入口"), ("sister", "姐姐开车走急救通道"), ("mother", "母亲开车停消防通道")), "最终安排会让母亲开车吗？", "no", YESNO, "这件接送安排有多急？", 1),
    ]
    return scenario("clinic", "家属就医", state, facts)


def shop() -> dict[str, Any]:
    state = {
        "概况": (
            "订单 S-552 的黑色跑鞋昨天签收。尺码买了 42，实际偏小，用户平时穿 43。只在室内试过一次，鞋底没有室外磨损。"
            "吊牌已经剪掉。鞋盒还在。购买不满 7 天，退货窗口是 7 天。用户明确要退款，不要换货。"
            "支付用的是信用卡。运费险未购买。商品页面写着吊牌剪掉仍可退，只要没有室外穿着。"
            "规则：7 天内、无室外磨损可以退。吊牌剪掉按页面承诺不构成拒绝理由。用户要退款就不要改成换大一码。"
            "没有运费险时，退回运费由买家承担，商品款仍退。信用卡退款退回原支付方式，不要改成钱包余额。"
        ),
        "记录": {
            "签收": "昨天",
            "购买天数": 1,
            "退货窗口天": 7,
            "吊牌已剪": True,
            "室外磨损": False,
            "鞋盒还在": True,
            "用户要退款": True,
            "用户要换货": False,
            "有运费险": False,
            "支付方式": "信用卡",
        },
        "_must": ["昨天签收", "吊牌已经剪掉", "不要换货", "运费险未购买"],
    }
    facts = [
        pack("lookup", "鞋子是昨天签收的吗？", True, "已经超过 7 天退货窗口吗？", False, "退货时限属于哪一档？", "inside", opts(("inside", "购买约 1 天，窗口 7 天，还在期内"), ("edge", "刚好第 8 天"), ("late", "已经一个月")), "现在发起退货来得及吗？", "yes", YESNO, "时限问题有多急？", 1),
        pack("lookup", "尺码买小了吗？", True, "用户平时穿 42 吗？", False, "尺码问题是哪一种？", "small", opts(("small", "买了 42，平时穿 43，偏小"), ("large", "买大了"), ("color", "颜色错了")), "这是质量损坏吗？", "no", YESNO, "尺码问题有多急？", 1),
        pack("lookup", "鞋底有室外磨损吗？", False, "只在室内试过一次吗？", True, "穿着程度属于哪一种？", "indoor", opts(("indoor", "室内试过一次，无室外磨损"), ("outdoor", "已经户外跑过"), ("damaged", "鞋面破损")), "穿着程度阻止退货吗？", "no", YESNO, "磨损争议的风险有多高？", 1, RISK),
        pack("lookup", "吊牌已经剪掉了吗？", True, "吊牌还在鞋上吗？", False, "吊牌状态怎么影响退货？", "ok", opts(("ok", "页面写明剪掉仍可退"), ("block", "剪掉就不能退"), ("fine", "剪掉要扣一半货款")), "剪掉吊牌要扣一半货款吗？", "no", YESNO, "吊牌问题有多急？", 0),
        pack("judgment", "按页面承诺，剪掉吊牌仍可退吗？", True, "可以因为吊牌剪掉而拒绝退货吗？", False, "审核吊牌时应该怎么做？", "allow", opts(("allow", "不把剪吊牌当作拒绝理由"), ("deny", "直接拒绝"), ("photo", "要求把吊牌粘回去")), "鞋盒在，有助于退货吗？", "yes", YESNO, "错误拒绝的风险有多高？", 2, RISK),
        pack("lookup", "用户要的是退款吗？", True, "用户想换大一码吗？", False, "用户的诉求是哪一种？", "refund", opts(("refund", "退款，不要换货"), ("exchange", "换成 43 码"), ("repair", "免费修补")), "可以自作主张改成换货吗？", "no", YESNO, "违背退款诉求的风险有多高？", 2, RISK),
        pack("judgment", "应该退款而不是换货吗？", True, "规则允许改成换大一码吗？", False, "售后方案选哪个？", "refund", opts(("refund", "退回商品款"), ("exchange", "换成 43 码"), ("keep", "劝用户留下")), "换货会违反用户的明确要求吗？", "yes", YESNO, "方案选错有多急？", 1),
        pack("lookup", "购买了运费险吗？", False, "退回运费应该由买家承担吗？", True, "退回运费谁出？", "buyer", opts(("buyer", "没有运费险，买家承担退回运费"), ("seller", "卖家承担"), ("insurance", "运费险承担")), "商品款还退吗？", "yes", YESNO, "运费争议的风险有多高？", 1, RISK),
        pack("judgment", "没有运费险就不退商品款吗？", False, "运费和商品款要分开处理吗？", True, "款项怎么拆？", "split", opts(("split", "退商品款，退回运费由买家出"), ("none", "商品款和运费都不退"), ("both", "连买家的退回运费也由卖家出")), "可以把运费从商品款里扣掉而不说明吗？", "no", YESNO, "款项算错的风险有多高？", 2, RISK),
        pack("lookup", "支付方式是信用卡吗？", True, "应该把退款打进站内钱包吗？", False, "退款去向是哪里？", "card", opts(("card", "退回原信用卡"), ("wallet", "改成钱包余额"), ("cash", "现金")), "可以改退到另一张卡吗？", "no", YESNO, "退款路径有多急？", 1),
        pack("lookup", "订单号是 S-552 吗？", True, "这是食品订单吗？", False, "商品是什么？", "shoes", opts(("shoes", "黑色跑鞋"), ("food", "食品"), ("phone", "手机")), "颜色是黑色吗？", "yes", YESNO, "订单核对有多急？", 0),
        pack("lookup", "鞋盒还在吗？", True, "包装已经全部扔掉了吗？", False, "退回时包装怎么处理？", "box", opts(("box", "用还在的鞋盒寄回"), ("none", "没有包装所以不能退"), ("new", "必须买一个新礼盒")), "缺少鞋盒是这单的情况吗？", "no", YESNO, "包装问题有多急？", 0),
        pack("judgment", "无室外磨损满足退货条件吗？", True, "室内试穿一次就算室外穿着吗？", False, "试穿记录支持退货吗？", "yes", opts(("yes", "室内一次且鞋底无室外磨损，支持退货"), ("no", "试过就不能退"), ("maybe", "要送到鉴定中心")), "需要因为试穿扣款吗？", "no", YESNO, "试穿争议的风险有多高？", 1, RISK),
        pack("lookup", "用户是在退货窗口内提出的吗？", True, "窗口只剩最后一天所以要拒绝吗？", False, "窗口使用情况属于哪一种？", "early", opts(("early", "昨天签收，远在 7 天内"), ("last", "最后 2 小时"), ("closed", "窗口已关")), "要催用户今天内寄出吗？", "no", YESNO, "窗口压力有多大？", 0),
        pack("judgment", "页面承诺和吊牌已剪可以同时成立吗？", True, "页面承诺会被吊牌规则取消吗？", False, "两条规定冲突时听谁的？", "page", opts(("page", "听商品页面：剪掉吊牌仍可退"), ("tag", "听通用吊牌规则并拒绝"), ("agent", "听客服心情")), "需要向用户引用页面承诺吗？", "yes", YESNO, "引用错规则的风险有多高？", 2, RISK),
        pack("lookup", "这双鞋偏小是尺寸问题而不是假货争议吗？", True, "用户指控买到假货了吗？", False, "问题类型属于哪一种？", "size", opts(("size", "尺码偏小"), ("fake", "假货"), ("missing", "空包裹")), "要转给打假部门吗？", "no", YESNO, "问题定性有多急？", 0),
        pack("judgment", "退款原路退回并且不换货，符合用户要求和规则吗？", True, "改成钱包余额并换 43 码，符合要求吗？", False, "退款执行方式选哪个？", "card", opts(("card", "信用卡原路退商品款，不换货"), ("wallet", "退到钱包并换货"), ("coupon", "只发优惠券")), "可以扣掉无运费险来拒绝整单吗？", "no", YESNO, "执行方式有多急？", 1),
        pack("lookup", "昨天签收意味着商品在用户手里吗？", True, "包裹还在快递柜没取吗？", False, "货物状态是哪一种？", "received", opts(("received", "已签收并试穿"), ("transit", "运输中"), ("lost", "丢失")), "需要先做丢件理赔吗？", "no", YESNO, "物流状态有多急？", 0),
        pack("judgment", "买家承担的只是退回运费，不是商品款吗？", True, "买家要同时损失商品款和运费吗？", False, "买家成本应该是哪一种？", "ship", opts(("ship", "只承担退回运费"), ("both", "商品款加运费都不退"), ("zero", "运费也由卖家出")), "这个成本和有没有运费险直接相关吗？", "yes", YESNO, "向用户说错费用的风险有多高？", 2, RISK),
        pack("judgment", "按规则应该同意退款、原路退回、不换货、买家自付退回运费吗？", True, "按规则应该拒绝，因为吊牌剪了而且没有运费险吗？", False, "最终售后决定是哪一个？", "approve", opts(("approve", "同意退款到信用卡，不换货，退回运费买家承担"), ("deny", "因吊牌和运费险拒绝"), ("exchange", "改成换 43 码")), "最终决定会改成换货吗？", "no", YESNO, "这单售后有多急？", 1),
    ]
    return scenario("shop", "鞋码退换", state, facts)


def calendar() -> dict[str, Any]:
    state = {
        "概况": (
            "周四 15:00 到 16:00 已经有牙医预约，改期要收 200 元。新邀请是周四 15:30 到 16:30 的产品评审，组织者是直属老板，不参加需要请假。"
            "周五 10:00 有一场可选的内训，和任何安排都不冲突。报告截止日期是周五 18:00，报告还没开始写，预计要 4 小时。"
            "周四晚上 19:00 以后用户没有安排。牙医诊所周四只有 15:00 这一个空档。"
            "规则：已经付费且难改的牙医不直接覆盖。和老板的评审时间重叠，就不能直接接受，应该提议改到周四 19:00 以后并说明牙医冲突。"
            "可选内训可以不去。报告还没开始且周五截止，周四评审若改到晚上，周五上午不要再排内训，留给写报告。"
        ),
        "记录": {
            "牙医": "周四 15:00-16:00",
            "改牙医科费": 200,
            "评审": "周四 15:30-16:30",
            "组织者": "直属老板",
            "不参加要请假": True,
            "内训": "周五 10:00 可选",
            "内训冲突": False,
            "报告截止": "周五 18:00",
            "报告已开始": False,
            "报告预计小时": 4,
            "周四晚空闲": True,
            "牙医另有空档": False,
        },
        "_must": ["15:30", "200 元", "直属老板", "报告还没开始写"],
    }
    facts = [
        pack("lookup", "周四下午已经有牙医预约吗？", True, "牙医可以免费改期吗？", False, "牙医改期成本是多少？", "fee", opts(("fee", "改期收 200 元，且周四只有这一个空档"), ("free", "可以免费改"), ("none", "没有牙医")), "牙医应该被直接覆盖吗？", "no", YESNO, "丢掉牙医号的风险有多高？", 2, RISK),
        pack("lookup", "新评审和牙医时间重叠吗？", True, "评审在周五早上吗？", False, "重叠的时段是哪一个？", "thu", opts(("thu", "周四 15:30 落在牙医的 15:00-16:00 里"), ("fri", "周五上午"), ("none", "不重叠")), "两场都在周四下午吗？", "yes", YESNO, "冲突判断有多急？", 2),
        pack("lookup", "评审组织者是直属老板吗？", True, "这是可以默默忽略的可选会吗？", False, "缺席评审的后果是哪一种？", "leave", opts(("leave", "不参加需要请假"), ("free", "可以无故缺席"), ("bonus", "缺席有奖励")), "可以直接拒绝老板而不说明吗？", "no", YESNO, "得罪老板的风险有多高？", 2, RISK),
        pack("judgment", "可以直接接受评审并覆盖牙医吗？", False, "应该提议改时间而不是直接接受吗？", True, "对评审邀请的第一动作是哪个？", "propose", opts(("propose", "不直接接受，提议改期并说明牙医"), ("accept", "直接接受并覆盖牙医"), ("decline", "直接拒绝不说明")), "提议的时间应该避开牙医吗？", "yes", YESNO, "处理邀请有多急？", 2),
        pack("lookup", "周四 19:00 以后用户有空吗？", True, "周四晚上已经排满了吗？", False, "可提议的替代时间是哪一个？", "thu_night", opts(("thu_night", "周四 19:00 以后"), ("thu_1530", "就保持 15:30"), ("sat", "周六凌晨")), "牙医诊所周四还有别的空档吗？", "no", YESNO, "找替代时间有多急？", 1),
        pack("judgment", "替代评审应该放在周四晚上而不是挪牙医吗？", True, "应该付 200 元改牙医来迎合 15:30 吗？", False, "哪一件更不该动？", "dentist", opts(("dentist", "牙医难改且要花钱，评审改到晚上"), ("review", "放弃评审"), ("both", "两场都按原时间出席")), "规则写了不要直接覆盖牙医吗？", "yes", YESNO, "付改期费的必要度有多高？", 0),
        pack("lookup", "周五内训是可选的吗？", True, "周五内训和现有安排冲突吗？", False, "内训本身有没有时间冲突？", "no", opts(("no", "周五 10:00 可选且不冲突"), ("yes", "和牙医冲突"), ("required", "必修")), "内训因为时间冲突必须去吗？", "no", YESNO, "内训决策有多急？", 0),
        pack("judgment", "报告还没开始并且周五 18:00 截止吗？", True, "报告已经写完了吗？", False, "报告工作量属于哪一档？", "four", opts(("four", "还没开始，预计 4 小时，周五 18:00 截止"), ("done", "已经写完"), ("month", "下个月才截止")), "周五上午适合再排一场可去可不去的内训吗？", "no", YESNO, "报告截止的紧急度是多少？", 1),
        pack("judgment", "评审若改到周四晚上，周五上午就应该留给报告吗？", True, "评审改期后仍应该去可选内训吗？", False, "周五上午怎么安排？", "report", opts(("report", "不去内训，用来写报告"), ("training", "去可选内训"), ("dentist", "再看一次牙医")), "内训是必修所以必须去吗？", "no", YESNO, "挤掉写报告的风险有多高？", 2, RISK),
        pack("lookup", "不参加评审需要请假吗？", True, "请假可以代替改期提议吗？", False, "若完全不参加评审，手续是哪一种？", "leave", opts(("leave", "需要请假"), ("silent", "直接消失"), ("email", "只发一个表情")), "规则首选是请假不去，还是改期参加？", "move", opts(("move", "提议改期并参加"), ("skip", "请假不去"), ("both", "既不请假也不改期")), "处理缺席手续有多急？", 1),
        pack("lookup", "牙医预约长一小时，评审也跨过 16:00 吗？", True, "评审 16:00 就结束所以不重叠吗？", False, "重叠长度大约是多少？", "half", opts(("half", "至少 15:30 到 16:00 这 30 分钟重叠"), ("zero", "没有重叠"), ("allday", "重叠一整天")), "只迟到 30 分钟去评审、牙医也不改，可行吗？", "no", YESNO, "硬挤两场的风险有多高？", 2, RISK),
        pack("lookup", "产品评审是新邀请吗？", True, "评审早就接受了吗？", False, "邀请状态属于哪一种？", "new", opts(("new", "新邀请，还没接受"), ("accepted", "已经接受"), ("declined", "已经拒绝")), "还来得及提议改期吗？", "yes", YESNO, "回复邀请有多急？", 2),
        pack("judgment", "说明里应该提到牙医冲突吗？", True, "应该假装没有冲突直接接受吗？", False, "给老板的回复要包含什么？", "reason", opts(("reason", "说明周四 15:00 有难改的牙医，提议周四 19:00 以后"), ("accept", "只回复接受"), ("no", "不回复")), "需要提到 200 元改期费吗？", "yes", YESNO, "回复质量有多急？", 1),
        pack("lookup", "报告预计要 4 小时吗？", True, "报告半小时就能写完吗？", False, "4 小时报告放在哪天做？", "friday", opts(("friday", "周五白天，跳过可选内训"), ("thursday_1500", "周四 15:00，和两场会叠在一起"), ("next", "下周一")), "周四晚上如果开了评审，报告主要还是周五做吗？", "yes", YESNO, "报告排期有多急？", 1),
        pack("lookup", "诊所周四只有 15:00 一个空档吗？", True, "牙医可以改到周四早上吗？", False, "牙医科目的灵活性属于哪一档？", "fixed", opts(("fixed", "只有 15:00，改期还要 200 元"), ("flexible", "当天随便改"), ("none", "没有预约")), "用诊所空档来决定谁让路吗？", "yes", YESNO, "忽略诊所限制的风险有多高？", 2, RISK),
        pack("judgment", "可选内训本身不冲突，所以拒绝它的理由是报告而不是时间撞车吗？", True, "内训和牙医时间重叠吗？", False, "不去内训的原因是哪一个？", "report", opts(("report", "把周五上午留给还没开始的报告"), ("clash", "和牙医冲突"), ("ban", "公司禁止内训")), "内训组织者是老板吗？", "no", YESNO, "拒绝内训的风险有多高？", 0, RISK),
        pack("lookup", "周四晚上空闲可以承接改期后的评审吗？", True, "周四晚上还要看牙医吗？", False, "周四晚上的用途是哪一种？", "review", opts(("review", "改期后的产品评审"), ("dentist", "牙医"), ("training", "内训")), "晚上开会会碰到牙医吗？", "no", YESNO, "晚上档期有多急？", 1),
        pack("judgment", "直接拒绝评审而不提议替代时间，符合规则吗？", False, "不参加才需要请假，改期参加就不必请假吗？", True, "请假和改期怎么选？", "propose", opts(("propose", "先提议改到周四晚上，不先请假"), ("leave", "直接请假缺席"), ("both", "改期成功后仍提交缺席假")), "老板的会可以用沉默处理吗？", "no", YESNO, "处理方式有多急？", 2),
        pack("lookup", "报告截止日期是周五 18:00 吗？", True, "报告周一才交吗？", False, "截止时间属于哪一档？", "friday", opts(("friday", "周五 18:00"), ("monday", "下周一"), ("none", "没有截止")), "现在还没开始写，来得及在截止前写完 4 小时吗？", "yes", YESNO, "截止日期的压力有多大？", 1),
        pack("judgment", "按规则应该保留牙医、把评审改到周四晚上、周五上午写报告并不去内训吗？", True, "按规则应该接受 15:30 评审、取消牙医并去内训吗？", False, "最终日程方案选哪一个？", "plan", opts(("plan", "保留牙医，评审改到周四 19:00 后，不去内训，周五写报告"), ("accept", "接受 15:30 并取消牙医"), ("train", "去内训，报告以后再说")), "最终方案会取消牙医吗？", "no", YESNO, "这组日程决定有多急？", 2),
    ]
    return scenario("calendar", "日程冲突", state, facts)


def sms() -> dict[str, Any]:
    state = {
        "概况": (
            "用户收到短信：发件人是短号 1069，不是银行官方号。短信说包裹扣关，要用户点链接 example-parcel.cc 并填写短信验证码。"
            "用户今天没有在途包裹。用户的银行是明德银行，短信里的链接域名不是明德银行的官网。"
            "短信要求提供验证码和银行卡密码。用户还没有点击。"
            "规则：非官方短号、陌生域名、索要验证码或密码，一律当成钓鱼。不要点击，不要回短信。"
            "没有在途包裹就不能为了取包裹去填码。应在银行官方 App 里看通知，而不是用短信链接。可以把短信上报到手机系统的垃圾短信。"
        ),
        "记录": {
            "发件人": "短号 1069",
            "官方银行号": False,
            "链接域名": "example-parcel.cc",
            "银行": "明德银行",
            "域名是银行官网": False,
            "索要验证码": True,
            "索要密码": True,
            "有在途包裹": False,
            "已经点击": False,
        },
        "_must": ["1069", "example-parcel.cc", "没有在途包裹", "不要点击"],
    }
    facts = [
        pack("lookup", "发件人是短号 1069 吗？", True, "发件人是银行官方号吗？", False, "发件人属于哪一种？", "short", opts(("short", "非官方短号 1069"), ("bank", "明德银行官方号"), ("friend", "通讯录联系人")), "可以因为是短号就信任吗？", "no", YESNO, "发件人风险有多高？", 2, RISK),
        pack("lookup", "链接域名是 example-parcel.cc 吗？", True, "这个域名是明德银行官网吗？", False, "链接应该怎么定性？", "phish", opts(("phish", "陌生域名，不是银行官网"), ("official", "银行官网"), ("store", "用户常用商店")), "可以点开看看再决定吗？", "no", YESNO, "点击链接的风险有多高？", 2, RISK),
        pack("lookup", "短信在索要验证码吗？", True, "短信只是通知、不要任何信息吗？", False, "短信想要什么？", "code", opts(("code", "短信验证码，以及后面提到的银行卡密码"), ("nothing", "不要信息"), ("address", "只要公开的门店地址")), "验证码可以告诉发件人吗？", "no", YESNO, "泄露验证码的风险有多高？", 2, RISK),
        pack("lookup", "短信还索要银行卡密码吗？", True, "银行会用短信要密码吗？", False, "索要密码说明什么？", "phish", opts(("phish", "钓鱼，银行不会这样要密码"), ("normal", "正常风控"), ("joke", "朋友开玩笑")), "应该把密码回复过去吗？", "no", YESNO, "密码风险有多高？", 2, RISK),
        pack("lookup", "用户今天有在途包裹吗？", False, "存在需要清关的包裹吗？", False, "包裹理由成立吗？", "no", opts(("no", "没有在途包裹，清关理由不成立"), ("yes", "确实有包裹被扣"), ("unknown", "概况没提包裹")), "可以为了取包裹去填验证码吗？", "no", YESNO, "被包裹理由说服的风险有多高？", 2, RISK),
        pack("lookup", "用户的银行是明德银行吗？", True, "短信链接代表明德银行吗？", False, "要查通知应该去哪里？", "app", opts(("app", "明德银行官方 App"), ("link", "短信里的链接"), ("search", "搜索短信里的短域名")), "可以拨打短信里留的电话吗？", "no", YESNO, "走错渠道的风险有多高？", 2, RISK),
        pack("lookup", "用户已经点击链接了吗？", False, "还来得及避免输入信息吗？", True, "当前进度属于哪一种？", "unopened", opts(("unopened", "还没点击"), ("typed", "已经输入密码"), ("paid", "已经转账")), "需要立刻冻结因为已经中招了吗？", "no", YESNO, "因为还没点击，紧急处理是阻止点击吗？", 2),
        pack("judgment", "这条短信应该当成钓鱼吗？", True, "这条短信应该当成银行官方通知吗？", False, "定性选哪一个？", "phish", opts(("phish", "非官方号、陌生域名、索要验证码和密码"), ("bank", "官方扣款通知"), ("parcel", "真实包裹短信")), "单一可疑点就够定性，还是这里有多项？", "many", opts(("many", "短号、域名、验证码、密码、无包裹，多项同时成立"), ("one", "只有错别字"), ("zero", "没有可疑点")), "钓鱼定性的置信风险有多高？", 2, RISK),
        pack("judgment", "规则要求不要点击吗？", True, "规则允许先点开链接检查域名吗？", False, "对链接的动作是哪个？", "avoid", opts(("avoid", "不点击"), ("open", "点开看看"), ("forward", "转发给家人一起点")), "不点击就无法确认，所以应该点吗？", "no", YESNO, "阻止点击有多急？", 2),
        pack("judgment", "应该回复这条短信吗？", False, "回复“停”或“退订”也算回复吗？", True, "回复策略选哪个？", "silent", opts(("silent", "不要回短信"), ("stop", "回复退订"), ("code", "先把验证码发过去再拉黑")), "拉黑或上报要不要先回一条？", "no", YESNO, "回复短信的风险有多高？", 2, RISK),
        pack("lookup", "没有包裹却用包裹当理由，这是虚假由头吗？", True, "用户忘了一个国际包裹吗？", False, "包裹说法和事实哪一个为准？", "fact", opts(("fact", "以没有在途包裹为准"), ("sms", "以短信为准"), ("seller", "先给陌生链接付款")), "要去链接里填单号吗？", "no", YESNO, "虚假包裹通知的风险有多高？", 2, RISK),
        pack("lookup", "域名 example-parcel.cc 看起来像包裹站，但是不是银行域名吗？", True, "看起来像就足够信任吗？", False, "域名判断应该看什么？", "owner", opts(("owner", "是不是明德银行官网，而不是像不像"), ("looks", "看起来像官方就可以"), ("https", "只要有锁图标")), "类似名称能代替官方域名吗？", "no", YESNO, "域名混淆的风险有多高？", 2, RISK),
        pack("judgment", "应该到官方 App 查看，而不是用短信链接吗？", True, "官方 App 和短信链接是同一条渠道吗？", False, "查账渠道选哪个？", "app", opts(("app", "打开已安装的明德银行 App"), ("link", "点短信链接"), ("new", "按短信指示安装新 App")), "App 里如果没有这条通知，说明短信不可信吗？", "yes", YESNO, "改用官方 App 有多急？", 2),
        pack("lookup", "可以把短信上报为垃圾短信吗？", True, "上报之前必须先点击确认内容吗？", False, "上报方式属于哪一种？", "system", opts(("system", "用手机系统的垃圾短信上报，不点链接"), ("reply", "回复举报二字"), ("call", "打短信里的电话举报")), "上报是规则允许的吗？", "yes", YESNO, "上报这件事有多急？", 1),
        pack("lookup", "验证码和密码都出现在要求里吗？", True, "只是要一个收件人姓名吗？", False, "敏感信息请求有几类？", "two", opts(("two", "验证码和银行卡密码两类"), ("one", "只要姓名"), ("zero", "不要信息")), "给出其中任意一个可以吗？", "no", YESNO, "信息请求的风险有多高？", 2, RISK),
        pack("judgment", "还没点击，所以不需要因为已经泄露而冻结卡片吗？", True, "即便没点击也应该把密码发给对方核对吗？", False, "账户保护动作选哪个？", "watch", opts(("watch", "不点击、不回复，改用官方 App 查看；不是已经泄露后的紧急挂失"), ("send", "先把密码发给对方验证是不是银行"), ("freeze", "短信一来就在链接里冻结")), "保护动作要在短信提供的页面里做吗？", "no", YESNO, "保护动作有多急？", 2),
        pack("lookup", "短号 1069 加陌生域名已经满足钓鱼规则的两项吗？", True, "必须等钱被转走才算钓鱼吗？", False, "规则的触发条件属于哪一种？", "before", opts(("before", "看到非官方号和索要密码就可以定性，不用等损失"), ("after", "等发生盗刷"), ("never", "短信永远不算钓鱼")), "现在已经满足多项条件吗？", "yes", YESNO, "等待损失发生的风险有多高？", 2, RISK),
        pack("lookup", "用户还没有把验证码填进去吗？", True, "已经在链接里输入了密码吗？", False, "损失状态属于哪一种？", "none", opts(("none", "尚未点击，未见已泄露"), ("leaked", "密码已提交"), ("gone", "钱已转走")), "要按已盗刷流程报警吗？", "no", YESNO, "误判已盗刷的风险有多高？", 1, RISK),
        pack("judgment", "家人如果也收到同样的短信，应该告诉他们不要点吗？", True, "应该把链接转发给家人让他们帮忙看看吗？", False, "对家人的建议是哪一种？", "warn", opts(("warn", "提醒不要点击、不要回验证码"), ("forward", "转发链接让他们打开"), ("ignore", "完全不用提")), "转发链接本身安全吗？", "no", YESNO, "扩散链接的风险有多高？", 2, RISK),
        pack("judgment", "按规则应该不点、不回、不上短信里的网站，改用官方 App 并上报垃圾短信吗？", True, "按规则应该点击链接并填写验证码和密码吗？", False, "最终处理选哪一个？", "block", opts(("block", "不点不回，官方 App 核查，上报垃圾短信"), ("comply", "按短信填写验证码和密码"), ("pay", "先给链接里的账户转一笔解冻费")), "最终处理会点击短信链接吗？", "no", YESNO, "处理这条短信有多急？", 2),
    ]
    return scenario("sms", "可疑短信", state, facts)


def dinner() -> dict[str, Any]:
    state = {
        "概况": (
            "周六晚上有 4 位客人来家里吃饭，加上用户自己是 5 人。今天是周四。牛奶昨天过期。鸡肉是今天买的，还在保质期内。"
            "米够 5 人吃。酱油只剩一点点，不够做这顿。冰箱里没有蔬菜。客人里有一位不吃香菜。"
            "没有人声明花生过敏。用户打算做鸡肉和米饭，还没买菜。"
            "规则：过期牛奶倒掉，不能用来待客。当天的鸡肉可以使用。酱油不够就要补买，不能假设够用。"
            "菜单要有蔬菜。避开香菜。没有花生过敏声明就不必做无花生厨房，但不要主动做花生酱。周六前把缺的东西买齐。"
        ),
        "记录": {
            "客人": 4,
            "总人数": 5,
            "今天": "周四",
            "聚餐": "周六晚上",
            "牛奶过期": True,
            "鸡肉当天": True,
            "米够": True,
            "酱油够": False,
            "有蔬菜": False,
            "有人不吃香菜": True,
            "花生过敏": False,
        },
        "_must": ["4 位客人", "牛奶昨天过期", "酱油只剩一点点", "不吃香菜"],
    }
    facts = [
        pack("lookup", "周六晚上有 4 位客人吗？", True, "只有用户一个人吃吗？", False, "用餐人数是多少？", "five", opts(("five", "4 位客人加用户，共 5 人"), ("one", "1 人"), ("ten", "10 人")), "要按 5 人的量准备吗？", "yes", YESNO, "人数确认有多急？", 0),
        pack("lookup", "今天是周四、聚餐在周六吗？", True, "客人今晚就到吗？", False, "采购窗口属于哪一种？", "two_days", opts(("two_days", "还有周四和周五，周六晚上才用餐"), ("now", "一小时后就用餐"), ("nextmonth", "下个月")), "必须今晚宴请吗？", "no", YESNO, "时间还够买菜吗？", 1),
        pack("lookup", "牛奶昨天过期了吗？", True, "过期牛奶还能用来待客吗？", False, "牛奶怎么处理？", "discard", opts(("discard", "倒掉，不待客"), ("serve", "做给客人喝"), ("keep", "继续放着当新鲜奶")), "过期食品可以留给自己、新鲜的留给客人吗？", "no", YESNO, "使用过期牛奶的风险有多高？", 2, RISK),
        pack("lookup", "鸡肉是今天买的吗？", True, "鸡肉已经过期了吗？", False, "鸡肉能不能用？", "yes", opts(("yes", "当天购买，在保质期内，可以使用"), ("no", "必须扔掉"), ("freeze", "已经不适合做熟")), "鸡肉要不要因为牛奶过期一起扔掉？", "no", YESNO, "鸡肉安全性有多急？", 0),
        pack("lookup", "米够 5 个人吃吗？", True, "需要因为米饭不够而加买米吗？", False, "主食怎么处理？", "enough", opts(("enough", "米够，不用加买"), ("buy", "米不够，要买"), ("noodle", "没有米，改面")), "米饭可以作为主食吗？", "yes", YESNO, "主食问题有多急？", 0),
        pack("judgment", "酱油不够这顿用吗？", True, "可以假设剩下的酱油够 5 个人吗？", False, "酱油怎么处理？", "buy", opts(("buy", "不够，需要补买"), ("skip", "够用，不买"), ("make", "客人自己带酱油")), "酱油属于采购清单吗？", "yes", YESNO, "缺调味料的影响有多大？", 1),
        pack("lookup", "冰箱里有蔬菜吗？", False, "菜单需要蔬菜吗？", True, "蔬菜怎么处理？", "buy", opts(("buy", "没有蔬菜，要买，菜单里要有蔬菜"), ("skip", "不吃蔬菜"), ("can", "打开过期罐头代替")), "只做鸡肉和米饭符合规则吗？", "no", YESNO, "菜单缺蔬菜的风险有多高？", 1, RISK),
        pack("lookup", "有客人不吃香菜吗？", True, "可以在公共菜里放香菜吗？", False, "香菜怎么处理？", "avoid", opts(("avoid", "菜单避开香菜"), ("add", "每道菜都放"), ("side", "只给不吃的人单独一盘香菜")), "这是过敏还是口味回避？", "taste", opts(("taste", "不吃香菜，概况没说香菜过敏"), ("allergy", "香菜过敏"), ("none", "所有人都爱吃香菜")), "忽略口味的风险有多高？", 1, RISK),
        pack("lookup", "有人声明花生过敏吗？", False, "应该把厨房改成无花生车间吗？", False, "花生酱怎么处理？", "skip", opts(("skip", "没有过敏声明，不做花生酱，也不必无花生车间"), ("ban", "拆掉所有含花生的调料"), ("serve", "做花生酱凉菜")), "可以主动做一道花生菜吗？", "no", YESNO, "花生风险有多高？", 1, RISK),
        pack("judgment", "过期牛奶和当天鸡肉要区别处理吗？", True, "两种蛋白质都该扔掉吗？", False, "蛋白质食材选哪个？", "chicken", opts(("chicken", "用鸡肉，倒掉牛奶"), ("milk", "用过期牛奶，扔掉鸡肉"), ("none", "都不用")), "区别的依据是保质期吗？", "yes", YESNO, "食材判错的风险有多高？", 2, RISK),
        pack("lookup", "用户还没买菜吗？", True, "菜已经买齐了吗？", False, "采购状态属于哪一种？", "todo", opts(("todo", "还没买，清单要在周六前完成"), ("done", "已经买齐"), ("delivery", "客人会带全部食材")), "采购要包含酱油和蔬菜吗？", "yes", YESNO, "采购有多急？", 1),
        pack("lookup", "总人数是 5 吗？", True, "可以按 2 人的量买菜吗？", False, "采购分量按谁算？", "five", opts(("five", "5 人"), ("guests", "只算 4 位客人，不算用户"), ("one", "只算用户")), "少算人数会不够吃吗？", "yes", YESNO, "分量错误的风险有多高？", 1, RISK),
        pack("judgment", "菜单应是鸡肉、米饭和蔬菜，并且没有香菜吗？", True, "菜单应是过期牛奶和香菜沙拉吗？", False, "主菜单选哪一套？", "meal", opts(("meal", "鸡肉、米饭、蔬菜，不加香菜"), ("milk", "过期牛奶配香菜"), ("peanut", "花生酱拌面")), "米饭需要另买吗？", "no", YESNO, "定菜单有多急？", 1),
        pack("lookup", "周六前买齐就来得及吗？", True, "必须周四凌晨出门买菜吗？", False, "最迟采购时间属于哪一档？", "before", opts(("before", "周六聚餐前"), ("month", "下个月"), ("after", "客人走了再买")), "周四知道缺货，还算及时吗？", "yes", YESNO, "采购时限有多急？", 1),
        pack("lookup", "酱油剩下的量不够做这顿吗？", True, "酱油和米都不够吗？", False, "调味料和主食哪一个缺？", "sauce", opts(("sauce", "酱油不够，米够"), ("rice", "米不够，酱油够"), ("both", "都不够")), "可以只买米不买酱油吗？", "no", YESNO, "买错品类的风险有多高？", 1, RISK),
        pack("judgment", "不吃香菜的人仍能吃没有香菜的鸡肉和蔬菜吗？", True, "必须给每个人做完全不同的菜单吗？", False, "口味限制怎么满足？", "omit", opts(("omit", "公共菜不加香菜即可"), ("separate", "做 5 套完全不同的菜"), ("cancel", "取消请客")), "这个限制和花生过敏一样严重吗？", "no", YESNO, "口味处理有多急？", 1),
        pack("lookup", "没有花生过敏声明，所以不必清空花生制品吗？", True, "概况写了有人花生过敏吗？", False, "过敏信息属于哪一种？", "none", opts(("none", "没有人声明花生过敏"), ("peanut", "有花生过敏"), ("all", "所有食物都过敏")), "要按过敏聚餐的最高级别准备吗？", "no", YESNO, "过度准备的必要度有多高？", 0),
        pack("judgment", "过期牛奶不能留给客人，也不能当成这顿的食材吗？", True, "过期一天的牛奶对客人是安全的吗？", False, "牛奶的最终动作是哪个？", "discard", opts(("discard", "倒掉"), ("cook", "煮开给客人"), ("coffee", "做咖啡招待")), "牛奶过期和鸡肉的保质期一样吗？", "no", YESNO, "错误使用牛奶的风险有多高？", 2, RISK),
        pack("lookup", "客人周六晚上才到，所以周四的任务是采购而不是开饭吗？", True, "周四晚上就要上菜吗？", False, "周四应该做什么？", "shop", opts(("shop", "列清单并买齐酱油和蔬菜"), ("serve", "周四晚上开席"), ("nothing", "什么都不用做")), "鸡肉现在就要煮掉吗？", "no", YESNO, "周四行动有多急？", 1),
        pack("judgment", "按规则应该倒掉牛奶、使用鸡肉和现有的米、补买酱油和蔬菜、菜里不加香菜吗？", True, "按规则应该用过期牛奶待客并做花生酱香菜菜吗？", False, "最终备餐方案选哪一个？", "plan", opts(("plan", "倒掉牛奶，鸡肉配米饭和蔬菜，补买酱油和蔬菜，不加香菜"), ("bad", "过期牛奶、香菜和花生酱"), ("order", "全部取消，不招待")), "最终菜单会使用过期牛奶吗？", "no", YESNO, "备餐这件事有多急？", 1),
    ]
    return scenario("dinner", "周末待客", state, facts)


def incident() -> dict[str, Any]:
    state = {
        "概况": (
            "支付接口错误率 6%，平时低于 1%。二十分钟前有一次支付服务部署。错误集中在支付路径，浏览商品正常。"
            "可以回滚到上一个版本。没有发现数据丢失。值班是后端组的周衡，不是前端。"
            "状态页还没更新。客服已经接到付款失败的来电。没有安全入侵告警。"
            "规则：错误率明显高于基线且紧跟本服务部署时，先回滚支付服务，不要先重写前端。"
            "没有数据丢失就不要启动数据恢复。状态页要写支付受影响、商品浏览正常，不要写成全站宕机。"
            "由后端值班执行回滚。有客服来电就要给客服一句可用口径。没有入侵告警就不要按安全事件升级。"
        ),
        "记录": {
            "错误率": 6,
            "基线错误率": 1,
            "部署分钟前": 20,
            "只影响支付": True,
            "浏览正常": True,
            "可回滚": True,
            "数据丢失": False,
            "值班": "后端周衡",
            "状态页已更新": False,
            "客服来电": True,
            "入侵告警": False,
        },
        "_must": ["错误率 6%", "二十分钟前", "可以回滚", "没有发现数据丢失"],
    }
    facts = [
        pack("lookup", "支付错误率明显高于平时吗？", True, "错误率仍低于 1% 的基线吗？", False, "错误率属于哪一档？", "high", opts(("high", "6%，基线低于 1%"), ("normal", "低于基线"), ("zero", "没有错误")), "这是需要处理的异常吗？", "yes", YESNO, "错误率异常有多急？", 2),
        pack("lookup", "二十分钟前刚部署了支付服务吗？", True, "最近一天都没有部署吗？", False, "时间和部署的关系是哪一种？", "after", opts(("after", "部署后 20 分钟内错误升高"), ("unrelated", "和部署无关"), ("week", "上周的部署")), "部署是可疑变更吗？", "yes", YESNO, "变更关联有多急？", 2),
        pack("lookup", "错误集中在支付路径吗？", True, "用户连商品都刷不出来吗？", False, "影响面属于哪一种？", "pay", opts(("pay", "支付失败，浏览商品正常"), ("all", "全站打不开"), ("none", "没有用户影响")), "应该说全站宕机吗？", "no", YESNO, "影响面判断有多急？", 2),
        pack("lookup", "可以回滚到上一个版本吗？", True, "没有可回滚的版本吗？", False, "首选技术动作是哪个？", "rollback", opts(("rollback", "回滚支付服务"), ("rewrite", "先重写前端"), ("wait", "什么都不做等它自己好")), "规则要求先回滚吗？", "yes", YESNO, "回滚决策有多急？", 2),
        pack("lookup", "发现数据丢失了吗？", False, "应该启动数据恢复吗？", False, "数据动作选哪个？", "none", opts(("none", "没有数据丢失，不启动恢复"), ("restore", "从备份全量恢复"), ("delete", "清空支付表")), "回滚会代替数据恢复吗？", "yes", YESNO, "误启动数据恢复的风险有多高？", 2, RISK),
        pack("lookup", "值班是后端的周衡吗？", True, "应该把回滚交给前端吗？", False, "谁执行回滚？", "backend", opts(("backend", "后端值班周衡"), ("frontend", "前端"), ("sales", "销售")), "需要换一个不值班的人来做吗？", "no", YESNO, "找错执行人的风险有多高？", 2, RISK),
        pack("lookup", "状态页还没更新吗？", True, "状态页已经写了全站宕机吗？", False, "状态页应该怎么写？", "partial", opts(("partial", "支付受影响，商品浏览正常"), ("down", "全站宕机"), ("ok", "一切正常")), "现在更新状态页吗？", "yes", YESNO, "状态页有多急？", 1),
        pack("lookup", "客服接到付款失败的来电了吗？", True, "客服还没有任何客户联系吗？", False, "给客服的口径应该是哪一种？", "line", opts(("line", "支付暂时失败，浏览正常，正在回滚"), ("fine", "让客户反复重试直到成功"), ("hack", "告诉客户被盗号了")), "需要给客服口径吗？", "yes", YESNO, "客服口径有多急？", 2),
        pack("lookup", "有安全入侵告警吗？", False, "应该按安全事件升级吗？", False, "事件类型属于哪一种？", "change", opts(("change", "部署后的支付故障，不是入侵"), ("breach", "安全入侵"), ("drill", "计划中的演练")), "要先隔离全部服务器吗？", "no", YESNO, "误报安全事件的风险有多高？", 1, RISK),
        pack("judgment", "错误升高紧跟支付部署，所以先回滚支付服务吗？", True, "应该先重写前端吗？", False, "第一技术步骤选哪个？", "rollback", opts(("rollback", "回滚刚部署的支付服务"), ("frontend", "重写前端"), ("db", "因为没丢数据也先做一次全量恢复")), "浏览正常会改变回滚决定吗？", "no", YESNO, "第一步骤有多急？", 2),
        pack("judgment", "状态页必须避免写成全站宕机吗？", True, "浏览正常就可以把状态页保持空白吗？", False, "状态页的范围怎么写？", "pay", opts(("pay", "只写支付受影响、浏览正常"), ("site", "写全站宕机"), ("blank", "先不更新")), "空白状态页符合已经有客服来电的情况吗？", "no", YESNO, "写错状态页的风险有多高？", 2, RISK),
        pack("lookup", "平时基线低于 1% 吗？", True, "6% 可以解释成正常波动吗？", False, "和基线比，6% 说明什么？", "incident", opts(("incident", "明显高于基线，是故障"), ("noise", "正常噪声"), ("better", "比平时更好")), "需要看基线再行动吗？", "yes", YESNO, "基线比较有多急？", 1),
        pack("lookup", "部署的是支付服务而不是首页吗？", True, "应该回滚一个没改过的前端吗？", False, "回滚对象是哪个？", "pay", opts(("pay", "二十分钟前的支付服务"), ("home", "没改动的首页"), ("all", "所有历史版本一起删掉")), "回滚范围要和错误路径一致吗？", "yes", YESNO, "回滚错服务的风险有多高？", 2, RISK),
        pack("judgment", "没有数据丢失，所以不做恢复，只做回滚吗？", True, "没有数据丢失反而更应该全量恢复吗？", False, "回滚和恢复的关系是哪一种？", "rollback_only", opts(("rollback_only", "回滚，不启动数据恢复"), ("restore", "先恢复数据再观察"), ("both", "恢复数据并重写前端")), "数据丢失是启动恢复的前提吗？", "yes", YESNO, "数据动作的风险有多高？", 2, RISK),
        pack("lookup", "客服需要的是一句现在就能用的口径吗？", True, "要等故障完全结束才告诉客服吗？", False, "口径的时点属于哪一种？", "now", opts(("now", "现在就给，说明正在回滚"), ("later", "明天再给"), ("never", "不告诉客服")), "口径里要提到浏览仍正常吗？", "yes", YESNO, "口径时效有多急？", 2),
        pack("lookup", "后端值班在，所以不用把指挥权交给销售吗？", True, "周衡是前端负责人吗？", False, "指挥和执行落在哪一组？", "backend", opts(("backend", "后端值班"), ("frontend", "前端"), ("support", "让客服自己改配置")), "客服负责执行回滚吗？", "no", YESNO, "职责分清有多急？", 1),
        pack("judgment", "没有入侵告警，就不要按安全事件升级吗？", True, "支付失败本身就证明被入侵了吗？", False, "升级路径选哪个？", "ops", opts(("ops", "按部署故障处理，不升安全事件"), ("sec", "按入侵升级"), ("pr", "先开新闻发布会")), "安全升级的前提是入侵告警吗？", "yes", YESNO, "升错级的风险有多高？", 1, RISK),
        pack("lookup", "商品浏览正常能帮用户区分影响面吗？", True, "浏览正常表示支付也正常吗？", False, "对用户的影响描述哪句对？", "split", opts(("split", "可以浏览，支付会失败"), ("dead", "网站完全打不开"), ("ok", "支付也正常")), "客服可以建议用户先浏览、稍后再付款吗？", "yes", YESNO, "描述影响面有多急？", 1),
        pack("judgment", "二十分钟这个间隔足够把部署当成原因吗？", True, "应该忽略时间关系先重做前端吗？", False, "因果判断选哪个？", "deploy", opts(("deploy", "错误紧跟支付部署，先回滚该部署"), ("frontend", "当成前端文案问题"), ("random", "当成随机波动不处理")), "规则把这个时间关系写成了行动条件吗？", "yes", YESNO, "因果判断有多急？", 2),
        pack("judgment", "按规则应该由后端回滚支付、更新部分影响的状态页、给客服口径，并且不做数据恢复和安全升级吗？", True, "按规则应该宣布全站宕机、重写前端并启动数据恢复吗？", False, "最终故障处理选哪一个？", "plan", opts(("plan", "后端回滚支付，状态页写部分影响，通知客服，不恢复数据，不升安全"), ("panic", "全站宕机、重写前端、全量恢复"), ("ignore", "保持观察，不回滚也不通知")), "最终处理会宣布全站宕机吗？", "no", YESNO, "这场故障处理有多急？", 2),
    ]
    return scenario("incident", "支付接口故障", state, facts)


SCENARIOS = [
    support(),
    home(),
    expense(),
    commute(),
    clinic(),
    shop(),
    calendar(),
    sms(),
    dinner(),
    incident(),
]

