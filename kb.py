# -*- coding: utf-8 -*-
r"""判决书切分与标签抽取（纯函数，可离线自测）。

四条实测得出的关键约束，别改：

1. **必须先截落款再抽法条**。25.2% 的判决书在落款之后附了「附：法律条文《…》第…条 …」，
   不截的话每份都"引"了七八条，全是附录里的。

2. **推理锚点不能只用 `find("本院认为")`**。1.9% 的文书首选措辞是
   「本院经审查认为」「法院认为」「本院经审理认为」，而它们**后面**又会出现
   「本院认为」。只找「本院认为」会把前一段论证错划进案情段。
   要取 `本院认为|本院经(审理|审查)认为|法院认为` 里**位置最靠前**的那个。

3. **判决主文要同时按 `一、二、…` 和 `。；` 切项**。29% 的主文完全不用编号，
   是连续句子；不切句会把「给付…。驳回…」整体当成驳回而误判方向。

4. **案由用「后缀闭包」抽取，不用正则硬切**。正则 `([一-鿿]{2,20}纠纷)一案`
   命中率看着高（95.5%），但其中 31.8% 带出人名/公司名
   （`汕富物业有限公司诉被告隋学礼物业服务合同纠纷`）。改用数据驱动的后缀闭包后
   0 噪声、99% 命中。
"""
import re
from collections import Counter

# ---- 切分锚点 ----
REASON_RE = re.compile(r"本院认为|本院经(?:审理|审查)认为|法院认为")
DISP_RE = re.compile(r"判决如下|判决：")
# 落款：审判长/审判员/人民陪审员/代理审判员/助理审判员/独任审判员/（代）书记员
SIGN_RE = re.compile(
    r"审\s*判\s*长|审\s*判\s*员|人民陪审员|代理审判员|助理审判员|独任审判员|代?\s*书\s*记\s*员"
)
# 主文后面的程序性尾句，不属于判决主文。
# ⚠ 别加 `本判决生效后`——标准给付句式就是「被告X于本判决生效后十日内给付原告Y…」，
#   加了会在第一项就命中，把后面全切掉（连"驳回其他诉讼请求"一起），
#   判决方向会全部误判成"支持"。实测：多加这一条，支持率从 44.8% 飙到 56.1%。
# 也别加裸 `如果未` 和 `加倍支付`，它们只出现在 `如果未按…` 这个完整尾句里。
PROC_RE = re.compile(
    r"如果未按|案件受理费|诉讼费|如不服本判决|本判决为终审|"
    r"上述债务|权利人可|保全费由|鉴定费由"
)

# ---- 法条 ----
LAW_RE = re.compile(r"《[^》]{2,45}》第[一二三四五六七八九十百千零〇两\d]+条")
LAW_ANCHOR_RE = re.compile(r"(?:依照|依据|根据|按照|参照)[^。；]{0,500}?之规定")

# ---- 主文分项 ----
ITEM_SPLIT_RE = re.compile(r"(?<![一二三四五六七八九十])[一二三四五六七八九十]{1,3}、|(?<=。)|(?<=；)")


# ==================================================================
#  切分
# ==================================================================
def split_judgment(text):
    """把判决书全文切成 (case_text, reasoning, verdict)。

    case_text  = 「本院认为」之前（= API 的输入形态）
    reasoning  = 推理段（法条在这里）
    verdict    = 判决主文（落款之前、程序性尾句之前）
    """
    if not text:
        return "", "", ""
    t = text

    # 1) 推理锚点：取最先出现的那个变体
    best = None
    for m in REASON_RE.finditer(t):
        if best is None or m.start() < best.start():
            best = m
    if best is None:
        return t, "", ""
    case_text = t[:best.start()]
    rest = t[best.start():]

    # 2) 判决主文起点
    d = DISP_RE.search(rest)
    if not d:
        return case_text, rest, ""
    reasoning = rest[:d.start()]
    after = rest[d.end():]

    # 3) 截掉落款（及落款之后的「附：法律条文」附录）
    s = SIGN_RE.search(after)
    if s:
        after = after[:s.start()]

    # 4) 截掉程序性尾句
    p = PROC_RE.search(after)
    if p and p.start() > 8:          # 太靠前就不截，免得把主文切没
        after = after[:p.start()]

    verdict = after.strip().strip("：: 。")
    return case_text, reasoning, verdict


# ==================================================================
#  案由：后缀闭包词表
# ==================================================================
NOISE_RE = re.compile(r"[）)（(]|公司|有限|某|[\d年月日]")
CAUSE_TAIL_RE = re.compile(r"(.{0,28}?)(?:一案|起诉|诉至)")


def _candidates(case_text):
    """从案情段里取「一案」之前的窗口，抽出以「纠纷」结尾（或「劳动争议」）的候选后缀。"""
    out = []
    for m in CAUSE_TAIL_RE.finditer(case_text[:1200]):
        w = m.group(1)
        # 以「纠纷」结尾的后缀，长度 4–20
        for L in range(4, min(21, len(w) + 1)):
            s = w[-L:]
            if s.endswith("纠纷") and not NOISE_RE.search(s):
                out.append(s)
        if w.endswith("劳动争议"):
            out.append("劳动争议")
    return out


def build_cause_vocab(case_texts, min_df=8):
    """数据驱动的规范案由词表。

    后缀 S 是「完整案由」当且仅当不存在单字 c 使 `c+S` 的 df ≥ 0.8·df(S)——
    这样 `服务合同纠纷` 会被判为可扩展（真的完整形式是 `物业服务合同纠纷`），
    而 `离婚纠纷` 不含可扩展前缀，保持完整。
    """
    df = Counter()
    for ct in case_texts:
        for s in set(_candidates(ct)):
            df[s] += 1
    vocab = []
    for s, c in df.items():
        if c < min_df:
            continue
        # (a) 看有没有「c + s」这种更长后缀也常见（说明 s 还能向左扩展）
        extendable = False
        for s2, c2 in df.items():
            if len(s2) == len(s) + 1 and s2.endswith(s) and c2 >= 0.8 * c:
                extendable = True
                break
        if extendable:
            continue
        # (b) 反方向：如果**去掉左边一个字**之后 df 反而更高，说明它前面粘了个残片。
        #     单字扩展查不出这种情况，因为左边一字各不相同——
        #     `司劳动争议纠纷`(79) 左边可能是 险公司/支公司/限公司，没有哪个单字占 80%，
        #     但去掉「司」得到的 `劳动争议纠纷`(209) 更高 → 判为噪声。
        #     ⚠ 只比对 len-1 的那一个，不要比对所有更短后缀：
        #       `合同纠纷`(1326) 是 `物业服务合同纠纷`(406) 的后缀且更高，
        #       但去掉一个字的 `业服务合同纠纷` 极低——按长后缀比会误杀具体案由。
        noisy = False
        if len(s) >= 5:
            shorter = s[1:]
            c2 = df.get(shorter, 0)
            # ⚠ 两点都不能少：
            # 1) 必须用**严格大于**而不是 >=。`_candidates` 是窗口的所有后缀，
            #    完整案由和它的左缩短形式（`物业服务合同纠纷` / `业服务合同纠纷`）
            #    由同一批窗口生成，df 天然相等。用 >= 会把所有具体案由误判成噪声
            #    （实测：只剩 合同纠纷/责任纠纷 这类泛称）。
            # 2) 必须带**容差**。`融借款合同纠纷`(399) 比 `金融借款合同纠纷`(398)
            #    只多 1 份（有文书只写「融借款」），严格 > 会把后者砍掉，
            #    约 400 份案子被归到泛称「合同纠纷」。给 15% 容差即可。
            if c2 > c * 1.15:
                noisy = True
        if not noisy:
            vocab.append(s)
    # 长的优先匹配
    vocab.sort(key=len, reverse=True)
    return vocab


def extract_cause(case_text, vocab):
    """在案情段里找最长的、能匹配到「一案/起诉/诉至」之前的规范案由。"""
    for m in CAUSE_TAIL_RE.finditer(case_text[:1200]):
        w = m.group(1)
        for v in vocab:                     # vocab 已按长度降序
            if w.endswith(v):
                return v
    # 退一步：整段里出现过的规范案由（取最长）
    for v in vocab:
        if v in case_text:
            return v
    return None


# ==================================================================
#  法条
# ==================================================================
def _norm_law(s):
    return re.sub(r"\s+", "", s)


def extract_laws(reasoning):
    """从推理段抽法条。锚定「依照/依据/根据 … 之规定」这一句（3,520 份里 100% 存在）。"""
    if not reasoning:
        return []
    pool = []
    for m in LAW_ANCHOR_RE.finditer(reasoning):
        pool += LAW_RE.findall(m.group(0))
    if not pool:                            # 锚点句没找到就退而求其次，扫全段
        pool = LAW_RE.findall(reasoning)
    seen, out = set(), []
    for s in pool:
        s = _norm_law(s)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ==================================================================
#  判决方向
# ==================================================================
def verdict_items(verdict):
    """把主文切成小项。按 `一、二、…` 切，同时按 `。；` 切——29% 的主文没有编号。"""
    if not verdict:
        return []
    parts = re.split(r"[一二三四五六七八九十]{1,3}、", verdict)
    items = []
    for p in parts:
        for q in re.split(r"[。；;]", p):
            q = q.strip()
            if q:
                items.append(q)
    return items


def _is_dismiss_of_plaintiff(item):
    """这一项是不是「驳回原告」。判给被告的驳回（反诉）不算。"""
    if "不准予" in item and "离婚" in item:
        return True
    if "不准" in item and "离婚" in item:
        return True
    if "驳回" not in item:
        return False
    # 驳回被告 / 驳回反诉原告 是对原告有利的，不算驳回原告
    if re.search(r"驳回(被告|反诉原告|反诉被告)", item):
        return False
    return True


def derive_direction(verdict):
    """支持 / 部分支持 / 驳回。"""
    items = verdict_items(verdict)
    if not items:
        return None
    has_dismiss = any(_is_dismiss_of_plaintiff(i) for i in items)
    has_grant = any(not _is_dismiss_of_plaintiff(i) for i in items)
    if has_dismiss and has_grant:
        return "部分支持"
    if has_dismiss:
        return "驳回"
    return "支持"
