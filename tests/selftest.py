# -*- coding: utf-8 -*-
r"""离线自测：切分、案由抽取、法条抽取、判决方向。

跑：python tests/selftest.py
预期能复现数据勘察时的数字（100% 切分、25 个规范案由、0 噪声、方向 44.8/48.7/6.5）。
"""
import io
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import kb

DATA = os.environ.get(
    "JUDGE_DATA",
    r"D:\BaiduNetdiskDownload\作业4_判决书\judge_data\judgments.jsonl")


def load():
    rows = []
    with io.open(DATA, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    rows = load()
    n = len(rows)
    print(f"载入 {n} 份判决书")
    print("=" * 74)

    ok = fail = 0
    no_reason = no_disp = no_verdict = no_laws = 0
    appendix_leak = 0
    case_texts, dirs, law_counts = [], Counter(), []
    causes = Counter()

    for r in rows:
        t = r["text"]
        ct, reasoning, verdict = kb.split_judgment(t)
        if not ct or not reasoning:
            no_reason += 1
            fail += 1
            continue
        if not verdict:
            no_verdict += 1
        # 附录泄漏检查：落款之后的内容不该进 verdict
        if "附：" in verdict or "附:" in verdict:
            appendix_leak += 1
        laws = kb.extract_laws(reasoning)
        if not laws:
            no_laws += 1
        law_counts.append(len(laws))
        d = kb.derive_direction(verdict)
        dirs[d or "(判不出)"] += 1
        case_texts.append(ct)
        ok += 1

    print("\n【1】切分")
    print(f"  成功切分: {ok}/{n}   失败: {fail}")
    print(f"  无推理段: {no_reason}   无主文: {no_verdict}")
    print(f"  主文里残留「附：」附录: {appendix_leak}  (应为 0)")

    print("\n【2】法条抽取")
    # 少数文书确实不引条文（引的是《…通知》而没有"第X条"，或是程序性不予审理），
    # 实测 4/3520 = 0.11%，属正常，只警告不判失败。API 侧法条为空时回退用先例的。
    print(f"  抽不到法条: {no_laws} / {n}  ({no_laws/max(n,1):.2%})"
          + ("  ← 注意" if no_laws > n * 0.01 else "  ✓ 正常"))
    law_counts.sort()
    if law_counts:
        print(f"  每份条数: min={law_counts[0]} P50={law_counts[len(law_counts)//2]} "
              f"max={law_counts[-1]}  均值={sum(law_counts)/len(law_counts):.1f}")

    print("\n【3】判决方向")
    tot = sum(dirs.values())
    for k in ("支持", "部分支持", "驳回", "(判不出)"):
        v = dirs.get(k, 0)
        print(f"  {k:6s} {v:5d}  {v/tot:6.1%}")
    print(f"  (勘察基准: 支持 44.8% / 部分支持 48.7% / 驳回 6.5% / 判不出 0)")

    print("\n【4】案由词表")
    vocab = kb.build_cause_vocab(case_texts)
    print(f"  规范案由数: {len(vocab)}")
    hit = 0
    for ct in case_texts:
        c = kb.extract_cause(ct, vocab)
        if c:
            hit += 1
            causes[c] += 1
    print(f"  命中率: {hit}/{len(case_texts)} = {hit/len(case_texts):.1%}")
    noise = [c for c in causes if re.search(r"公司|某|[\d）)]", c)]
    print(f"  带噪声的案由: {len(noise)}  {noise[:5]}")
    print("\n  Top 12 案由:")
    for k, v in causes.most_common(12):
        print(f"     {v:5d}  {k}")

    print("\n" + "=" * 74)
    bad = (fail or appendix_leak or len(noise) or dirs.get("(判不出)", 0)
           or no_laws > n * 0.01)
    print("自测结果: " + ("✓ 全部通过" if not bad else "✗ 有问题（见上）"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
