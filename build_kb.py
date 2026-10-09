# -*- coding: utf-8 -*-
r"""离线建库：judgments.jsonl → _derived/kb.jsonl + BM25 倒排索引。

跑：python build_kb.py
产物（都在 _derived/，已 gitignore）：
  kb.jsonl        3,520 条带标签的先例 {id, case_text, cause, laws, direction, verdict}
  bm25.npz        Okapi BM25 倒排（CSR）
  bm25_vocab.pkl  term -> tid
  cause_vocab.json 规范案由词表

BM25 用 array('i')/array('h') 累加而不是 Python list of tuple——后者几百万条
postings 要吃掉上 GB 的对象开销。这是从 rag/build_viz_index.py 学来的。
"""
import array
import io
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import kb

HERE = os.path.dirname(os.path.abspath(__file__))
DERIVED = os.path.join(HERE, "_derived")
DATA = os.environ.get(
    "JUDGE_DATA",
    r"D:\BaiduNetdiskDownload\作业4_判决书\judge_data\judgments.jsonl")

K1, B = 1.5, 0.75
MIN_TOKEN_LEN = 2
# 停用词：只看法律文书里高频但无区分度的虚词，别把「合同/借款/离婚」这类删掉
STOP = set("的 了 和 与 及 或 为 在 是 等 有 中 上 下 之 其 该 本 我 你 他 她 它 就 都 也 还 但".split())


def tokenize(text):
    import jieba
    out = []
    for t in jieba.lcut(text):
        t = t.strip().lower()
        if len(t) < MIN_TOKEN_LEN or t in STOP:
            continue
        if not any(c.isalnum() or ("一" <= c <= "鿿") for c in t):
            continue
        out.append(t)
    return out


def main():
    if not os.path.exists(DATA):
        print(f"[错误] 找不到数据文件：{DATA}")
        print("       可用环境变量 JUDGE_DATA 指定路径")
        return 1
    os.makedirs(DERIVED, exist_ok=True)

    t0 = time.time()
    rows = []
    with io.open(DATA, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[载入] {len(rows)} 份判决书  ({time.time()-t0:.1f}s)")

    # ---------- 1. 切分 + 打标 ----------
    recs = []
    for r in rows:
        ct, reasoning, verdict = kb.split_judgment(r["text"])
        recs.append({
            "id": r["id"],
            "case_text": ct,
            "reasoning": reasoning,
            "cause": None,                       # 下面统一填
            "laws": kb.extract_laws(reasoning),
            "direction": kb.derive_direction(verdict),
            "verdict": verdict,
        })

    # ---------- 2. 案由词表 ----------
    vocab_cause = kb.build_cause_vocab([r["case_text"] for r in recs])
    n_hit = 0
    cause_counts = Counter()
    for r in recs:
        c = kb.extract_cause(r["case_text"], vocab_cause)
        r["cause"] = c
        if c:
            n_hit += 1
            cause_counts[c] += 1
    print(f"[案由] 词表 {len(vocab_cause)} 个，命中 {n_hit}/{len(recs)} = {n_hit/len(recs):.1%}")

    # ---------- 3. BM25 倒排 ----------
    print("[bm25] 分词并建倒排 ...")
    vocab, post_rows, post_tfs = {}, {}, {}
    doc_len = np.zeros(len(recs), dtype=np.int32)
    n_tok = 0
    for i, r in enumerate(recs):
        toks = tokenize(r["case_text"])
        doc_len[i] = len(toks)
        n_tok += len(toks)
        if not toks:
            continue
        tf = Counter(toks)
        for t, n in tf.items():
            tid = vocab.get(t)
            if tid is None:
                tid = len(vocab)
                vocab[t] = tid
                post_rows[tid] = array.array("i")
                post_tfs[tid] = array.array("h")
            post_rows[tid].append(i)
            post_tfs[tid].append(min(n, 32767))

    V = len(vocab)
    offs = np.zeros(V + 1, dtype=np.int64)
    for tid in range(V):
        offs[tid + 1] = offs[tid] + len(post_rows[tid])
    total = int(offs[-1])
    postings = np.empty(total, dtype=np.int32)
    tfs = np.empty(total, dtype=np.int16)
    for tid in range(V):
        a, b = offs[tid], offs[tid + 1]
        postings[a:b] = np.frombuffer(post_rows[tid], dtype=np.int32)
        tfs[a:b] = np.frombuffer(post_tfs[tid], dtype=np.int16)
        post_rows[tid] = post_tfs[tid] = None
    del post_rows, post_tfs

    np.savez_compressed(os.path.join(DERIVED, "bm25.npz"),
                        postings=postings, tf=tfs, offsets=offs, doc_len=doc_len)
    with open(os.path.join(DERIVED, "bm25_vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f, protocol=4)

    # ---------- 4. 落盘 ----------
    with io.open(os.path.join(DERIVED, "kb.jsonl"), "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with io.open(os.path.join(DERIVED, "cause_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_cause, f, ensure_ascii=False)

    print(f"[bm25] 词表 {V} 词 / {n_tok} token / postings {total}")
    print(f"[落盘] → {DERIVED}")
    for fn in ("kb.jsonl", "bm25.npz", "bm25_vocab.pkl", "cause_vocab.json"):
        p = os.path.join(DERIVED, fn)
        print(f"        {fn:20s} {os.path.getsize(p)/1e6:.1f} MB")
    print(f"[完成] 总耗时 {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
