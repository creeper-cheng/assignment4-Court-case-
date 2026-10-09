# -*- coding: utf-8 -*-
r"""判案 API。标准库 HTTP，监听 127.0.0.1:8000。

  GET  /health                 → 200 {"ok": true}     不需要 key
  POST /judge  (Bearer key)    → {cause, laws, direction, verdict}

输入 case_text = 判决书「本院认为」**之前**的部分。
输出四项全部必填；错误 key → 401；单件 120 秒内返回（非流式）。

流程：抽案由 → 在该案由桶内 BM25 检索相似先例 → 交 DeepSeek 生成 → 校验 + 兜底。
"""
import hmac
import io
import json
import os
import pickle
import re
import sys
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

import kb
import llm

HERE = os.path.dirname(os.path.abspath(__file__))
DERIVED = os.path.join(HERE, "_derived")
HOST, PORT = "127.0.0.1", 8000

K1, B = 1.5, 0.75
TOP_K = 4                 # 先例条数
PRECEDENT_BRIEF = 600     # 先例案情摘要字数
NEW_CASE_CLIP = 4000      # 新案情的截断长度
TOTAL_BUDGET = 100.0      # 全流程秒数预算；超了就立刻返回兜底，绝不 524

DIRECTIONS = {"支持", "部分支持", "驳回"}

# 留出法测试用的钩子：JUDGE_EXCLUDE_IDS=J0001,J0002 时，这些先例不参与检索，
# 用来模拟"老师给的新案件不在语料里"。生产运行时不设这个变量。
EXCLUDE_IDS = {s.strip() for s in os.environ.get("JUDGE_EXCLUDE_IDS", "").split(",") if s.strip()}

SYSTEM = (
    "你是中国民事审判实务专家。依据给定的先例，对一个新案件作出判决预测。"
    "只依据提供的材料，不要引用先例之外的法条，不要编造当事人名称或金额。"
    "严格输出 JSON，不要任何解释性文字。"
)

USER_TMPL = """# 新案件（待判决）

{case}

# 同类先例（{n} 件，按相似度排序）

{precedents}

# 要求

输出 JSON，四项全部必填：

{{
  "cause": "案由，标准名称，从这份清单里选最贴切的：{causes}",
  "laws": ["《法律全称》第X条", "..."],
  "direction": "支持 或 部分支持 或 驳回",
  "verdict": "判决主文"
}}

- laws：**以「先例1」（与本案最相似的那一件）引用过的法条为主**，
  必要时再从其他先例补充；保留 2–4 条，最多 5 条。
  格式必须写成《法律全称》第X条（条号用中文数字），
  例如《中华人民共和国合同法》第一百零七条、《中华人民共和国民事诉讼法》第六十四条。
  **不要写先例里没出现过的法条**，也不要为了凑数把不相关的法条列进来。
- direction 三选一：支持 = 原告请求全部得到支持；部分支持 = 支持一部分、驳回其他；
  驳回 = 驳回原告全部诉讼请求。
  **必须依据本案的诉讼请求与证据独立判断，不要照搬先例的判决方向。**
  特别注意：只有在本案原告的**确有部分请求站不住脚**（例如证据不足、主张超出合同约定
  或法律标准、部分诉请未获支持）时，才判「部分支持」，并在 verdict 里写
  「驳回原告的其他诉讼请求」。若每一项诉讼请求都能得到支持，direction 就是「支持」，
  **此时 verdict 里不要添加「驳回原告其他诉讼请求」这一项**。
- verdict：判决主文，即「判决如下」之后的那几项（照先例的句式写）。
  **当事人名称只能从新案件原文里取，禁止编造；金额、日期、期限也只能来自新案件原文**，
  原文没写的用「（按本案证据确定）」占位，不要编造具体数字。
"""


# ==================================================================
#  载入
# ==================================================================
class KB:
    rows = None
    postings = None
    tf = None
    offsets = None
    doc_len = None
    avg = 1.0
    idf = None
    vocab = None
    cause_vocab = None
    by_cause = {}
    exclude_mask = None
    ready = False
    err = None


def load():
    need = ["kb.jsonl", "bm25.npz", "bm25_vocab.pkl", "cause_vocab.json"]
    miss = [f for f in need if not os.path.exists(os.path.join(DERIVED, f))]
    if miss:
        KB.err = "缺少产物：" + ", ".join(miss) + "  —— 先跑 python build_kb.py"
        return False
    KB.rows = []
    with io.open(os.path.join(DERIVED, "kb.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                KB.rows.append(json.loads(line))
    d = np.load(os.path.join(DERIVED, "bm25.npz"))
    KB.postings, KB.tf, KB.offsets, KB.doc_len = d["postings"], d["tf"], d["offsets"], d["doc_len"].astype(np.float32)
    KB.avg = float(KB.doc_len.mean()) or 1.0
    df = np.diff(KB.offsets)
    KB.idf = np.log(1 + (len(KB.doc_len) - df + 0.5) / (df + 0.5)).astype(np.float32)
    with open(os.path.join(DERIVED, "bm25_vocab.pkl"), "rb") as f:
        KB.vocab = pickle.load(f)
    with io.open(os.path.join(DERIVED, "cause_vocab.json"), encoding="utf-8") as f:
        KB.cause_vocab = json.load(f)
    idx = {}
    for i, r in enumerate(KB.rows):
        if r.get("cause"):
            idx.setdefault(r["cause"], []).append(i)
    KB.by_cause = {k: np.array(v, dtype=np.int64) for k, v in idx.items()}
    KB.exclude_mask = np.array([r["id"] in EXCLUDE_IDS for r in KB.rows], dtype=bool)
    KB.ready = True
    return True


def bm25_scores(query, idx=None):
    """在（可选的）行号子集上算 BM25。"""
    import jieba
    toks = [t.strip().lower() for t in jieba.lcut(query)]
    toks = [t for t in toks if len(t) >= 2 and t in KB.vocab]
    n = len(KB.rows)
    s = np.zeros(n, dtype=np.float32)
    if not toks:
        return s, []
    for t in set(toks):
        tid = KB.vocab[t]
        a, e = int(KB.offsets[tid]), int(KB.offsets[tid + 1])
        rows = KB.postings[a:e]
        tf = KB.tf[a:e].astype(np.float32)
        dl = KB.doc_len[rows]
        s[rows] += KB.idf[tid] * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / KB.avg))
    if idx is not None:
        m = np.zeros(n, dtype=np.float32)
        m[idx] = s[idx]
        s = m
    return s, toks


def retrieve(case_text, k=TOP_K):
    """先抽案由，在该案由桶内 BM25 检索；桶太少就退回全库。"""
    cause = kb.extract_cause(case_text, KB.cause_vocab)
    idx = KB.by_cause.get(cause)
    used_bucket = cause if (idx is not None and len(idx) >= 20) else None
    s, toks = bm25_scores(case_text, KB.by_cause.get(used_bucket) if used_bucket else None)
    if EXCLUDE_IDS:
        s = s.copy()
        s[KB.exclude_mask] = 0.0
    order = np.argsort(-s)[:k]
    return cause, used_bucket, [int(i) for i in order if s[i] > 0], toks


# ==================================================================
#  判案
# ==================================================================
def _fmt_precedent(i, rank):
    r = KB.rows[i]
    brief = r["case_text"][:PRECEDENT_BRIEF]
    laws = "；".join(r.get("laws") or []) or "（原件未引用具体条文）"
    return (f"【先例{rank}】案由：{r.get('cause') or '未知'}\n"
            f"案情摘要：{brief}…\n"
            f"引用法条：{laws}\n"
            f"判决方向：{r.get('direction') or '未知'}\n"
            f"判决主文：{r.get('verdict') or '（无）'}")


def _fallback(case_text, hits, reason):
    """LLM 失败时的降级：直接拿最相似先例的结论，保证四项都不为空。"""
    cause = kb.extract_cause(case_text, KB.cause_vocab) or "合同纠纷"
    if hits:
        top = KB.rows[hits[0]]
        return {
            "cause": cause,
            "laws": list(top.get("laws") or ["《中华人民共和国民事诉讼法》第六十四条"]),
            "direction": top.get("direction") or "部分支持",
            "verdict": top.get("verdict") or "驳回原告的诉讼请求。",
            "_fallback": reason,
        }
    return {"cause": cause, "laws": ["《中华人民共和国民事诉讼法》第六十四条"],
            "direction": "部分支持", "verdict": "驳回原告的诉讼请求。", "_fallback": reason}


def _norm_laws(v):
    out = []
    if isinstance(v, str):
        v = re.split(r"[；;、\n]", v)
    if isinstance(v, list):
        for x in v:
            x = re.sub(r"\s+", "", str(x or ""))
            if x and ("《" in x and "》" in x):
                out.append(x)
    return out[:8]


def judge(case_text):
    t_start = time.time()
    cause, bucket, hits, toks = retrieve(case_text)
    prec = "\n\n".join(_fmt_precedent(i, n + 1) for n, i in enumerate(hits))

    left = TOTAL_BUDGET - (time.time() - t_start)
    if left < 15:
        return _fallback(case_text, hits, "检索耗时过长")

    user = USER_TMPL.format(
        case=case_text[:NEW_CASE_CLIP], n=len(hits), precedents=prec or "（无）",
        causes="、".join(KB.cause_vocab))
    try:
        obj = llm.llm_json(SYSTEM, user, timeout=int(max(20, min(left, llm.TIMEOUT))))
    except Exception as e:
        return _fallback(case_text, hits, f"{type(e).__name__}: {str(e)[:120]}")

    out = {}
    # cause：优先用检索抽出来的（案由就写在案情里，比模型猜的准）
    c = str(obj.get("cause") or "").strip()
    out["cause"] = cause or c or "合同纠纷"
    laws = _norm_laws(obj.get("laws"))
    if not laws and hits:
        laws = list(KB.rows[hits[0]].get("laws") or [])
    out["laws"] = laws or ["《中华人民共和国民事诉讼法》第六十四条"]
    d = str(obj.get("direction") or "").strip()
    out["direction"] = d if d in DIRECTIONS else (
        KB.rows[hits[0]].get("direction") if hits else "部分支持") or "部分支持"
    v = str(obj.get("verdict") or "").strip()
    if not v and hits:
        v = KB.rows[hits[0]].get("verdict") or ""
    out["verdict"] = v or "驳回原告的诉讼请求。"

    out["_cause_source"] = "regex" if cause else "llm"
    out["_bucket"] = bucket
    out["_hits"] = hits
    out["_elapsed"] = round(time.time() - t_start, 2)
    return out


# ==================================================================
#  HTTP
# ==================================================================
API_KEY = os.environ.get("JUDGE_KEY", "")
SEM = threading.Semaphore(2)


class Handler(BaseHTTPRequestHandler):
    server_version = "judge/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def log_error(self, fmt, *args):
        print("[HTTP] " + (fmt % args), file=sys.stderr, flush=True)

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _auth_ok(self):
        """返回 (是否通过, 状态码)。没有 key 或 key 不对 → 401。"""
        if not API_KEY:
            return False, 500          # 服务端没配 key，不能让任何人都能调
        h = self.headers.get("Authorization") or ""
        m = re.match(r"^Bearer\s+(.+)$", h.strip(), re.I)
        if not m:
            return False, 401
        return hmac.compare_digest(m.group(1).strip(), API_KEY), 401

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/health", "/"):
            return self._json({"ok": True})
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.split("?")[0]
        if path != "/judge":
            return self._json({"error": "not found"}, 404)
        ok, code = self._auth_ok()
        if not ok:
            return self._json({"error": "unauthorized"}, code)

        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except Exception as e:
            return self._json({"error": f"请求体不是合法 JSON: {e}"}, 400)
        case_text = (body.get("case_text") or "").strip()
        if not case_text:
            return self._json({"error": "case_text 为空"}, 400)

        if not SEM.acquire(timeout=30):
            return self._json({"error": "服务繁忙，请稍后重试"}, 503)
        try:
            res = judge(case_text)
        except Exception as e:
            res = _fallback(case_text, [], f"{type(e).__name__}: {str(e)[:120]}")
        finally:
            SEM.release()

        # 只返回四个必填字段；下划线开头的是排查用的
        return self._json({k: v for k, v in res.items() if not k.startswith("_")})


def main():
    ok = load()
    if not ok:
        print("=" * 74)
        print(f"[错误] {KB.err}")
        print("=" * 74)
    if not API_KEY:
        print("[警告] 环境变量 JUDGE_KEY 未设置，/judge 会返回 500。请用 run_judge.bat 启动。")
    print("=" * 74)
    print("  判案 API")
    print(f"  监听       http://{HOST}:{PORT}")
    if KB.ready:
        print(f"  先例       {len(KB.rows)} 件 / 案由桶 {len(KB.by_cause)} 个")
        print(f"  BM25       词表 {len(KB.vocab)} 词")
    base, key, model = llm.llm_config()
    print(f"  DeepSeek   {'已配置 ' + str(model) if (base and key) else '未配置'}")
    print(f"  鉴权       {'已启用' if API_KEY else '密钥缺失'}")
    print("  Ctrl+C 停止")
    print("=" * 74)
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[停止]")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
