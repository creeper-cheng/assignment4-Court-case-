# -*- coding: utf-8 -*-
r"""端到端验收：用留出法模拟老师的 10 个新案件。

做法：从 3,520 份里挑 10 份（覆盖各主要案由），用 JUDGE_EXCLUDE_IDS 把它们
从检索里排除——即模拟"这些案子不在语料里"。然后逐件调 /judge，
对比案由/法条/判决方向，并记录每件耗时（必须 < 120 秒）。

跑：python tests/e2e.py
"""
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DERIVED = os.path.join(ROOT, "_derived")
BASE = "http://127.0.0.1:8000"
N_HOLDOUT = 10


def load_env():
    cfg = {}
    for line in io.open(os.path.join(ROOT, "judge.env"), encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def pick_holdout(rows):
    """按案由分层抽样：每个主要案由取 1 件，最大的再补，凑够 10 件。"""
    by = defaultdict(list)
    for r in rows:
        by[r.get("cause") or "(无案由)"].append(r)
    order = sorted(by, key=lambda c: -len(by[c]))
    picked = []
    for c in order:
        if len(picked) >= N_HOLDOUT:
            break
        picked.append(by[c][0])
    i = 0
    while len(picked) < N_HOLDOUT and order:
        c = order[i % len(order)]
        if len(by[c]) > 1:
            picked.append(by[c][1])
        i += 1
        if i > 50:
            break
    return picked[:N_HOLDOUT]


def post(path, obj, key=None, timeout=180):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(obj, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    return urllib.request.urlopen(req, timeout=timeout)


def get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def main():
    cfg = load_env()
    key = cfg.get("JUDGE_KEY", "")
    rows = [json.loads(l) for l in io.open(os.path.join(DERIVED, "kb.jsonl"), encoding="utf-8") if l.strip()]
    holdout = pick_holdout(rows)
    ids = [r["id"] for r in holdout]
    print(f"留出 {len(holdout)} 件模拟新案件：")
    for r in holdout:
        print(f"   {r['id']}  {r.get('cause')}  方向={r.get('direction')}")

    # ---- 起服务（排除留出件）----
    env = dict(os.environ)
    env.update(cfg)
    env["JUDGE_EXCLUDE_IDS"] = ",".join(ids)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.Popen([sys.executable, "-u", os.path.join(ROOT, "serve.py")],
                            cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for _ in range(40):
            time.sleep(0.5)
            try:
                st, d = get("/health")
                if st == 200 and d.get("ok") is True:
                    break
            except Exception:
                pass
        else:
            print("!! 服务起不来")
            return 1

        ok = bad = 0

        def check(name, cond, detail=""):
            nonlocal ok, bad
            if cond:
                ok += 1
                print(f"  OK   {name}  {detail}")
            else:
                bad += 1
                print(f"  FAIL {name}  {detail}")

        print("\n=== A. 连通与鉴权 ===")
        st, d = get("/health")
        check("/health 返回 {\"ok\": true}", st == 200 and d == {"ok": True}, str(d))

        # 不带 key
        try:
            post("/judge", {"case_text": "测试"}, key=None, timeout=30)
            check("不带 key → 401", False, "居然成功了")
        except urllib.error.HTTPError as e:
            check("不带 key → 401", e.code == 401, f"HTTP {e.code}")
        # 错 key
        try:
            post("/judge", {"case_text": "测试"}, key="wrong-key", timeout=30)
            check("错 key → 401", False, "居然成功了")
        except urllib.error.HTTPError as e:
            check("错 key → 401", e.code == 401, f"HTTP {e.code}")

        print("\n=== B. 逐件判案（留出件不在检索里）===")
        causes_hit = laws_hit = dirs_hit = 0
        times = []
        for i, r in enumerate(holdout, 1):
            t0 = time.time()
            try:
                resp = post("/judge", {"case_text": r["case_text"]}, key=key)
                res = json.loads(resp.read().decode("utf-8"))
                err = ""
            except Exception as e:
                res, err = {}, f"{type(e).__name__}: {e}"
            dt = time.time() - t0
            times.append(dt)

            c_ok = res.get("cause") == r.get("cause")
            d_ok = res.get("direction") == r.get("direction")
            want = set(r.get("laws") or [])
            got = set(res.get("laws") or [])
            l_ok = bool(want & got)          # 至少命中一条就算过（法条一套常引多条）
            causes_hit += c_ok
            dirs_hit += d_ok
            laws_hit += l_ok

            print(f"\n  [{i}/{len(holdout)}] {r['id']}  {dt:.1f}s"
                  + (f"  {err}" if err else ""))
            print(f"      案由   预期={r.get('cause')}  实际={res.get('cause')}  {'✓' if c_ok else '✗'}")
            print(f"      方向   预期={r.get('direction')}  实际={res.get('direction')}  {'✓' if d_ok else '✗'}")
            print(f"      法条   预期{len(want)}条  实际{len(got)}条  交集{len(want & got)}条  {'✓' if l_ok else '✗'}")
            v = (res.get("verdict") or "").replace("\n", " ")
            print(f"      主文   {v[:96]}{'…' if len(v) > 96 else ''}")

        n = len(holdout)
        print("\n" + "=" * 74)
        print(f"  案由命中   {causes_hit}/{n} = {causes_hit/n:.0%}")
        print(f"  方向命中   {dirs_hit}/{n} = {dirs_hit/n:.0%}")
        print(f"  法条命中   {laws_hit}/{n} = {laws_hit/n:.0%}  （至少命中一条）")
        print(f"  耗时       min={min(times):.1f}s 均值={sum(times)/n:.1f}s max={max(times):.1f}s")
        print(f"  120 秒内   {'✓ 全部满足' if max(times) < 120 else '✗ 有超时'}")
        print("=" * 74)
        return 1 if bad else 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
