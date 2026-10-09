# -*- coding: utf-8 -*-
r"""试用判案 API 的小工具。

用法：
  python try_judge.py                     用内置样例案情调本地 API
  python try_judge.py --id J0003          用语料里某份判决书的案情（会显示与元件的对比）
  python try_judge.py --n 3               连测 3 件（随机抽），看耗时和稳定性
  python try_judge.py --url https://xxxx.trycloudflare.com    调公网地址
  python try_judge.py --file case.txt     用自己写的一个案情文本文件

为什么用脚本而不是 curl：案情是中文，curl 在 Windows 命令行里传中文会被 GBK 弄坏。
"""
import argparse
import io
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SAMPLE = (
    "江苏省宿迁市宿豫区人民法院民 事 判 决 书（2020）苏1311民初710号"
    "原告：张敏，男，1968年8月13日出生，居民，住所地江苏省宿迁市宿豫区。"
    "被告：王景文，男，1975年2月21日出生，居民。被告：张明荣，女，1976年11月22日出生，居民。"
    "原告张敏与被告王景文、张明荣民间借贷纠纷一案，本院于2020年3月25日立案后，"
    "依法适用简易诉讼程序，公开开庭进行了审理。本案现已审理终结。"
    "原告张敏向本院提出诉讼请求：1.判令被告王景文、张明荣立即给付借款本金113200元"
    "及逾期利息80000元；2.案件诉讼费由被告承担。"
    "被告王景文、张明荣未作答辩。"
    "本院经审理认定事实如下：2013年7月1日，被告王景文、张明荣向原告张敏借款并出具借条一份，"
    "载明：今借到张敏现金捌万元整（¥80000），月息两分。2014年5月20日，被告又向原告借款"
    "33200元，出具借条一份。上述款项经原告多次催要，被告至今未还。"
)


def load_cfg():
    cfg = {}
    p = os.path.join(HERE, "judge.env")
    if os.path.exists(p):
        for line in io.open(p, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg


def call(base, key, case_text, timeout=180):
    req = urllib.request.Request(
        base.rstrip("/") + "/judge",
        data=json.dumps({"case_text": case_text}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
        method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode("utf-8"))
    return body, time.time() - t0, r.status


def show(res, dt, truth=None):
    print(f"  耗时     {dt:.1f}s")
    print(f"  cause    {res.get('cause')}" + (f"      (原件: {truth['cause']})" if truth else ""))
    print(f"  direction{res.get('direction')}" + (f"      (原件: {truth['direction']})" if truth else ""))
    laws = res.get("laws") or []
    print(f"  laws     {len(laws)} 条:")
    for x in laws:
        print(f"             {x}")
    print(f"  verdict  {(res.get('verdict') or '')[:220]}")
    if truth:
        wl = set(truth.get("laws") or [])
        print(f"           法条与原件交集: {len(wl & set(laws))} 条 / 原件 {len(wl)} 条")


def main():
    ap = argparse.ArgumentParser(description="试用判案 API")
    ap.add_argument("bare_url", nargs="?", help="API 地址（等价于 --url，可直接写网址）")
    ap.add_argument("--url", help="API 地址")
    ap.add_argument("--key", help="不填则从 judge.env 读 JUDGE_KEY")
    ap.add_argument("--id", help="用语料里某份判决书的案情，如 J0003")
    ap.add_argument("--file", help="用自己写的案情文本文件（UTF-8）")
    ap.add_argument("--n", type=int, default=1, help="连测 n 件（随机抽语料）")
    args = ap.parse_args()

    # 允许直接写网址（python try_judge.py https://xxx），省得记 --url
    base = args.url or args.bare_url or "http://127.0.0.1:8000"
    if not base.startswith("http"):
        base = "https://" + base
    cfg = load_cfg()
    key = args.key or cfg.get("JUDGE_KEY")
    if not key:
        print("[错误] 没有 key，请在 judge.env 里设 JUDGE_KEY，或用 --key 指定")
        return 1

    kb_path = os.path.join(HERE, "_derived", "kb.jsonl")
    rows = []
    if os.path.exists(kb_path):
        rows = [json.loads(l) for l in io.open(kb_path, encoding="utf-8") if l.strip()]

    print(f"API: {base}\n")

    if args.file:
        cases = [io.open(args.file, encoding="utf-8").read()]
        truths = [None]
    elif args.id:
        r = next((x for x in rows if x["id"] == args.id), None)
        if not r:
            print(f"[错误] 语料里没有 {args.id}")
            return 1
        cases = [r["case_text"]]
        truths = [r]
    elif rows:
        picks = random.sample(rows, min(args.n, len(rows)))
        cases = [x["case_text"] for x in picks]
        truths = picks
    else:
        cases, truths = [SAMPLE], [None]

    times = []
    for i, (ct, truth) in enumerate(zip(cases, truths), 1):
        label = truth["id"] if truth else "内置样例"
        print("=" * 74)
        print(f"[{i}/{len(cases)}] {label}   案情 {len(ct)} 字")
        try:
            res, dt, st = call(base, key, ct)
        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}")
            return 1
        except Exception as e:
            print(f"  失败: {type(e).__name__}: {e}")
            return 1
        times.append(dt)
        show(res, dt, truth)

    print("=" * 74)
    print(f"共 {len(times)} 件  均值 {sum(times)/len(times):.1f}s  最慢 {max(times):.1f}s"
          f"  {'✓ 都在 120 秒内' if max(times) < 120 else '✗ 有超时'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
