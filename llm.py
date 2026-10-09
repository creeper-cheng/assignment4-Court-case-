# -*- coding: utf-8 -*-
r"""DeepSeek 非流式 JSON 调用。

作业明确要求「不要用流式输出」——Cloudflare 免费隧道单次请求超过约 2 分钟会断（524），
而流式反而容易踩这个坑。所以这里只用一次性返回。
"""
import json
import os
import re
import time

TIMEOUT = int(os.environ.get("JUDGE_LLM_TIMEOUT", "75"))
RETRY = 2


def llm_config():
    base = os.environ.get("LLM_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL")
    key = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    model = os.environ.get("LLM_MODEL") or ("deepseek-chat" if base else None)
    return base, key, model


def llm_json(system, user, timeout=None, temperature=0.1):
    """返回解析后的 dict。失败抛 RuntimeError。"""
    import requests
    base, key, model = llm_config()
    if not key or not base:
        raise RuntimeError("未配置 DeepSeek（需要 LLM_API_KEY / LLM_BASE_URL）")
    if not base.rstrip("/").endswith("/v1"):
        base = base.rstrip("/") + "/v1"
    timeout = timeout or TIMEOUT
    payload = {
        "model": model, "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last = ""
    # 不走系统代理：巨潮/DeepSeek 都是国内直连，走代理反而慢且易超时
    s = requests.Session()
    s.trust_env = False
    s.proxies = {"http": None, "https": None}
    for attempt in range(RETRY + 1):
        if attempt:
            time.sleep(1.5 * attempt)
        try:
            r = s.post(f"{base}/chat/completions", headers=headers,
                       json=payload, timeout=timeout)
            if r.status_code in (429,) or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                continue
            if r.status_code != 200:
                raise RuntimeError(f"DeepSeek HTTP {r.status_code}: {r.text[:200]}")
            txt = r.json()["choices"][0]["message"]["content"]
            try:
                return json.loads(txt)
            except json.JSONDecodeError:
                # 兜底：从回复里抠出第一个 {...}
                m = re.search(r"\{.*\}", txt, re.S)
                if m:
                    return json.loads(m.group(0))
                last = f"返回不是合法 JSON: {txt[:120]}"
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            last = f"{type(e).__name__}: {e}"
    raise RuntimeError(f"DeepSeek 调用失败（重试 {RETRY} 次）：{last}")
