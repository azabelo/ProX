#!/usr/bin/env python3
"""Call OpenRouter (openai/gpt-5.1) with a fixed fibonacci prompt, 10 times; print each reply."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-5.1"
PROMPT = "write a short python program to calculate the first 100 fib numbers"
NUM_RUNS = 10


def chat_once(api_key: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Optional but recommended by OpenRouter for rankings / attribution
            "HTTP-Referer": "https://github.com/local/openrouter_data_generation",
            "X-Title": "fib_prompt_gpt51_ten_times",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {err_body}") from e
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected response shape: {data!r}") from e


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY in the environment.", file=sys.stderr)
        sys.exit(1)

    for i in range(1, NUM_RUNS + 1):
        print(f"\n{'=' * 72}")
        print(f"Response {i}/{NUM_RUNS}")
        print("=" * 72 + "\n")
        print(chat_once(api_key))


if __name__ == "__main__":
    main()
