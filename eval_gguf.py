#!/usr/bin/env python3
"""Exact-match SQL accuracy of a GGUF model served by llama.cpp, on the held-out examples from train.py.

    python3 eval_gguf.py --model model-q4_k_m.gguf --eval evalset/eval_set.jsonl --label finetuned-q4_k_m

The prompt is the one train.py used (system message, then the schema and the question); llama-server applies the
chat template stored in the GGUF. Answers are compared as SQL: markdown fences, whitespace, case and a trailing
semicolon are ignored.
"""
import argparse
import concurrent.futures as cf
import http.client
import json
import os
import re
import signal
import subprocess
import time

SYSTEM = "You translate questions into SQL for the given schema. Reply with one SQL statement only."


def norm_sql(s):
    m = re.search(r"```(?:sql)?\s*(.*?)```", s, re.S | re.I)
    if m:
        s = m.group(1)
    return re.sub(r"\s+", " ", s.strip().rstrip(";").lower())


def ask(port, ex):
    body = {"model": "m", "temperature": 0, "max_tokens": 128, "stream": False,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"Schema:\n{ex['context']}\n\nQuestion: {ex['question']}"}]}
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    d = json.loads(c.getresponse().read())
    if "choices" not in d:
        raise RuntimeError(f"server error: {d}")
    return d["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--port", type=int, default=9301)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    examples = [json.loads(l) for l in open(args.eval)]
    server = subprocess.Popen(["llama-server", "-m", args.model, "--port", str(args.port), "-np", "4", "-c", "8192"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                c = http.client.HTTPConnection("127.0.0.1", args.port, timeout=2)
                c.request("GET", "/health")
                if c.getresponse().status == 200:
                    break
            except OSError:
                pass
            time.sleep(1)  # still loading (503) or not listening yet
        else:
            raise RuntimeError("llama-server did not become healthy")
        t0 = time.time()
        with cf.ThreadPoolExecutor(4) as ex:
            answers = list(ex.map(lambda e: ask(args.port, e), examples))
        wall = time.time() - t0
    finally:
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(5)
        except subprocess.TimeoutExpired:
            server.kill()
    correct = sum(norm_sql(a) == norm_sql(e["answer"]) for a, e in zip(answers, examples))
    result = {"label": args.label or os.path.basename(args.model), "examples": len(examples),
              "exact_match": correct / len(examples), "file_mb": round(os.path.getsize(args.model) / 1e6),
              "seconds": round(wall, 1)}
    print(json.dumps(result))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
