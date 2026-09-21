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


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm_sql(s):
    m = re.search(r"```(?:sql)?\s*(.*?)```", s, re.S | re.I)
    if m:
        s = m.group(1)
    return re.sub(r"\s+", " ", s.strip().rstrip(";").lower())


def ask(port, ex):
    t0 = time.time()
    return ask_text(port, ex), (time.time() - t0) * 1000


def ask_text(port, ex):
    body = {"model": "m", "temperature": 0, "max_tokens": 128, "stream": False,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"Schema:\n{ex['context']}\n\nQuestion: {ex['question']}"}]}
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    try:
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        d = json.loads(c.getresponse().read())
    finally:
        c.close()
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
    ap.add_argument("--predictions", default="", help="write one JSON line per example: the answer and its latency")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N examples")
    args = ap.parse_args()
    examples = read_jsonl(args.eval)
    if not examples:
        raise ValueError("evaluation set is empty")
    if args.limit:
        examples = examples[:args.limit]
    server = subprocess.Popen(["llama-server", "-m", args.model, "--port", str(args.port), "-np", str(args.concurrency), "-c", "8192"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                c = http.client.HTTPConnection("127.0.0.1", args.port, timeout=2)
                c.request("GET", "/health")
                status = c.getresponse().status
                c.close()
                if status == 200:
                    break
            except OSError:
                pass
            time.sleep(1)  # still loading (503) or not listening yet
        else:
            raise RuntimeError("llama-server did not become healthy")
        t0 = time.time()
        with cf.ThreadPoolExecutor(args.concurrency) as ex:
            timed = list(ex.map(lambda e: ask(args.port, e), examples))
        wall = time.time() - t0
        answers = [a for a, _ in timed]
        lat = sorted(ms for _, ms in timed)
    finally:
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(5)
        except subprocess.TimeoutExpired:
            server.kill()
    correct = sum(norm_sql(a) == norm_sql(e["answer"]) for a, e in zip(answers, examples))
    result = {"label": args.label or os.path.basename(args.model), "examples": len(examples),
              "exact_match": correct / len(examples), "file_mb": round(os.path.getsize(args.model) / 1e6),
              "seconds": round(wall, 1), "concurrency": args.concurrency,
              "latency_ms_p50": round(lat[len(lat) // 2]), "latency_ms_p95": round(lat[int(0.95 * (len(lat) - 1))])}
    if args.predictions:
        with open(args.predictions, "w") as f:
            for (a, ms), e in zip(timed, examples):
                f.write(json.dumps({"answer": a, "latency_ms": round(ms)}) + "\n")
    print(json.dumps(result))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
