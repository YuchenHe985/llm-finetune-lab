#!/usr/bin/env python3
"""Can the model tell when its answer is probably wrong?

About one answer in six is a silent wrong result: valid SQL that runs and returns the wrong rows. A syntax check
cannot catch it. This script asks for the greedy answer plus several sampled answers per question, runs every
candidate on the generated test database, and treats the share of candidates that return the same rows as the
greedy answer as a confidence score. Answers below a threshold are withheld (sent to a person). It reports, for each
threshold, how many questions are still answered and how many of the answered ones are wrong.

    python3 abstain_eval.py --model model-q4_k_m.gguf --eval eval_set.jsonl --predictions predictions.jsonl
"""
import argparse
import concurrent.futures as cf
import http.client
import json
import re
import signal
import subprocess
import time

import sql_exec_eval as E
from eval_gguf import SYSTEM


def sample(port, ex, seed, temperature):
    body = {"model": "m", "temperature": temperature, "top_p": 0.95, "seed": seed, "max_tokens": 128, "stream": False,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"Schema:\n{ex['context']}\n\nQuestion: {ex['question']}"}]}
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    try:
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        return json.loads(c.getresponse().read())["choices"][0]["message"]["content"]
    finally:
        c.close()


def invented_constant(example, sql):
    """True if the SQL compares against a string that does not appear in the question (a paraphrased or made-up value)."""
    def words(t):
        return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()

    question = words(example["question"])
    return any(words(s) and words(s) not in question for s in E.literals(E.first_statement(sql))[0])


def signature(example, sql):
    """The rows a candidate returns on the first generated database, or None if it does not run."""
    sql = E.first_statement(sql)
    try:
        con = E.build_db(example["context"], example["answer"], E.SEEDS[0])
        return tuple(sorted(map(repr, E.run(con, sql))))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--predictions", required=True, help="the greedy answers from eval_gguf.py")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--port", type=int, default=9302)
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate only the first N examples, matching eval_gguf.py --limit N")
    args = ap.parse_args()
    examples = E.read_jsonl(args.eval)
    greedy = [row["answer"] for row in E.read_jsonl(args.predictions)]
    examples, greedy = E.apply_limit(examples, greedy, args.limit)
    E.require_aligned(examples, greedy)
    keep = [i for i, e in enumerate(examples) if E.usable(e)]

    server = subprocess.Popen(["llama-server", "-m", args.model, "--port", str(args.port), "-np", "4", "-c", "8192"],
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
            time.sleep(1)
        else:
            raise RuntimeError("llama-server did not become healthy")
        jobs = [(i, s) for i in keep for s in range(args.samples)]
        with cf.ThreadPoolExecutor(4) as pool:
            sampled = list(pool.map(lambda j: sample(args.port, examples[j[0]], 1000 + j[1], args.temperature), jobs))
    finally:
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(5)
        except subprocess.TimeoutExpired:
            server.kill()

    by_example = {i: [] for i in keep}
    for (i, _), text in zip(jobs, sampled):
        by_example[i].append(text)
    records = []
    for i in keep:
        base = signature(examples[i], greedy[i])
        agree = 1 + sum(1 for t in by_example[i] if base is not None and signature(examples[i], t) == base)
        confidence = agree / (1 + args.samples) if base is not None else 0.0
        cls, _ = E.classify(examples[i], greedy[i])
        records.append({"confidence": confidence, "class": cls, "invented": invented_constant(examples[i], greedy[i])})

    rows = []
    for tau in sorted({r["confidence"] for r in records}):
        answered = [r for r in records if r["confidence"] >= tau]
        good = sum(r["class"] in ("exact", "equivalent") for r in answered)
        wrong = sum(r["class"] == "wrong" for r in answered)
        rows.append({"threshold": tau, "coverage": len(answered) / len(records), "execution_match": good / len(answered),
                     "silent_wrong": wrong / len(answered)})
    policies = {
        "answer everything": lambda r: True,
        "withhold when a string constant is not in the question": lambda r: not r["invented"],
        "withhold unless all samples agree": lambda r: r["confidence"] == 1.0,
        "withhold on either signal": lambda r: r["confidence"] == 1.0 and not r["invented"],
    }
    table = []
    for name, keep_it in policies.items():
        answered = [r for r in records if keep_it(r)]
        table.append({"policy": name, "coverage": len(answered) / len(records),
                      "execution_match": sum(r["class"] in ("exact", "equivalent") for r in answered) / len(answered),
                      "silent_wrong": sum(r["class"] == "wrong" for r in answered) / len(answered)})
    wrong = [r for r in records if r["class"] == "wrong"]
    right = [r for r in records if r["class"] in ("exact", "equivalent")]
    result = {"questions": len(records), "samples": args.samples, "temperature": args.temperature, "rows": rows, "policies": table,
              "invented_constant_flags": {"of_wrong_answers": sum(r["invented"] for r in wrong) / len(wrong),
                                          "of_correct_answers": sum(r["invented"] for r in right) / len(right)},
              "records": records}
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
