#!/usr/bin/env python3
"""Execution-based scoring of generated SQL.

String exact match under-counts (equivalent queries are written differently) and says nothing about how a wrong
query fails. This script builds small SQLite databases from each example's schema, fills them with generated rows
that include the constants the reference query uses, runs the reference and the generated query, and puts every
prediction in one of four classes:

  exact       the same SQL after normalisation
  equivalent  different SQL that returns the same result on every generated database
  wrong       runs but returns a different result: a silent wrong answer, nothing signals the mistake
  error       does not run (unknown column or table, syntax error, not a SELECT): a guardrail can catch it

The generated databases approximate execution accuracy; they cannot prove two queries are equivalent, only that
no difference showed up on the test data (three databases of 24 rows per table).

    python3 sql_exec_eval.py --eval eval_set.jsonl --predictions predictions.jsonl --label finetuned-q4_k_m
"""
import argparse
import json
import random
import re
import sqlite3
import sys
from collections import Counter

WORDS = ["alpha", "beta", "gamma", "delta", "north", "south", "east", "west", "red", "blue", "green", "one", "two", "three"]
ROWS = 24
SEEDS = (11, 22, 33)


def normalise(sql):
    m = re.search(r"```(?:sql)?\s*(.*?)```", sql, re.S | re.I)
    if m:
        sql = m.group(1)
    return re.sub(r"\s+", " ", sql.strip().rstrip(";").lower())


def first_statement(sql):
    m = re.search(r"```(?:sql)?\s*(.*?)```", sql, re.S | re.I)
    if m:
        sql = m.group(1)
    return sql.strip().split(";")[0].strip()


def literals(sql):
    strings = [a or b for a, b in re.findall(r'"([^"]*)"|\'([^\']*)\'', sql)]
    bare = re.sub(r'"[^"]*"|\'[^\']*\'', " ", sql)
    numbers = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", bare)
    return strings, numbers


def build_db(context, gold, seed):
    """An in-memory database with the example's schema and rows that make the reference query's conditions true."""
    con = sqlite3.connect(":memory:")
    con.executescript(re.sub(r"(?i)create table ", "CREATE TABLE IF NOT EXISTS ", context))
    strings, numbers = literals(gold)
    rnd = random.Random(seed)
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    for table in tables:
        cols = [(c[1], c[2].upper()) for c in con.execute(f'PRAGMA table_info("{table}")')]
        for _ in range(ROWS):
            row = []
            for _, ctype in cols:
                numeric = any(t in ctype for t in ("INT", "REAL", "FLOAT", "NUM", "DOUBLE"))
                if numeric:
                    pool = [float(n) for n in numbers] + [float(n) + d for n in numbers for d in (-1, 1)]
                    v = rnd.choice(pool) if pool and rnd.random() < 0.5 else rnd.randint(0, 60)
                    row.append(int(v) if float(v).is_integer() else v)
                else:
                    pool = strings + numbers
                    row.append(rnd.choice(pool) if pool and rnd.random() < 0.5 else rnd.choice(WORDS))
            con.execute(f'INSERT INTO "{table}" VALUES ({",".join("?" * len(cols))})', row)
    return con


def run(con, sql):
    steps = [0]

    def budget():
        steps[0] += 1
        return 1 if steps[0] > 2000 else 0

    con.set_progress_handler(budget, 1000)
    rows = con.execute(sql).fetchall()
    con.set_progress_handler(None, 0)
    return [tuple(round(v, 6) if isinstance(v, float) else v for v in r) for r in rows]


def same(gold_rows, pred_rows, ordered):
    if ordered:
        return gold_rows == pred_rows
    return sorted(map(repr, gold_rows)) == sorted(map(repr, pred_rows))


def clauses(sql):
    s = normalise(sql)
    sel = re.search(r"select (.*?)(?: from |$)", s)
    whe = re.search(r" where (.*)$", s)
    return (sel.group(1) if sel else ""), (whe.group(1) if whe else "")


def usable(example):
    """Examples whose schema does not load in SQLite or whose reference query does not run are dropped for every model."""
    try:
        con = build_db(example["context"], example["answer"], SEEDS[0])
        run(con, example["answer"])
        return True
    except sqlite3.Error:
        return False


def classify(example, answer):
    gold, pred = example["answer"], first_statement(answer)
    if normalise(pred) == normalise(gold):
        return "exact", ""
    if not re.match(r"(?is)^\s*(select|with)\b", pred):
        return "error", "not a SELECT"
    ordered = " order by " in normalise(gold)
    detail = ""
    for seed in SEEDS:
        con = build_db(example["context"], gold, seed)
        g = run(con, gold)
        try:
            p = run(con, pred)
        except sqlite3.Error as e:
            msg = str(e)
            return "error", ("unknown column or table" if "no such" in msg else "syntax error" if "syntax" in msg or "near" in msg else "other error")
        if not same(g, p, ordered):
            gs, gw = clauses(gold)
            ps, pw = clauses(pred)
            detail = "condition differs" if gs == ps else "selected columns or aggregate differ" if gw == pw else "both differ"
            return "wrong", detail
    return "equivalent", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    examples = [json.loads(l) for l in open(args.eval)]
    preds = [json.loads(l) for l in open(args.predictions)]
    counts, details, skipped, samples = Counter(), Counter(), 0, []
    for ex, p in zip(examples, preds):
        if not usable(ex):
            skipped += 1
            continue
        cls, detail = classify(ex, p["answer"])
        counts[cls] += 1
        if detail:
            details[(cls, detail)] += 1
        if cls == "wrong" and len(samples) < 5:
            samples.append({"question": ex["question"], "reference": ex["answer"], "generated": first_statement(p["answer"])})
    n = sum(counts.values())
    if n == 0:
        sys.exit(f"nothing to score: {len(examples)} examples, {skipped} skipped, {len(preds)} predictions")
    lat = sorted(p["latency_ms"] for p in preds)
    result = {"label": args.label, "scored": n, "skipped": skipped,
              "exact": counts["exact"] / n, "equivalent": counts["equivalent"] / n,
              "execution_match": (counts["exact"] + counts["equivalent"]) / n,
              "wrong_result": counts["wrong"] / n, "error": counts["error"] / n,
              "breakdown": {f"{c}: {d}": v / n for (c, d), v in sorted(details.items())},
              "wrong_examples": samples}
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items() if k != "wrong_examples"}))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    sys.exit(main())
