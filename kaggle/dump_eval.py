import json
from datasets import load_dataset

# the same call, order and seed train.py uses, so these are exactly the held-out examples
raw = load_dataset("b-mc2/sql-create-context", split="train").shuffle(seed=1)
with open("eval_set.jsonl", "w") as f:
    for i in range(400):
        ex = raw[i]
        f.write(json.dumps({"context": ex["context"], "question": ex["question"], "answer": ex["answer"]}) + "\n")
print("wrote 400 held-out examples")
