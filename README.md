# llm-finetune-lab

**Question:** can a 0.5B-parameter model, fine-tuned with LoRA, answer questions about a database by writing SQL, on a laptop and without sending
data anywhere? **Answer:** as a tool that suggests SQL for a person to review, yes; as something that answers on its own, no. About one answer in six runs
fine and returns the wrong rows, and the checks that are cheap to build catch only some of those.

This repository is the fine-tuning, the evaluation that led to that conclusion, and the training scripts. The evaluation is the substantial part: it scores
generated SQL by running it, measures how the model fails, and tests guardrails against those failures.

## Why this question

The serving work in [radixgates](https://github.com/YuchenHe985/radixgates) is about running models reliably. This asks the question that comes before it: is the model worth
serving? Text-to-SQL is a useful test case for three reasons. Running it on the same machine keeps the data there, which matters when the database is sensitive. A wrong
answer can be checked by running it, so quality can be measured without a person grading each one. And the database schema is a long prefix shared by many questions, the
workload that prefix-cache routing is built for, and the one radixgates' real-engine experiments use. The point of the exercise is the evaluation: what it takes to trust a
small model with a database, and what a team would want to know before shipping it.

## Acceptance criteria and what was measured

No team asked for this; the targets below are assumptions for an internal analytics assistant, not requirements from a real user.

| Criterion | Target | Measured (fine-tuned, Q4_K_M) | Met |
| --- | --- | --- | --- |
| Result accuracy: generated SQL returns the right rows | at least 90% | 82.6% | no |
| Silent wrong answers: runs, returns the wrong rows | at most 5% | 16.4% (9.1% with the guardrails below, answering 76% of questions) | no |
| Latency per query on a laptop | p95 under 1 s | 611 ms (Apple M1, one request at a time) | yes |
| Model size | under 1 GB | 398 MB | yes |
| Data stays on the machine | required | runs locally in llama.cpp | yes |

**Decision:** do not ship it as an autonomous answerer. Ship it as a suggestion tool: show the generated SQL, let the person run it, and use the
guardrails to flag the answers most likely to be wrong. What would change the decision is listed at the end.

## Results

Held-out questions from [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context); 391 of 400 are scored (9 have schemas that do not
load in SQLite and are dropped for every model). Answers are scored by execution (`sql_exec_eval.py`): each schema gets small generated SQLite databases
that contain the constants the reference query uses, and the generated query must return the same rows as the reference on all three.

| Model | Size | Same SQL | Runs and is right | Runs but wrong | Does not run | Latency p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B-Instruct, Q4_K_M | 491 MB | 11.0% | 42.7% | 41.9% | 15.3% | 315 / 564 ms |
| Fine-tuned, F16 | 994 MB | 72.6% | 83.1% | 15.9% | 1.0% | 580 / 925 ms |
| Fine-tuned, Q8_0 | 531 MB | 73.4% | 83.6% | 15.3% | 1.0% | 451 / 720 ms |
| Fine-tuned, Q4_K_M | 398 MB | 71.4% | 82.6% | 16.4% | 1.0% | 400 / 611 ms |

- Fine-tuning roughly doubles the share of right answers (42.7% to 82.6%) and nearly removes queries that do not run (15.3% to 1.0%).
- String comparison understates accuracy by about ten points, because equivalent queries are written differently ("same SQL" against "runs and is right").
- **Only 1% of the fine-tuned model's answers fail to run.** Checking the SQL against the schema catches those and nothing else: the 16% that run but return
  the wrong rows raise no error.
- Quantization costs little: Q4_K_M is 60% smaller than F16 and about 1 point worse, which is inside the noise of 391 questions (a standard error of about 2 points).
- Latency is one request at a time on an M1 with Metal; with four requests in parallel the Q4_K_M p95 rises to 1.4 s.

### Why the answers are wrong

`results/failure_audit.md` lists the first 30 wrong answers in dataset order with a verdict for each. The verdicts are one judgement per case on a small sample,
a guide and not a measurement:

- **18 model errors.** Eight change a string constant so it no longer matches the question ("nouvelair" becomes "nouvelle-air", "western kentucky" becomes
  "western ky", "1400 m" becomes "1400m"); four add or drop a condition; three add a COUNT where a value was asked for; two quote a number; one question is garbled.
- **5 reference errors:** the generated SQL is at least as reasonable as the reference (a question asking for "bigger than 6" whose reference says `< 6`).
- **7 ambiguous questions:** nothing says which aggregate to use, and the reference picked one.

If that sample is representative, the true silent-wrong rate is nearer 10% than 16%: still twice the target.

### Guardrails (exploratory, post-hoc)

`abstain_eval.py` asks for four more sampled answers per question (temperature 0.7), runs all five on the test databases, and tries two signals for
withholding an answer: the samples disagree, or the SQL compares against a string that is not in the question.

These policies were selected and reported on the same 391-question evaluation set, and sample agreement currently uses one generated database per
question. Treat the numbers below as hypothesis-generating rather than a production validation; a release decision needs a frozen policy evaluated on
a separate, representative set of real schemas and questions.

| Policy | Questions still answered | Right | Silent wrong |
| --- | ---: | ---: | ---: |
| Answer everything | 100% | 82.6% | 16.4% |
| Withhold when a string constant is not in the question | 96.7% | 85.2% | 13.8% |
| Withhold unless all five samples agree | 78.0% | 89.2% | 10.8% |
| Withhold on either signal | 76.2% | 90.9% | 9.1% |

The string-constant rule is cheap and precise: it fires on 18.8% of the wrong answers and on 0.3% of the right ones, and it costs no extra inference.
Sample disagreement is expensive (five generations per question) and catches more, but a large share of the errors are systematic, the model makes the same
mistake every time, so agreement cannot see them. Together they bring silent wrong answers from 16.4% to 9.1%, which is not enough to remove the person.

### What would change the decision

1. **Evaluate on the real schema and questions.** A public dataset of mostly single-table lookups is the biggest threat to these numbers; the result could be better or
   worse on a real workload.
2. **Enforce what the string-constant rule only flags:** constrain the decoder so string literals are substrings of the question. Invented or paraphrased constants (eight of the 18
   model errors in the audit) could then no longer occur, although the constraint does not make the model pick the right substring.
3. **Handle ambiguity by asking:** when the question does not say which aggregate or which column, ask instead of guessing.
4. **A larger model.** Measure how many points a 1.5B or 3B model buys against its latency and memory before assuming it.

## How the model was trained

Qwen2.5-0.5B-Instruct with LoRA (rank 16 on all attention and MLP projections: 8.80M trainable parameters of 502.8M, 1.75%), 8,000 training examples,
500 steps at a global batch of 16, fp32 master weights with fp16 autocast, on a Kaggle notebook with two Tesla T4s. The loss is on the answer tokens only.
`train.py` runs under `torchrun` on one or two GPUs with DistributedDataParallel; the global batch is fixed, so both runs see the same data and reach the
same result (71.8% and 72.0% exact match on all 400 questions for 1 and 2 GPUs, held-out loss 0.306 to 0.060).

| | 1 GPU | 2 GPUs |
| --- | ---: | ---: |
| Seconds per optimizer step (after warm-up) | 0.790 | 0.450 |
| Tokens per second | 1,924 | 3,378 |
| Speedup | | **1.76x** (88% of linear) |
| Peak memory per GPU | 8.6 GiB | 8.6 GiB |

The T4s talk over PCIe (`kaggle/nccl_probe.py`: an all-reduce of 256 MB takes 66 ms, about 3.9 GB/s), and only the LoRA gradients (about 35 MB) are
synchronised, which is why scaling is close to linear. The adapter is merged into the base weights, converted to GGUF with llama.cpp's
`convert_hf_to_gguf.py` and quantized with `llama-quantize`; `eval_gguf.py` serves each file with `llama-server`.

### Problems that came up

- **CUDA out of memory on one GPU at a per-device batch of 8.** The vocabulary has 152K entries, so the logits of an 8 x 384 batch take about 1.9 GB in fp32 and a
  long batch pushed the 15 GB card over. Fixed with a per-device batch of 4 and more gradient accumulation (global batch still 16).
- **The base model often does not answer with a query.** 12% of its answers are fragments such as `theme = 'carole king' AND ...`; they count as "does not run", and scoring
  by execution shows a base model that is far better than string comparison (11%) suggests (42.7%).
- **The evaluation harness started sending requests while the model was still loading;** it now waits for `/health` to return 200.
- **The Kaggle image ships a `torchao` too old for its `peft` and `transformers`;** the notebook uninstalls it first.

### Storing and shipping the result

The merged fp16 model shares only 27.6% of its bytes with the base, the token embedding that LoRA did not train: LoRA changed about 99% of the values in every
projection matrix, so a chunk-level store measured with [cdc-chunker](https://github.com/YuchenHe985/cdc-chunker) at Hugging Face Xet's chunk sizes keeps 716 MB of the
988 MB file as new data. The adapter alone (8.8M parameters, about 35 MB in fp32) is what to store or send when the base is already there. The GGUF
numbers and the per-tensor breakdown are in the [model-file section](https://github.com/YuchenHe985/cdc-chunker#deduplication-of-model-files) of that repository.

## Limits

One small model, one public dataset, one seed, 500 steps. The generated test databases approximate execution accuracy: they can show two queries differ, not prove
they are equivalent, so "runs and is right" is an upper bound. The failure audit is 30 cases and one judgement each. The reference SQL in this dataset has errors of its
own. Scaling was measured on two PCIe T4s only.

## Reproduce

The training output now records the installed `torch`, `transformers`, `peft`, and `datasets` versions. For a long-lived reproduction, also pin the exact
model, dataset, and llama.cpp revisions in the run manifest; the historical artifacts in this repository predate that metadata capture.

Training runs in a Kaggle notebook with two T4s and internet access:

```bash
KAGGLE_USERNAME=<you> python3 kaggle/build_kernel.py full     # writes kernel/run_kaggle.py, one file that embeds train.py
kaggle kernels push -p kernel                                  # trains on 1 then 2 GPUs; writes results and the merged model
```

`kaggle/dump_eval.py` writes the 400 held-out questions in the order `train.py` selects them. Export and evaluation run locally with llama.cpp installed:

```bash
python3 convert_hf_to_gguf.py merged --outfile model-f16.gguf --outtype f16      # from the llama.cpp repository
llama-quantize model-f16.gguf model-q4_k_m.gguf q4_k_m
python3 eval_gguf.py --model model-q4_k_m.gguf --eval eval_set.jsonl --label q4_k_m --predictions predictions.jsonl
python3 sql_exec_eval.py --eval eval_set.jsonl --predictions predictions.jsonl --label q4_k_m
python3 abstain_eval.py --model model-q4_k_m.gguf --eval eval_set.jsonl --predictions predictions.jsonl
```

Raw numbers, predictions and the audit are in `results/`.

## Licence

MIT.
