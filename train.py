"""LoRA fine-tuning of a small causal LM for text-to-SQL, on one GPU or several with DistributedDataParallel.

Run with torchrun so the same script covers 1 and 2 GPUs:

    torchrun --standalone --nproc_per_node=2 train.py --out results_2gpu.json

The global batch size is fixed, so 1 GPU uses more gradient-accumulation steps than 2 GPUs and both runs
see the same data in the same order; what changes is the wall-clock time.
"""
import argparse
import contextlib
import json
import os
import random
import re
import time

import torch
import torch.distributed as dist
import datasets
import peft
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = "You translate questions into SQL for the given schema. Reply with one SQL statement only."


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dataset", default="b-mc2/sql-create-context")
    ap.add_argument("--train-n", type=int, default=8000)
    ap.add_argument("--eval-n", type=int, default=400)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--per-device-bs", type=int, default=4,
                    help="micro-batch per GPU; 4 is the measured T4-safe setting")
    ap.add_argument("--global-batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank-r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-before", action="store_true", help="also evaluate before training (rank 0)")
    ap.add_argument("--save-merged", default="", help="directory for the merged model (rank 0)")
    ap.add_argument("--out", default="results.json")
    return ap.parse_args()


def encode(tok, ex, max_len):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Schema:\n{ex['context']}\n\nQuestion: {ex['question']}"}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(ex["answer"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return (p + a)[:max_len], ([-100] * len(p) + a)[:max_len], p


def collate(rows, pad_id, device):
    n = max(len(r[0]) for r in rows)
    ids = torch.full((len(rows), n), pad_id, dtype=torch.long)
    lab = torch.full((len(rows), n), -100, dtype=torch.long)
    att = torch.zeros((len(rows), n), dtype=torch.long)
    for i, (x, y, _) in enumerate(rows):
        ids[i, :len(x)], lab[i, :len(y)], att[i, :len(x)] = torch.tensor(x), torch.tensor(y), 1
    return ids.to(device), lab.to(device), att.to(device)


def norm_sql(s):
    """Compare SQL, not formatting: drop markdown fences, whitespace, case and a trailing semicolon."""
    m = re.search(r"```(?:sql)?\s*(.*?)```", s, re.S | re.I)
    if m:
        s = m.group(1)
    return re.sub(r"\s+", " ", s.strip().rstrip(";").lower())


@torch.no_grad()
def evaluate(model, tok, eval_rows, eval_raw, device, bs=16):
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    for i in range(0, len(eval_rows), bs):
        ids, lab, att = collate(eval_rows[i:i + bs], tok.pad_token_id, device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids=ids, attention_mask=att).logits
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                                                 lab[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        tot_loss += loss.item()
        tot_tok += (lab[:, 1:] != -100).sum().item()
    exact = 0
    tok.padding_side = "left"
    for i in range(0, len(eval_rows), bs):
        prompts = [tok.decode(r[2]) for r in eval_rows[i:i + bs]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model.generate(**enc, max_new_tokens=128, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, seq in enumerate(out):
            text = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            exact += norm_sql(text) == norm_sql(eval_raw[i + j]["answer"])
    model.train()
    return {"loss": tot_loss / tot_tok, "perplexity": float(torch.exp(torch.tensor(tot_loss / tot_tok))),
            "exact_match": exact / len(eval_rows)}


def main():
    args = parse()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    raw = load_dataset(args.dataset, split="train").shuffle(seed=1)
    eval_raw = [raw[i] for i in range(args.eval_n)]
    train_raw = [raw[i] for i in range(args.eval_n, args.eval_n + args.train_n)]
    train_rows = [encode(tok, ex, args.max_len) for ex in train_raw]
    eval_rows = [encode(tok, ex, args.max_len) for ex in eval_raw]

    # fp32 master weights with fp16 autocast: robust for Qwen, which can overflow in pure fp16
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
    lora = LoraConfig(r=args.rank_r, lora_alpha=args.alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    peft_model = get_peft_model(base, lora)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    model = DDP(peft_model, device_ids=[local]) if world > 1 else peft_model
    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")

    result = {"world_size": world, "model": args.model,
              "software": {"torch": torch.__version__, "transformers": transformers.__version__,
                           "peft": peft.__version__, "datasets": datasets.__version__},
              "trainable_params": trainable, "total_params": total,
              "train_examples": len(train_rows), "eval_examples": len(eval_rows), "global_batch": args.global_batch,
              "per_device_bs": args.per_device_bs, "gpu": torch.cuda.get_device_name(local)}
    if args.eval_before and rank == 0:
        result["eval_before"] = evaluate(peft_model, tok, eval_rows, eval_raw, device)
        print("before:", result["eval_before"], flush=True)
    if world > 1:
        dist.barrier()

    accum = max(1, args.global_batch // (args.per_device_bs * world))
    steps = len(train_rows) // args.global_batch
    losses, step_times, tokens = [], [], 0
    peft_model.train()
    for epoch in range(args.epochs):
        order = list(range(len(train_rows)))
        random.Random(args.seed + epoch).shuffle(order)
        for step in range(steps):
            t0 = time.time()
            mine = order[step * args.global_batch:(step + 1) * args.global_batch][rank::world]
            step_loss = 0.0
            for m in range(accum):
                chunk = [train_rows[k] for k in mine[m * args.per_device_bs:(m + 1) * args.per_device_bs]]
                ids, lab, att = collate(chunk, tok.pad_token_id, device)
                sync = contextlib.nullcontext() if (world == 1 or m == accum - 1) else model.no_sync()
                with sync:
                    with torch.autocast("cuda", dtype=torch.float16):
                        out = model(input_ids=ids, attention_mask=att, labels=lab)
                    scaler.scale(out.loss / accum).backward()
                step_loss += out.loss.item() / accum
                tokens += int(att.sum().item())
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in peft_model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step_times.append(time.time() - t0)
            losses.append(step_loss)
            if rank == 0 and step % 25 == 0:
                print(f"step {step}/{steps} loss {step_loss:.4f} {step_times[-1]:.2f}s", flush=True)

    warm = min(3, len(step_times) // 2)
    train_time = sum(step_times[warm:])
    tok_t = torch.tensor([float(tokens)], device=device)
    mem = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**30], device=device)
    if world > 1:
        dist.all_reduce(tok_t)
        dist.all_reduce(mem, op=dist.ReduceOp.MAX)
    result.update({"steps": steps * args.epochs, "grad_accumulation": accum, "train_seconds_after_warmup": train_time,
                   "seconds_per_step": train_time / max(1, len(step_times) - warm),
                   "tokens_per_second": tok_t.item() * (len(step_times) - warm) / max(1, len(step_times)) / max(train_time, 1e-9),
                   "peak_memory_gib_per_gpu": mem.item(),
                   "loss_first10": sum(losses[:10]) / len(losses[:10]), "loss_last10": sum(losses[-10:]) / len(losses[-10:])})
    if rank == 0:
        result["eval_after"] = evaluate(peft_model, tok, eval_rows, eval_raw, device)
        print("after:", result["eval_after"], flush=True)
        if args.save_merged:
            merged = peft_model.merge_and_unload()
            merged.to(torch.float16).save_pretrained(args.save_merged, safe_serialization=True)
            tok.save_pretrained(args.save_merged)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
