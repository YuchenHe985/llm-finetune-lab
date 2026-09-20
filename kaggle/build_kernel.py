"""Builds the single-file Kaggle script (the Kaggle CLI pushes one code file) that embeds train.py, runs it
with 1 and 2 GPUs through torchrun and prints a comparison.

    KAGGLE_USERNAME=<you> python3 kaggle/build_kernel.py smoke   # a tiny run to catch errors
    KAGGLE_USERNAME=<you> python3 kaggle/build_kernel.py full    # the measured run
"""
import json
import os
import pathlib
import sys

mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
cfg = {"smoke": {"train_n": 96, "eval_n": 16, "max_len": 256, "save_merged": False},
       "full": {"train_n": 8000, "eval_n": 400, "max_len": 384, "save_merged": True}}[mode]
train = (pathlib.Path(__file__).resolve().parent.parent / "train.py").read_text()
script = f'''import json, os, subprocess, sys, time
CFG = {cfg!r}
TRAIN = {train!r}
os.chdir("/kaggle/working")
# the image ships a torchao that peft/transformers reject as too old; they do not need it
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
open("train.py", "w").write(TRAIN)

def run(world):
    cmd = ["torchrun", "--standalone", "--nproc_per_node", str(world), "train.py",
           "--train-n", str(CFG["train_n"]), "--eval-n", str(CFG["eval_n"]), "--max-len", str(CFG["max_len"]),
           "--per-device-bs", "4", "--out", f"results_{{world}}gpu.json"]
    if world == 1:
        cmd.append("--eval-before")
        if CFG["save_merged"]:
            cmd += ["--save-merged", "merged"]
    print("RUN", " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, env=dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
    print(f"finished world={{world}} in {{time.time() - t0:.0f}}s wall (includes model load and evaluation)", flush=True)

for w in (1, 2):
    run(w)
r1 = json.load(open("results_1gpu.json"))
r2 = json.load(open("results_2gpu.json"))
summary = {{
    "gpu": r1["gpu"], "trainable_params": r1["trainable_params"], "total_params": r1["total_params"],
    "eval_before": r1.get("eval_before"), "eval_after_1gpu": r1["eval_after"], "eval_after_2gpu": r2["eval_after"],
    "seconds_per_step": {{"1gpu": r1["seconds_per_step"], "2gpu": r2["seconds_per_step"]}},
    "speedup_2gpu_vs_1gpu": r1["seconds_per_step"] / r2["seconds_per_step"],
    "scaling_efficiency": r1["seconds_per_step"] / r2["seconds_per_step"] / 2,
    "tokens_per_second": {{"1gpu": r1["tokens_per_second"], "2gpu": r2["tokens_per_second"]}},
    "peak_memory_gib_per_gpu": {{"1gpu": r1["peak_memory_gib_per_gpu"], "2gpu": r2["peak_memory_gib_per_gpu"]}},
    "loss_first10_last10_1gpu": [r1["loss_first10"], r1["loss_last10"]],
    "loss_first10_last10_2gpu": [r2["loss_first10"], r2["loss_last10"]],
}}
json.dump(summary, open("summary.json", "w"), indent=2)
print("SUMMARY", json.dumps(summary, indent=2), flush=True)
'''
out = pathlib.Path("kernel")  # then: kaggle kernels push -p kernel
out.mkdir(exist_ok=True)
(out / "run_kaggle.py").write_text(script)
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": os.environ.get("KAGGLE_USERNAME", "your-username") + "/llm-finetune-lab", "title": "llm finetune lab", "code_file": "run_kaggle.py",
    "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
    "enable_tpu": "false", "enable_internet": "true", "dataset_sources": [], "competition_sources": [],
    "kernel_sources": [], "model_sources": []}, indent=2))
print(f"built kernel/run_kaggle.py ({mode})")
