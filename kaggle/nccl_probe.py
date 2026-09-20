import os, subprocess, time

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()

print(sh("nvidia-smi -L"))
print(sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv"))
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(" ", i, torch.cuda.get_device_name(i))
print(sh("python -c \"import transformers; print('transformers', transformers.__version__)\""))
print(sh("pip list 2>/dev/null | grep -i -E '^(peft|datasets|accelerate|bitsandbytes|trl|safetensors) '"))
print(sh("curl -s -o /dev/null -w 'huggingface.co HTTP %{http_code}' --max-time 15 https://huggingface.co"))

import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank, world):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29511")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    x = torch.ones(1 << 26, device="cuda")  # 256 MB of float32
    for _ in range(3):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(10):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.time() - t) / 10
    if rank == 0:
        gb = 256e6 * 2 * (world - 1) / world / 1e9
        print(f"NCCL all-reduce of 256 MB across {world} GPUs: {dt * 1000:.1f} ms per call, about {gb / dt:.2f} GB/s bus bandwidth")
    dist.destroy_process_group()

if __name__ == "__main__":
    n = torch.cuda.device_count()
    if n >= 2:
        mp.spawn(worker, args=(n,), nprocs=n)
    else:
        print("fewer than 2 GPUs: no NCCL test")
