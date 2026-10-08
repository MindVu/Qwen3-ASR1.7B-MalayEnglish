import re
import matplotlib.pyplot as plt

log_file = "/home/users/ntu/csducmin/scratch/Qwen3-ASR1.7B-MalayEnglish/runs/lora_r16/train_log.txt"

steps_train = []
loss_train = []

steps_eval = []
loss_eval = []

steps_mem = []
mem_alloc = []
mem_reserved = []
mem_peak = []

# Regex patterns to match lines like:
# [step 59] loss=0.74 grad_norm=... | gpu_mem_alloc_gb=4.41 gpu_mem_reserved_gb=31.73 gpu_mem_peak_gb=28.52
# [step 60] eval_loss=0.428481 eval_runtime=...
train_pattern = re.compile(r"\[step (\d+)\].*loss=([0-9.]+).*gpu_mem_alloc_gb=([0-9.]+) gpu_mem_reserved_gb=([0-9.]+) gpu_mem_peak_gb=([0-9.]+)")
eval_pattern = re.compile(r"\[step (\d+)\].*eval_loss=([0-9.]+)")

with open(log_file, "r") as f:
    for line in f:
        # Match eval loss (must do this first or exclude from train match)
        eval_match = eval_pattern.search(line)
        if eval_match:
            step = int(eval_match.group(1))
            loss = float(eval_match.group(2))
            steps_eval.append(step)
            loss_eval.append(loss)
            continue
            
        # Match training loss and memory metrics (ignoring the end-of-run train_loss summary)
        train_match = train_pattern.search(line)
        if train_match and "train_loss=" not in line:
            step = int(train_match.group(1))
            loss = float(train_match.group(2))
            alloc = float(train_match.group(3))
            reserved = float(train_match.group(4))
            peak = float(train_match.group(5))
            
            steps_train.append(step)
            loss_train.append(loss)
            
            steps_mem.append(step)
            mem_alloc.append(alloc)
            mem_reserved.append(reserved)
            mem_peak.append(peak)

# ---------------------------------------------------------
# 1. Plot Training and Evaluation Loss
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(steps_train, loss_train, label='Train Loss', alpha=0.7)
if steps_eval:
    plt.plot(steps_eval, loss_eval, label='Eval Loss', marker='o', color='red')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('loss_plot.png', dpi=300)
print("Saved loss plot to 'loss_plot.png'")

# ---------------------------------------------------------
# 2. Plot GPU Memory Usage
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(steps_mem, mem_alloc, label='Allocated GB', linewidth=2)
plt.plot(steps_mem, mem_reserved, label='Reserved GB', linewidth=2)
plt.plot(steps_mem, mem_peak, label='Peak GB', linewidth=2, linestyle='--')
plt.xlabel('Steps')
plt.ylabel('GPU Memory (GB)')
plt.title('GPU Memory Usage Over Time')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('memory_plot.png', dpi=300)
print("Saved memory plot to 'memory_plot.png'")
