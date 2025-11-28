# 🚀 GPT-2 (124M) — Rebuilt from Scratch GPT-2

A **minimal**, **high-performance**, and **fully functional** reproduction of GPT-2 (124M parameters) implemented in **PyTorch**, inspired by Karpathy’s *“Let’s Build GPT”*.

Engineered with **modern optimizations**:
Flash Attention ⚡ | `torch.compile()` 🚀 | BF16 Mixed Precision 💡 | Fused AdamW 🔥 | Efficient Sharded Data Loading 📦

---

## 🌟 Why This Repo Exists

GPT-2 is still the **sweet spot** for:

* Learning LLM internals **without drowning in 1B+ parameters**
* Training on consumer hardware or Kaggle GPUs
* Experimenting with performance tricks used in modern GPT models

This codebase:

* **Faithfully reproduces GPT-2 architecture**
* **Trains faster** & **uses less memory**
* Enables **hands-on research** and **model extension**

---

## 🧠 Architecture & Implementation Choices

| Feature                      | What It Means                        | Why It Matters                                    |
| ---------------------------- | ------------------------------------ | ------------------------------------------------- |
| **Decoder-Only Transformer** | No encoder / cross-attention         | Standard for generative LLMs                      |
| **Pre-LayerNorm**            | Normalize before attention & MLP     | Better gradient flow, stabilizes deep networks    |
| **Weight Tying**             | Token embeddings = output projection | Fewer parameters, improved coherence              |
| **GPT-2 Init**               | Normal(0, 0.02)                      | Correct loss scaling at start of training (~10.8) |
| **Residual Scaling**         | Each block scaled by 1/√(2L)         | Avoids variance blow-up                           |
| **Flash Attention**          | `scaled_dot_product_attention`       | Huge speed + memory savings                       |
| **torch.compile()**          | Kernel fusion & runtime optimization | Up to **2.3× acceleration**                       |
| **BF16 Training**            | Autocast, no GradScaler              | Faster + cheaper training with stable numerics    |
| **Fused AdamW**              | Single-kernel optimizer              | Better GPU utilization                            |
| **Cosine LR + Warm-up**      | Proven stable LLM training schedule  | Prevents collapse early in training               |

> ⚠️ First loss should ≈ **10.8**
> (i.e., `-ln(1 / vocab_size)` with vocab=50257).
> Anything else? Something’s wrong.

---

## 📂 Project Structure

```
/
├── train_gpt2.py      # GPT-2 model + training loop w/ optimizations
├── fineweb.py         # Tokenize + shard FineWeb-Edu dataset
├── out/               # Saved checkpoints
│   └── out_ckpt.pt    # Trained model weights (Git LFS)
└── README.md          # You're reading it!
```

---

## ⚙️ Setup

### 1️⃣ Install Dependencies

```bash
pip install torch numpy transformers datasets tiktoken tqdm requests
```

### 2️⃣ Enable Git LFS

```bash
git lfs install
git lfs track "*.pt"
```

---

## 📚 Dataset — FineWeb-Edu

A high-quality web-scale dataset curated for LLM training.

### Run Data Preprocessing

```bash
python fineweb.py
```

This creates a folder:

```
fineweb/
 ├── edufineweb_train_000001.bin
 ├── ...
```

> Each shard ≈ 100M tokens — efficient streaming for long training runs.

---

## 🚂 Training the Model

Default configuration:
**12 layers | 12 heads | 768 hidden dim | ~124M params**

### Standard Single-GPU Training

```bash
python train_gpt2.py
```

### Unlock Max Performance

(Toggle inside script or via CLI)

| Feature               | CLI Example            |
| --------------------- | ---------------------- |
| Compiled Model        | `--compile_model=True` |
| Mixed Precision BF16  | `--dtype=bfloat16`     |
| Gradient Accumulation | `--grad_accum_steps=8` |

### Multi-GPU Training (DDP)

```bash
torchrun --standalone --nproc_per_node=2 train_gpt2.py
```

DDP logic already handled — including
smart `require_backward_grad_sync` toggling during accumulation.

---

## 🤖 Text Generation (Inference)

Example script logic:

```python
model = GPT2().from_checkpoint("out/out_ckpt.pt")
print(model.generate("Hello world,"))
```

> Use your trained model for **interactive generation demos**!

---

## 🧪 Validation & Debugging

| Check                | Expected Value                         |
| -------------------- | -------------------------------------- |
| Initial Loss         | ~10.8                                  |
| FP16 instability?    | BF16 fixes it                          |
| Exploding gradients? | `grad_clip=1.0` included               |
| Bad generalization?  | Weight decay only on **2D parameters** |

---


## 🏗️ Future Work Ideas

* ✨ Extend to GPT-2 Medium/XL
* 🧩 Add LoRA for cheap fine-tuning
* 📈 Integrate WandB logging
* 🌐 Train with rotary embeddings & sliding attention
* 💬 Fine-tune on dialog datasets for chatbot usage

---

## 🙏 Acknowledgments

* **Andrej Karpathy** — nanoGPT architecture & awesome educational resources
* **OpenAI Authors** — GPT-2 architecture & original weights
* **Hugging Face** — FineWeb-Edu dataset + tokenizer tooling

---

## 📜 License

MIT — free to use, modify, and build upon.
