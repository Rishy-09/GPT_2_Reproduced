 # --- Imports from both train_gpt2.py and hellaswag.py ---
import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel as DDP

# --- Model Definitions (Copied from train_gpt2.py) ---
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_embd).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_embd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        print("loading from HuggingFace GPT-2 with those config_args:", config_args)
        # We need to manually set the vocab_size and block_size
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        # create a new model and load the weights
        model_hf = GPT2LMHeadModel.from_pretrained(model_type, **config_args)
        sd_hf = model_hf.state_dict()
        model_args = GPTConfig(**config_args)
        model = cls(model_args)
        sd = model.state_dict()
        # copy over the weights from the pre-trained model
        keys_hf = list(sd_hf.keys())
        keys_sd = list(sd.keys())
        # print("keys_hf len", len(keys_hf), "keys_sd len", len(keys_sd))
        # print("keys_hf", keys_hf)
        # print("keys_sd", keys_sd)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        for k in transposed:
            if k in sd_hf and k in sd:
                # print("transposing", k)
                sd_hf[k] = sd_hf[k].T

        # special treatment for position embedding
        if 'transformer.wpe.weight' in sd_hf and 'transformer.wpe.weight' in sd:
            sd_hf['transformer.wpe.weight'] = sd_hf['transformer.wpe.weight'][:model_args.block_size]

        # special treatment for vocab embedding
        if 'transformer.wte.weight' in sd_hf and 'transformer.wte.weight' in sd:
            sd_hf['transformer.wte.weight'] = sd_hf['transformer.wte.weight'][:model_args.vocab_size]

        # copy the weights
        sd.update(sd_hf)
        model.load_state_dict(sd)

        # check for keys that are not copied
        for k in sd_hf.keys():
            if k not in sd:
                print(f"skipping key {k} from hf model")

        # override other fields
        # if 'dropout' in override_args:
        #     model_args.dropout = override_args['dropout']

        return model

# --- HellaSwag Helper Functions ---
DATA_CACHE_DIR = os.path.join('/kaggle/working', 'hellaswag_data_cache')

def download_file(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

hellaswags = {
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
}

enc = tiktoken.get_encoding('gpt2')

def download_hellaswag_val(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    data_url = hellaswags["val"]
    data_filename = os.path.join(cache_dir, "hellaswag_val.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    return data_filename

def iterate_examples(split, cache_dir):
    data_filename = download_hellaswag_val(cache_dir)
    with open(data_filename, "r") as f:
        for line in f:
            example = json.loads(line)
            yield example

def render_example(example, enc):
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]
    ctx_tokens = enc.encode(ctx)
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))

    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return tokens, mask, label

def get_most_likely_row(tokens, mask, logits):
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    pred_norm = avg_loss.argmin().item()
    return pred_norm

@torch.no_grad()
def evaluate_hellaswag(model, device):
    torch.set_float32_matmul_precision('high')
    model.eval()

    num_correct_norm = 0
    num_total = 0
    
    hellaswag_cache_dir = os.path.join('/kaggle/working/out', 'hellaswag')
    hellaswag_examples = list(iterate_examples("val", hellaswag_cache_dir))
    
    print(f"Evaluating HellaSwag on {len(hellaswag_examples)} examples...")

    for i, example in enumerate(hellaswag_examples):
        tokens, mask, label = render_example(example, enc)
        tokens = tokens.to(device)
        mask = mask.to(device)
        
        logits, _ = model(tokens)
        
        pred_norm = get_most_likely_row(tokens, mask, logits)
        
        num_total += 1
        num_correct_norm += int(pred_norm == label)
        
        print(f"Example {num_total}: pred={pred_norm}, actual={label}, Correct Norm Acc: {num_correct_norm/num_total:.4f}")

    final_accuracy = num_correct_norm / num_total if num_total > 0 else 0.0
    print("--- Final HellaSwag Evaluation ---")
    print(f"Total Examples: {num_total}")
    print(f"Correct Predictions (Normalized): {num_correct_norm}")
    print(f"Final Normalized Accuracy: {final_accuracy:.4f}")


if __name__ == "__main__":
    # --- Configuration and Checkpoint Loading ---
    ckpt_path = "/kaggle/input/checkpoints/ckpt.pt" # Update this path if needed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint file not found at {ckpt_path}.")
        exit()
        
    print(f"Loading model from checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Load model configuration from checkpoint
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # Load model state dict
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    print("Model loaded successfully.")
    
    model.to(device)
    
    # --- Run the Evaluation ---
    evaluate_hellaswag(model, device)

