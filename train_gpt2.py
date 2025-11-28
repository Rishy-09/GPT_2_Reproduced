# train.py
# A robust training script for Andrej Karpathy's nanoGPT, adapted for Kaggle.
# This script handles large, sharded datasets (like FineWeb), checkpointing,
# distributed training, and integrated HellaSwag evaluation.

import os
import time
import math
import pickle
import glob
import json
import requests
import inspect
from dataclasses import dataclass
from contextlib import nullcontext
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
# DDP specific imports
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import tiktoken
import wandb

# -----------------------------------------------------------------------------
# DDP Setup Functions
# -----------------------------------------------------------------------------
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# -----------------------------------------------------------------------------
# EarlyStopping Class for Overfitting Mitigation
# -----------------------------------------------------------------------------
class EarlyStopping:
    """
    Implements early stopping to halt training when validation loss ceases to improve.
    """
    def __init__(self, patience=10, verbose=False, delta=0, path='best_checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model, optimizer, iter_num, best_val_loss, scaler, master_process):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if master_process:
                self.save_checkpoint(val_loss, model, optimizer, iter_num, best_val_loss, scaler)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose and master_process:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if master_process:
                self.save_checkpoint(val_loss, model, optimizer, iter_num, best_val_loss, scaler)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, iter_num, best_val_loss, scaler):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving best model to {self.path}...')
        
        raw_model = model.module if hasattr(model, 'module') else model
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': raw_model.config,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'scaler': scaler.state_dict(),
        }
        torch.save(checkpoint, self.path)
        self.val_loss_min = val_loss

# -----------------------------------------------------------------------------
# Sharded DataLoader for Large Datasets
# -----------------------------------------------------------------------------
class DataChunk(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx+self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx+1:idx+1+self.block_size].astype(np.int64))
        return x, y

class ShardedDataLoader:
    def __init__(self, data_base_path, batch_size, block_size, split, ddp_rank=0, ddp_world_size=1, master_process=True):
        self.data_base_path = data_base_path
        self.batch_size = batch_size
        self.block_size = block_size
        self.split = split
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.master_process = master_process

        shards_candidate1 = sorted(glob.glob(os.path.join(self.data_base_path, f"edufineweb_{split}_*.npy")))
        shards_candidate2 = sorted(glob.glob(os.path.join(self.data_base_path, f"fineweb_{split}_*.npy")))
        shards_candidate3 = sorted(glob.glob(os.path.join(self.data_base_path, f"{split}_*.npy")))

        if len(shards_candidate1) > 0:
            self.shards = shards_candidate1
            if self.master_process:
                print(f"Found {len(self.shards)} shards for split {split} using 'edufineweb_{split}_*.npy' pattern.")
        elif len(shards_candidate2) > 0:
            self.shards = shards_candidate2
            if self.master_process:
                print(f"Found {len(self.shards)} shards for split {split} using 'fineweb_{split}_*.npy' pattern.")
        elif len(shards_candidate3) > 0:
            self.shards = shards_candidate3
            if self.master_process:
                print(f"Found {len(self.shards)} shards for split {split} using '{split}_*.npy' pattern.")
        else:
            raise FileNotFoundError(f"No shards found for split {split} at base path {self.data_base_path}")
        
        self.reset()

    def reset(self):
        self.current_shard_idx = 0
        self.current_loader = self._get_loader_for_shard(self.current_shard_idx)
        self.iter_loader = iter(self.current_loader)

    def _get_loader_for_shard(self, shard_idx):
        filepath = self.shards[shard_idx]
        if self.master_process:
            print(f"Loading data from shard: {filepath}")

        data = np.memmap(filepath, dtype=np.uint16, mode='r')
        dataset = DataChunk(data, self.block_size)
        
        if len(dataset) == 0:
            if self.master_process:
                print(f"Warning: Shard {filepath} results in an empty dataset for block_size {self.block_size}. Returning an empty DataLoader.")
            return DataLoader([], batch_size=self.batch_size)
        
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=self.ddp_world_size, rank=self.ddp_rank, shuffle=True
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, pin_memory=True, num_workers=0)
        return loader

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                return next(self.iter_loader)
            except StopIteration:
                self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shards)
                if self.master_process:
                    print(f"Shard exhausted. Moving to shard {self.shards[self.current_shard_idx]}.")
                self.current_loader = self._get_loader_for_shard(self.current_shard_idx)
                # Important: update the sampler's epoch to ensure proper shuffling
                if hasattr(self.current_loader.sampler, 'set_epoch'):
                    # The epoch can be any number; iter_num is a good choice to ensure it changes
                    # This is handled in the training loop's estimate_loss function
                    pass 
                self.iter_loader = iter(self.current_loader)

# -----------------------------------------------------------------------------
# Model Definitions
# -----------------------------------------------------------------------------
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
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, master_process):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # Create optimizer and use fused version if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        if master_process:
            print(f"Using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# -----------------------------------------------------------------------------
# HellaSwag Functions
# -----------------------------------------------------------------------------
def download_file(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

hellaswags = {
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
}

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

# -----------------------------------------------------------------------------
# Main Training Function (main_worker)
# -----------------------------------------------------------------------------
def main_worker(rank, world_size, config):
    setup(rank, world_size)
    
    device = f'cuda:{rank}'
    torch.cuda.set_device(device)

    master_process = (rank == 0)

    # Extract config parameters
    out_dir = config['out_dir']
    data_dir = config['data_dir']
    eval_hellaswag_cache_dir = config['eval_hellaswag_cache_dir']
    eval_interval = config['eval_interval']
    log_interval = config['log_interval']
    eval_iters = config['eval_iters']
    eval_only = config['eval_only']
    always_save_checkpoint = config['always_save_checkpoint']
    init_from = config['init_from']
    resume_from_path = config['resume_from_path']
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    batch_size = config['batch_size']
    block_size = config['block_size']
    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    dropout = config['dropout']
    bias = config['bias']
    learning_rate = config['learning_rate']
    max_iters = config['max_iters']
    weight_decay = config['weight_decay']
    beta1 = config['beta1']
    beta2 = config['beta2']
    grad_clip = config['grad_clip']
    decay_lr = config['decay_lr']
    warmup_iters = config['warmup_iters']
    lr_decay_iters = config['lr_decay_iters']
    min_lr = config['min_lr']
    compile_model = config['compile_model']
    early_stopping_patience = config['early_stopping_patience']
    wandb_log = config['wandb_log']
    wandb_project = config['wandb_project']
    wandb_run_name = config['wandb_run_name']

    ddp_rank = rank
    ddp_world_size = world_size
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps_per_device = gradient_accumulation_steps // ddp_world_size

    tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
    if master_process:
        print(f"Tokens per iteration (global): {tokens_per_iter:,}")
        os.makedirs(out_dir, exist_ok=True)
        if wandb_log:
            wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = 'cuda'
    dtype = 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    train_loader = ShardedDataLoader(data_dir, batch_size, block_size, 'train', ddp_rank, ddp_world_size, master_process)
    val_loader = ShardedDataLoader(data_dir, batch_size, block_size, 'val', ddp_rank, ddp_world_size, master_process)
    
    enc = tiktoken.get_encoding("gpt2")
    meta_vocab_size = 50257
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']

    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=meta_vocab_size, dropout=dropout)
    
    iter_num = 0
    best_val_loss = 1e9
    checkpoint = None

    # --- Model/Optimizer Initialization ---
    if init_from == 'scratch':
        if master_process: print("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume':
        ckpt_path = resume_from_path if resume_from_path else os.path.join(out_dir, 'ckpt.pt')
        # All processes check for the file, preventing deadlocks
        if os.path.exists(ckpt_path):
            if master_process: print(f"Resuming training from checkpoint: {ckpt_path}")
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(ckpt_path, map_location=map_location)
            
            # Ensure model args match
            checkpoint_model_args = checkpoint['model_args']
            for k, v in model_args.items():
                assert checkpoint_model_args[k] == v, f"Model arg mismatch: {k}, ckpt:{checkpoint_model_args[k]} vs conf:{v}"

            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
            state_dict = checkpoint['model']
            
            # Fix for state dicts saved from compiled models
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']
        else:
            if master_process: print(f"WARNING: Checkpoint file not found at {ckpt_path}. Starting from scratch.")
            init_from = 'scratch' # Fallback to scratch
    
    # Fallback to scratch if resume failed
    if init_from == 'scratch':
        if master_process: print("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

    model.to(device)

    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type, master_process)
    if init_from == 'resume' and checkpoint and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        if master_process: print("Optimizer state loaded from checkpoint.")

    scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
    if init_from == 'resume' and checkpoint and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
        if master_process: print("GradScaler state loaded from checkpoint.")

    if compile_model:
        if master_process: print("Compiling the model...")
        model = torch.compile(model)

    model = DDP(model, device_ids=[rank])
    if master_process: print(f"Model wrapped in DDP on device {rank}.")
    raw_model = model.module

    def get_lr(it):
        if it < warmup_iters: return learning_rate * it / warmup_iters
        if it > lr_decay_iters: return min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters, device=device)
            loader = train_loader if split == 'train' else val_loader
            
            # Ensure the sampler's epoch is updated for proper shuffling in DDP
            if hasattr(loader.current_loader.sampler, 'set_epoch'):
                loader.current_loader.sampler.set_epoch(iter_num)
            
            for k in range(eval_iters):
                try:
                    X, Y = next(loader)
                except StopIteration: # Handle exhausted loader
                    loader.reset()
                    if hasattr(loader.current_loader.sampler, 'set_epoch'):
                        loader.current_loader.sampler.set_epoch(iter_num)
                    X, Y = next(loader)

                X, Y = X.to(device), Y.to(device)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            
            dist.all_reduce(losses, op=dist.ReduceOp.SUM)
            losses /= ddp_world_size
            
            out[split] = losses.mean()
        model.train()
        train_loader.reset()
        val_loader.reset()
        return out

    @torch.no_grad()
    def evaluate_hellaswag():
        model.eval()
        num_correct_norm = 0
        num_total = 0
        
        hellaswag_examples = list(iterate_examples("val", eval_hellaswag_cache_dir))
        
        if master_process: print(f"Evaluating HellaSwag on {len(hellaswag_examples)} examples...")

        for i, example in enumerate(hellaswag_examples):
            if i % ddp_world_size != ddp_rank:
                continue
            
            tokens, mask, label = render_example(example, enc)
            tokens = tokens.to(device)
            mask = mask.to(device)
            
            with ctx:
                logits, _ = model(tokens)
            
            pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        
        total_correct = torch.tensor(num_correct_norm, device=device)
        total_examples = torch.tensor(num_total, device=device)
        dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_examples, op=dist.ReduceOp.SUM)
        acc_norm = total_correct.item() / total_examples.item() if total_examples.item() > 0 else 0.0

        model.train()
        return acc_norm

    early_stopper = EarlyStopping(patience=early_stopping_patience, verbose=True, path=os.path.join(out_dir, 'best_ckpt.pt'))
    t0 = time.time()

    if master_process: print("Starting training loop...")

    # Main training loop
    while True:
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % eval_interval == 0:
            losses = estimate_loss()
            if master_process:
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if wandb_log:
                    wandb.log({
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "lr": lr,
                    }, step=iter_num)

            # NOTE: HellaSwag and generation are skipped with compile=True due to potential
            # issues with dynamic shapes and control flow in compiled models.
            if not compile_model:
                acc_norm = evaluate_hellaswag()
                if master_process:
                    print(f"HellaSwag accuracy: {acc_norm:.4f}")
                    if wandb_log:
                        wandb.log({"hellaswag/acc": acc_norm}, step=iter_num)

            early_stopper(losses['val'], model, optimizer, iter_num, best_val_loss, scaler, master_process)
            if early_stopper.early_stop:
                if master_process: print("Early stopping triggered.")
                break

            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0 and master_process:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'scaler': scaler.state_dict()
                    }
                    ckpt_save_path = os.path.join(out_dir, 'ckpt.pt')
                    print(f"Saving checkpoint to {ckpt_save_path}")
                    torch.save(checkpoint, ckpt_save_path)
            
            if not compile_model and master_process:
                print("Generating a sample...")
                start_ids = enc.encode("\n", allowed_special={"<|endoftext|>"})
                x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None,...])
                with torch.no_grad(), ctx:
                    y = raw_model.generate(x, max_new_tokens=100, temperature=0.8, top_k=200)
                    print(enc.decode(y[0].tolist()))
                    print('---------------')

        if iter_num == 0 and eval_only:
            if master_process: print("Eval only mode, exiting after first evaluation.")
            break
        
        # Training step
        for micro_step in range(gradient_accumulation_steps_per_device):
            # DDP sync only on last micro-step
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps_per_device - 1)
            try:
                X, Y = next(train_loader)
            except StopIteration:
                train_loader.reset()
                X, Y = next(train_loader)
            
            X, Y = X.to(device), Y.to(device)

            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps_per_device
            scaler.scale(loss).backward()
        
        if grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps_per_device
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}")
            if wandb_log:
                wandb.log({"train/iter_loss": lossf, "time_ms": dt*1000}, step=iter_num)
        
        iter_num += 1

        if iter_num > max_iters:
            if master_process: print(f"Max iterations ({max_iters}) reached. Halting training.")
            break
    
    if wandb_log and master_process:
        wandb.finish()
    cleanup()
    if master_process: print("DDP process group destroyed.")

# -----------------------------------------------------------------------------
# Main entry point for the script
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    config = {
        'out_dir': '/kaggle/working/out',
        'data_dir': '/kaggle/input/fineweb/fineweb', # Example path
        'eval_hellaswag_cache_dir': os.path.join('/kaggle/working/out', 'hellaswag'),
        'eval_interval': 250,
        'log_interval': 10,
        'eval_iters': 100,
        'eval_only': False,
        'always_save_checkpoint': True,
        'init_from': 'resume', # 'scratch' or 'resume'
        'resume_from_path': "/kaggle/input/checkpoints/ckpt.pt", # full path to ckpt.pt if resuming
        'gradient_accumulation_steps': 16,
        'batch_size': 16,
        'block_size': 1024,
        'n_layer': 12,
        'n_head': 12,
        'n_embd': 768,
        'dropout': 0.0,
        'bias': False,
        'learning_rate': 6e-4,
        'max_iters': 600000, # Reduced for faster testing; original was 600000
        'weight_decay': 1e-1,
        'beta1': 0.9,
        'beta2': 0.95,
        'grad_clip': 1.0,
        'decay_lr': True,
        'warmup_iters': 2000,
        'lr_decay_iters': 600000, # Match max_iters
        'min_lr': 6e-5,
        'compile_model': True,
        'early_stopping_patience': 200, # In units of eval_interval
        'wandb_log': False, # Set to True to use wandb
        'wandb_project': 'nanogpt-fineweb',
        'wandb_run_name': 'run-1'
    }

    # NOTE: You may need to adjust data_dir to your actual Kaggle input directory.
    # e.g., '/kaggle/input/my-dataset-name/...'

    # Determine the number of available GPUs
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        print("This script is designed for multi-GPU DDP training. Running on a single process.")
        world_size = 1
        # Run directly without spawning if only one GPU or CPU
        main_worker(0, world_size, config)
    else:
        print(f"Found {n_gpus} GPU(s). Setting up DDP training.")
        world_size = n_gpus
        # Launch DDP processes
        mp.spawn(main_worker,
                 args=(world_size, config),
                 nprocs=world_size,
                 join=True)

    print("All training processes finished.")