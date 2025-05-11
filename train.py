"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# 以下是实验需要
import copy
from perturbation_manager import insert_perturbation_before_dropout, set_perturbation_mode
from datetime import datetime

# 以下是NPU需要
import torch_npu 
from torch_npu.contrib import transfer_to_npu
os.environ['ASCEND_LAUNCH_BLOCKING'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# 用于保存数据
import csv

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

# 使用NPU实验禁用
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    # 保存数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"train_loss_log_{timestamp}.csv")
    csv_file = open(csv_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["iter", "train_loss_clean", "train_loss_noisy", "delta_train_loss"])
    
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# NPU实验，关闭
ctx = torch.cuda.amp.autocast(enabled=False)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss_clean = 1e9
best_val_loss_noisy = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new clean model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt_clean.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss_clean = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

    

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

model_clean = model  # 当前已有的模型就是 clean

# model_noisy init 逻辑和上面完全一致
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new noisy model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf_noisy = GPTConfig(**model_args)
    model_noisy = GPT(gptconf_noisy)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt_noisy.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf_noisy = GPTConfig(**model_args)
    model_noisy = GPT(gptconf_noisy)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model_noisy.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss_noisy = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model_noisy = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

# crop down the model block size if desired, using model surgery
if block_size < model_noisy.config.block_size:
    model_noisy.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model_noisy.to(device)


insert_perturbation_before_dropout(model_clean, mode="none", forward_eps=1e-3, grad_eps=1e-3)
insert_perturbation_before_dropout(model_noisy, mode="both", forward_eps=1e-3, grad_eps=1e-3)

# initialize a GradScaler. If enabled=False scaler is a no-op
# 为了NPU实验，均关闭
scaler_clean = torch.cuda.amp.GradScaler(enabled=False)
scaler_noisy = torch.cuda.amp.GradScaler(enabled=False)

# optimizer
optimizer_clean = model_clean.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
optimizer_noisy = model_noisy.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer_clean.load_state_dict(checkpoint['optimizer_clean'])
    optimizer_noisy.load_state_dict(checkpoint['optimizer_noisy'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the clean and nosiy models ... (takes a ~minute)")
    unoptimized_model_clean = model_clean
    unoptimized_model_noisy = model_noisy
    model_clean = torch.compile(model_clean) # requires PyTorch 2.0
    model_noisy = torch.compile(model_noisy) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model_clean = DDP(model_clean, device_ids=[ddp_local_rank])
    model_noisy = DDP(model_noisy, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out_clean = {}
    out_noisy = {}
    model_clean.eval()
    model_noisy.eval()
    for split in ['train', 'val']:
        losses_clean = torch.zeros(eval_iters)
        losses_noisy = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits_clean, loss_clean = model_clean(X, Y)
                logits_noisy, loss_noisy = model_noisy(X, Y)
            losses_clean[k] = loss_clean.item()
            losses_noisy[k] = loss_noisy.item()
        out_clean[split] = loss_clean.mean()
        out_noisy[split] = loss_noisy.mean()
    model_clean.train()
    model_noisy.train()
    return out_clean, out_noisy

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model_clean = model_clean.module if ddp else model_clean # unwrap DDP container if needed
raw_model_noisy = model_noisy.module if ddp else model_noisy
running_mfu_clean = -1.0
running_mfu_noisy = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer_clean.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_noisy.param_groups:
        param_group['lr'] = lr
        
    # === 设置扰动模式 ===
    set_perturbation_mode(model_clean, "none")
    set_perturbation_mode(model_noisy, "both")

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses_clean, losses_noisy = estimate_loss()
        print(f"[Eval {iter_num}] Clean: train loss ={losses_clean['train']:.4f}, val loss ={losses_clean['val']:.4f}")
        print(f"[Eval {iter_num}] Noisy: train loss ={losses_noisy['train']:.4f}, val loss ={losses_noisy['val']:.4f}")
        print(f"Delta val loss: {losses_noisy['val'] - losses_clean['val']:.4f}")
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss_clean": losses_clean['train'],
                "val/loss_clean": losses_clean['val'],
                "train/loss_noisy": losses_noisy['train'],
                "val/loss_noisy": losses_noisy['val'],
                "delta_val_loss": losses_noisy['val'] - losses_clean['val'],
                "lr": lr,
                "mfu_clean": running_mfu_clean*100,
                "mfu_noisy": running_mfu_noisy*100,
            })
            
        # 不保存checkpoint
        # if losses_clean['val'] < best_val_loss_clean or always_save_checkpoint:
        #     best_val_loss_clean = losses_clean['val']
        #     if iter_num > 0:
        #         checkpoint_clean = {
        #             'model': raw_model_clean.state_dict(),
        #             'optimizer': optimizer_clean.state_dict(),
        #             'model_args': model_args,
        #             'iter_num': iter_num,
        #             'best_val_loss': best_val_loss_clean,
        #             'config': config,
        #         }
        #         print(f"saving clean checkpoint to {out_dir}")
        #         torch.save(checkpoint_clean, os.path.join(out_dir, 'ckpt_clean.pt'))
        # if losses_noisy['val'] < best_val_loss_noisy or always_save_checkpoint:
        #     best_val_loss_noisy = losses_noisy['val']
        #     if iter_num > 0:
        #         checkpoint_noisy = {
        #             'model': raw_model_noisy.state_dict(),
        #             'optimizer': optimizer_noisy.state_dict(),
        #             'model_args': model_args,
        #             'iter_num': iter_num,
        #             'best_val_loss': best_val_loss_noisy,
        #             'config': config,
        #         }
        #         print(f"saving clean checkpoint to {out_dir}")
        #         torch.save(checkpoint_noisy, os.path.join(out_dir, 'ckpt_noisy.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model_clean.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            model_noisy.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits_clean, loss_clean = model_clean(X, Y)
            loss_clean = loss_clean / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            
        with ctx:
            logits_noisy, loss_noisy = model_noisy(X, Y)
            loss_noisy = loss_noisy / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler_clean.scale(loss_clean).backward()
        scaler_noisy.scale(loss_noisy).backward()
    
    # clip the gradient
    if grad_clip != 0.0:
        scaler_clean.unscale_(optimizer_clean)
        torch.nn.utils.clip_grad_norm_(model_clean.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler_clean.step(optimizer_clean)
    scaler_clean.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer_clean.zero_grad(set_to_none=True)
    
    # clip the gradient
    if grad_clip != 0.0:
        scaler_noisy.unscale_(optimizer_noisy)
        torch.nn.utils.clip_grad_norm_(model_noisy.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler_noisy.step(optimizer_noisy)
    scaler_noisy.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer_noisy.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf_clean = loss_clean.item() * gradient_accumulation_steps
        lossf_noisy = loss_noisy.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu_clean = raw_model_clean.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu_clean = mfu_clean if running_mfu_clean == -1.0 else 0.9*running_mfu_clean + 0.1*mfu_clean
            mfu_noisy = raw_model_noisy.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu_noisy = mfu_noisy if running_mfu_noisy == -1.0 else 0.9*running_mfu_noisy + 0.1*mfu_noisy
        print(f"[iter {iter_num}] Clean: loss {lossf_clean:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu_clean*100:.2f}%")
        print(f"[iter {iter_num}] Noisy: loss {lossf_noisy:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu_noisy*100:.2f}%")
    
    # 每一步都写入
    if master_process:
        csv_writer.writerow([
            iter_num,
            lossf_clean,
            lossf_noisy,
            lossf_noisy - lossf_clean
        ])
    
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if master_process:
    csv_file.close()
    
if ddp:
    destroy_process_group()
