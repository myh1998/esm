# -*- coding: utf-8 -*-
"""
run_lora_screener.py  (Stage 0–2 集成 + A–D补丁版 + PPL评估优化)

新增/变更要点
A) r=0 或未命中 target 时，自动跳过 S1/S2，仅评估 baseline；避免 "optimizer got an empty parameter list"
B) freeze_base_params 返回是否存在 LoRA 参数；训练前检查，无 LoRA 则不进入训练环节
C) 更频繁的进度打印（--eval_every），便于观察是否"卡住"
D) 可选 --force_cuda0 强制把模型放到 cuda:0，减少 offload（默认仍是 device_map="auto"）
E) PPL评估优化：确保模型在正确状态下评估，避免训练状态干扰
"""

import os, json, time, math, argparse, gc, csv
from dataclasses import dataclass
import torch
from torch import nn
from torch.optim import AdamW
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import random, numpy as np
from pathlib import Path
from pprint import pprint
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError
import math, time, json, os, gc, argparse
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# === LoRA-GA 相关导入 ===
from peft import LoraGAConfig, get_peft_model, TaskType
from peft.utils.lora_ga_utils import estimate_gradient, LoraGAContext

from accelerate import Accelerator

import re

from datasets import load_dataset, load_from_disk
import esm

try:
    from lm_eval import simple_evaluate
    HAS_HARNESS = True
except Exception:
    HAS_HARNESS = False

import contextlib
# -------------------------
# 基础加载
# -------------------------

global_task_type = TaskType.CAUSAL_LM

def clear_cuda(tag=""):
    gc.collect()
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        with contextlib.suppress(Exception):
            torch.cuda.ipc_collect()
    if tag:
        print(f"[mem] cleared cuda cache: {tag}")

def hard_free(model=None, *others, move_model_to_cpu=True, tag=""):
    if move_model_to_cpu and model is not None:
        with contextlib.suppress(Exception):
            model.to("cpu")

    # 注意：这里只能帮助释放传进来的对象引用，
    # 调用方最好也把自己的变量设成 None
    for _ in (model, *others):
        pass

    gc.collect()
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        with contextlib.suppress(Exception):
            torch.cuda.ipc_collect()
        time.sleep(0.1)

    if tag:
        print(f"[mem] hard_free done: {tag}")

def _esm_masked_ce_loss(out, labels):
    logits = out["logits"] if isinstance(out, dict) else out.logits
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

def set_all_seeds(sd):
    random.seed(sd); np.random.seed(sd)
    torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _is_hf_network_error(err):
    msg = str(err)
    return (
        "huggingface.co" in msg
        and ("NameResolutionError" in msg or "MaxRetryError" in msg or "Temporary failure in name resolution" in msg)
    )

# def resolve_model_path(model_id, cache_dir=None):
#     try:
#         local_path = snapshot_download(
#             repo_id=model_id,
#             cache_dir=cache_dir,
#             local_files_only=True,
#         )
#         print(f"[info] Using cached model files at {local_path}")
#         return local_path
#     except LocalEntryNotFoundError:
#         print(f"[info] No local cache for {model_id}. Attempting download from Hugging Face...")
#         try:
#             local_path = snapshot_download(
#                 repo_id=model_id,
#                 cache_dir=cache_dir,
#                 local_files_only=False,
#             )
#             print(f"[info] Downloaded and cached model files at {local_path}")
#             return local_path
#         except Exception as err:
#             if _is_hf_network_error(err):
#                 print(f"[warn] Hugging Face network error while downloading {model_id}: {err}")
#             else:
#                 print(f"[warn] Failed to download {model_id}: {err}")
#             try:
#                 local_path = snapshot_download(
#                     repo_id=model_id,
#                     cache_dir=cache_dir,
#                     local_files_only=True,
#                 )
#                 print(f"[info] Found cached model after download failure at {local_path}")
#                 return local_path
#             except LocalEntryNotFoundError as inner_err:
#                 raise RuntimeError(
#                     f"Model {model_id} not available in cache and download failed."
#                 ) from inner_err

def resolve_model_path(model_id, cache_dir=None):
    """
    Resolve where to load the model from.

    优先级：
      1) model_id 本身就是本地目录（里面有 config.json）
      2) <project_root>/hf_models/<org>/<name> 下面的手动下载模型
      3) TRANSFORMERS_CACHE / HF_HOME 下面的手动下载模型
      4) Hugging Face 缓存 snapshot（local_files_only=True）
      5) 如果允许联网，再尝试下载
    """
    model_id = model_id.strip()

    # 1. model_id 直接是一个本地路径
    direct = Path(model_id).expanduser().resolve()
    if direct.is_dir() and (direct / "config.json").is_file():
        print(f"[info] Using explicit local model path: {direct}")
        return str(direct)

    # 2. 推断 project_root，并构造几个候选 base 目录
    # 当前文件在: <project_root>/lora_with_llm/try_this/...
    project_root = Path(__file__).resolve().parents[2]
    candidates = []

    # 2.1 环境变量给出的 cache_dir
    if cache_dir:
        candidates.append(Path(cache_dir).expanduser())

    # 2.2 项目自己的 hf_models 目录
    candidates.append(project_root / "hf_models")

    # 在这些 base 里查找  <base>/<org>/<name>/config.json
    for base in candidates:
        if not base:
            continue
        cand = base.joinpath(*model_id.split("/"))
        if cand.is_dir() and (cand / "config.json").is_file():
            print(f"[info] Using manually-downloaded local model: {cand}")
            return str(cand)

    # 3. 再不行就委托给 huggingface_hub 的缓存 / 下载逻辑
    try:
        local_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        print(f"[info] Using cached Hugging Face model at: {local_path}")
        return local_path
    except LocalEntryNotFoundError:
        print(f"[info] No local cache for {model_id}. Attempting download from Hugging Face...")
        try:
            local_path = snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                local_files_only=False,
            )
            print(f"[info] Downloaded model files to: {local_path}")
            return local_path
        except Exception as e:
            print(f"[warn] Failed to download {model_id}: {e}")
            raise RuntimeError(
                f"Model {model_id} not available locally and download failed."
            ) from e

def load_llm(model_id, dtype=torch.float16, force_cuda0=False):
    cache_dir = os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME")
    model_path = resolve_model_path(model_id, cache_dir=cache_dir)
    common_kwargs = dict(torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True)
    if force_cuda0:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map={"": "cuda:0"}, local_files_only=True, **common_kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", local_files_only=True, **common_kwargs
        )
    tok = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, trust_remote_code=True, local_files_only=True
    )
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, tok

def select_targets(model, layer_idx, pos_names):
    targets = []
    for pos in pos_names:
        if pos == "self_attn.qv_pair":
            targets += [
                f"model.layers.{layer_idx}.self_attn.q_proj",
                f"model.layers.{layer_idx}.self_attn.v_proj",
            ]
        else:
            targets.append(f"model.layers.{layer_idx}.{pos}")
    named = dict(model.named_modules())
    valid = [t for t in targets if t in named]
    if len(valid) < len(targets):
        missing = sorted(list(set(targets) - set(valid)))
        print(f"[warn] missing modules skipped: {missing}")
    return valid

def build_peft(model, target_modules, r):
    if r == 0 or len(target_modules) == 0:
        return model
    lcfg = LoraConfig(
        r=r, lora_alpha=4*r, lora_dropout=0.05,
        target_modules=target_modules, task_type=global_task_type, bias="none"
    )
    return get_peft_model(model, lcfg)


class NativeLoRALinear(nn.Module):
    """原生 LoRA 线性层：保留 base Linear 前向，并叠加低秩增量。"""
    def __init__(self, base_layer: nn.Linear, r: int, lora_alpha: int = None, lora_dropout: float = 0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"NativeLoRALinear only supports nn.Linear, got {type(base_layer)}")
        if r <= 0:
            raise ValueError(f"r must be > 0, got {r}")

        self.base_layer = base_layer
        self.r = int(r)
        self.lora_alpha = int(lora_alpha if lora_alpha is not None else r)
        self.scaling = self.lora_alpha / max(1, self.r)
        self.lora_dropout = nn.Dropout(float(lora_dropout)) if lora_dropout and lora_dropout > 0 else nn.Identity()

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.lora_A = nn.Parameter(torch.empty(self.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = F.linear(self.lora_dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B) * self.scaling
        return base_out + lora_out


def _module_name_matches(name, target_modules):
    for t in target_modules:
        if name == t or name.endswith(f".{t}"):
            return True
    return False


def _normalize_esm_targets(target_modules):
    alias = {
        "attention.self.query": "q_proj",
        "attention.self.key": "k_proj",
        "attention.self.value": "v_proj",
        "attention.output.dense": "out_proj",
        "intermediate.dense": "fc1",
        "output.dense": "fc2",
    }
    allow = {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}
    out = []
    for t in target_modules:
        key = str(t).strip()
        if not key:
            continue
        key = alias.get(key, key)
        leaf = key.split(".")[-1]
        if leaf in allow:
            out.append(key if "." in key else leaf)
    return out


def build_native_esm_lora(model, target_modules, r, lora_alpha=None, lora_dropout=0.0):
    if r == 0:
        return model, {"injected": 0, "matched": [], "missing": []}

    targets = _normalize_esm_targets(target_modules)
    if len(targets) == 0:
        return model, {"injected": 0, "matched": [], "missing": list(target_modules)}

    named = dict(model.named_modules())
    matched = []
    injected = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not _module_name_matches(name, targets):
            continue
        if isinstance(module, NativeLoRALinear):
            continue
        if "." not in name:
            parent = model
            child = name
        else:
            parent_name, child = name.rsplit(".", 1)
            parent = named.get(parent_name)
        if parent is None or not hasattr(parent, child):
            continue
        setattr(parent, child, NativeLoRALinear(module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout))
        matched.append(name)
        injected += 1

    missing = [t for t in targets if not any((m == t or m.endswith(f".{t}")) for m in matched)]
    if missing:
        print(f"[warn] ESM native LoRA targets missing: {missing}")
    print(f"[info] ESM native LoRA injected linear layers: {injected}")
    return model, {"injected": injected, "matched": matched, "missing": missing}

class TinySeqDataset(Dataset):
    def __init__(self, tok, texts, seq_len=256, max_samples=16):
        ids = tok("\n\n".join(texts), return_tensors="pt")["input_ids"][0]
        chunks = []
        for i in range(0, ids.size(0) - (seq_len + 1), seq_len):
            seg = ids[i:i + seq_len + 1]
            chunks.append(seg)
            if len(chunks) >= max_samples:
                break
        self.chunks = chunks
    def __len__(self): return len(self.chunks)
    def __getitem__(self, i): 
        seg = self.chunks[i]
        return seg[:-1], seg[:-1]  # (input_ids, labels)

def make_ga_dataloader(tok, eval_texts, seq_len=256, batch_size=1, max_samples=16):
    ds = TinySeqDataset(tok, eval_texts, seq_len=seq_len, max_samples=max_samples)
    def collate(batch):
        xs = [b[0] for b in batch]
        import torch.nn.utils.rnn as rnn
        x = rnn.pad_sequence(xs, batch_first=True, padding_value=tok.pad_token_id)
        return {"input_ids": x, "labels": x}
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

# def build_peft_loraga(model, tok, eval_texts, targets, r,
#                       ga_seq_len=256, ga_bs=1, ga_samples=16):
#     if r == 0 or len(targets) == 0:
#         return model

#     # 1) 构建用于 GA 的轻量 dataloader（保持你原来的实现）
#     dl = make_ga_dataloader(tok, eval_texts, seq_len=ga_seq_len,
#                             batch_size=ga_bs, max_samples=ga_samples)

#     # 2) 用 Accelerator 统一设备（关键！）
#     accelerator = Accelerator(mixed_precision="fp16")   # 或 "bf16"/"no"
#     model, dl = accelerator.prepare(model, dl)

#     # 3) 估计梯度（把 accelerator 传进去）
#     named_grad = estimate_gradient(
#         model=model,
#         dataloader=dl,
#         accelerator=accelerator,
#         quant_flag=False
#     )

#     # 4) GA 配置 + 构建 PEFT 模型
#     peft_cfg = LoraGAConfig(
#         r=r, lora_alpha=r, lora_dropout=0.0,
#         target_modules=targets, bias="none", task_type=global_task_type,
#         init_lora_weights="lora_ga"
#     )
#     from peft.utils.lora_ga_utils import LoraGAContext
#     with LoraGAContext(model=model, named_grad=named_grad):
#         model = get_peft_model(model=model, peft_config=peft_cfg)

#     return model

# def build_peft_loraga(model, tok, eval_texts, targets, r,
#                       ga_seq_len=256, ga_bs=1, ga_samples=16):
#     """
#     安全版本：在注入前检查维度是否允许该rank的LoRA注入；
#     若出现异常（例如 matmul shape 错误），自动跳过。
#     返回 (model, status, error_msg)
#     """
#     if r == 0 or len(targets) == 0:
#         return model, False, "Empty rank or targets"

#     try:
#         # [1] 初始化数据加载器与accelerator
#         dl = make_ga_dataloader(tok, eval_texts, seq_len=ga_seq_len,
#                                 batch_size=ga_bs, max_samples=ga_samples)
#         accelerator = Accelerator(mixed_precision="fp16")
#         model, dl = accelerator.prepare(model, dl)

#         # [2] 梯度估计
#         named_grad = estimate_gradient(
#             model=model,
#             dataloader=dl,
#             accelerator=accelerator,
#             quant_flag=False
#         )

#         # [3] 检查每个目标模块是否支持该rank
#         for target in targets:
#             module = model
#             for attr in target.split("."):
#                 module = getattr(module, attr, None)
#                 if module is None:
#                     return model, False, f"[INVALID] Cannot resolve target: {target}"
#             if hasattr(module, 'weight'):
#                 out_dim, in_dim = module.weight.shape
#                 if r > in_dim or r > out_dim:
#                     return model, False, f"[SKIP] Rank={r} > in_dim={in_dim}, out_dim={out_dim}"

#         # [4] 执行LoRA-GA注入
#         peft_cfg = LoraGAConfig(
#             r=r, lora_alpha=r, lora_dropout=0.0,
#             target_modules=targets,
#             bias="none", task_type=global_task_type,
#             init_lora_weights="lora_ga"
#         )
#         with LoraGAContext(model=model, named_grad=named_grad):
#             model = get_peft_model(model=model, peft_config=peft_cfg)

#         return model, True, "Success"

#     except Exception as e:
#         return model, False, str(e)

def build_peft_loraga(model, tok, eval_texts, targets, r,
                      ga_seq_len=256, ga_bs=1, ga_samples=16):
    if r == 0 or len(targets) == 0:
        return model, False, "Empty rank or targets"

    try:
        # 1) 先构建 GA 数据
        dl = make_ga_dataloader(tok, eval_texts, seq_len=ga_seq_len,
                                batch_size=ga_bs, max_samples=ga_samples)

        # 2) 用 Accelerate 仅做“估梯度”，随后拿回“真实模型”
        accelerator = Accelerator(mixed_precision="fp16")
        acc_model, dl = accelerator.prepare(model, dl)
        base = accelerator.unwrap_model(acc_model)   # ← 关键：拿到未包装的真实模型

        # 3) 在 base 上做目标解析与合法性检查
        for target in targets:
            m = base
            for attr in target.split("."):
                m = getattr(m, attr, None)
                if m is None:
                    return model, False, f"[INVALID] Cannot resolve target: {target}"
            if hasattr(m, 'weight'):
                out_dim, in_dim = m.weight.shape
                if r > in_dim or r > out_dim:
                    return model, False, f"[SKIP] Rank={r} > in_dim={in_dim}, out_dim={out_dim}"

        # 4) 估计梯度（对 acc_model 跑），但把 named_grad 用来给 base 注入
        named_grad = estimate_gradient(model=acc_model, dataloader=dl,
                                       accelerator=accelerator, quant_flag=False)

        # 5) 在 base（未包装的真实模型）上注入 LoRA-GA
        peft_cfg = LoraGAConfig(
            r=r, lora_alpha=r, lora_dropout=0.0,
            target_modules=targets, bias="none", task_type=global_task_type,
            init_lora_weights="lora_ga"
        )
        from peft.utils.lora_ga_utils import LoraGAContext
        with LoraGAContext(model=base, named_grad=named_grad):
            base = get_peft_model(model=base, peft_config=peft_cfg)

        # 6) 返回“带LoRA的 base”作为后续训练对象（单卡场景可直接替换）
        return base, True, "Success"

    except Exception as e:
        return model, False, str(e)

def count_lora_params(model):
    return sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and "lora_" in n)

def freeze_base_params(model):
    """返回：是否存在 LoRA 参数"""
    any_lora = False
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
            any_lora = True
        else:
            p.requires_grad_(False)
    return any_lora

# -------------------------
# 数据 & 度量
# -------------------------
def load_eval_texts(name="wikitext", config="wikitext-2-raw-v1", split="test", n=64):
    ds = load_dataset(name, config)[split]
    texts = [x["text"] for x in ds.select(range(max(n,1))) if len(x["text"].strip())>0]
    return texts

# @torch.no_grad()
# def eval_ppl(model, tok, eval_texts, max_length=1024, max_eval_tokens=4096):
#     """评估PPL，确保模型在正确状态下运行"""
#     # 保存原始状态
#     original_training = model.training
#     original_use_cache = getattr(model.config, 'use_cache', True)
#     original_gradient_checkpointing = getattr(model.config, 'gradient_checkpointing', False)
    
#     # 设置为评估最优状态
#     model.eval()
#     if hasattr(model.config, "use_cache"):
#         model.config.use_cache = True  # 评估时启用缓存提高速度
#     if hasattr(model.config, "gradient_checkpointing"):
#         model.config.gradient_checkpointing = False  # 评估时禁用
    
#     device = next(model.parameters()).device
    
#     try:
#         # 清理GPU缓存
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
            
#         enc = tok("\n\n".join(eval_texts), return_tensors="pt")
#         input_ids = enc["input_ids"][0][:max_eval_tokens].unsqueeze(0).to(device)
#         stride = min(1024, max_length)
#         nlls, cnt = [], 0
        
#         for i in range(0, input_ids.shape[1] - 1, stride):
#             beg = i
#             end = min(i + max_length, input_ids.shape[1])
#             src = input_ids[:, beg:end-1]
#             outputs = model(src, labels=src)
#             nlls.append(outputs.loss.detach().float().cpu())
#             cnt += 1
            
#         if cnt == 0:
#             return float("nan")
#         mean_nll = torch.stack(nlls).mean().item()
#         return math.exp(mean_nll)
        
#     finally:
#         # 恢复原始状态
#         if original_training:
#             model.train()
#         if hasattr(model.config, "use_cache"):
#             model.config.use_cache = original_use_cache
#         if hasattr(model.config, "gradient_checkpointing"):
#             model.config.gradient_checkpointing = original_gradient_checkpointing

@torch.no_grad()
def eval_ppl(model, tok, eval_texts, max_length=1024, max_eval_tokens=4096, stride=None):
    """基于滑窗的PPL评测：按token加权平均，默认无重叠；可配置半窗重叠提升稳定性。"""
    # 记录/切换状态
    was_training = model.training
    use_cache_orig = getattr(model.config, 'use_cache', None)
    grad_ckpt_orig = getattr(model.config, 'gradient_checkpointing', None)

    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    if hasattr(model.config, "gradient_checkpointing"):
        model.config.gradient_checkpointing = False

    device = next(model.parameters()).device
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        enc = tok("\n\n".join(eval_texts), return_tensors="pt")
        input_ids = enc["input_ids"][0][:max_eval_tokens].unsqueeze(0).to(device)

        if stride is None or stride <= 0:
            stride = max_length  # 默认无重叠

        nll_sum, tok_sum = 0.0, 0
        L = input_ids.shape[1]

        # 滑窗：每个chunk用 labels=chunk，让模型内部完成右移对齐
        for i in range(0, L - 1, stride):
            end = min(i + max_length, L)
            chunk = input_ids[:, i:end]
            out = model(chunk, labels=chunk)
            trg_len = chunk.size(1) - 1  # 有效预测位
            if trg_len <= 0:
                continue
            nll_sum += out.loss.item() * trg_len
            tok_sum += trg_len

        if tok_sum == 0:
            return float("nan")
        ppl = math.exp(nll_sum / tok_sum)
        return ppl

    finally:
        # 恢复状态
        if was_training:
            model.train()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = use_cache_orig
        if hasattr(model.config, "gradient_checkpointing"):
            model.config.gradient_checkpointing = grad_ckpt_orig


@torch.no_grad()
def measure_step_latency(model, tok, seq_len=256, steps=20):
    """测量延迟，确保在正确状态下运行"""
    # 保存原始状态
    original_training = model.training
    original_use_cache = getattr(model.config, 'use_cache', True)
    original_gradient_checkpointing = getattr(model.config, 'gradient_checkpointing', False)
    
    # 设置为评估状态
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True  # 评估时启用缓存提高速度
    if hasattr(model.config, "gradient_checkpointing"):
        model.config.gradient_checkpointing = False  # 评估时禁用
        
    device = next(model.parameters()).device
    dummy = tok("Hello " * 100, return_tensors="pt")
    ids = dummy["input_ids"][:, :seq_len].to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        # 预热
        for _ in range(3):
            _ = model(ids)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(steps):
            _ = model(ids)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        latency = (t1 - t0) / steps

        max_mem = None
        if device.type == "cuda":
            max_mem = torch.cuda.max_memory_allocated(device)/1024**2
        return latency, max_mem
        
    finally:
        # 恢复原始状态
        if original_training:
            model.train()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = original_use_cache
        if hasattr(model.config, "gradient_checkpointing"):
            model.config.gradient_checkpointing = original_gradient_checkpointing

# -------------------------
# Stage 0: 梯度 proxy（快速）
# -------------------------
def grad_proxy_score(model, tok, targets, r, seq_len=256):
    if r == 0 or len(targets)==0:
        return 0.0
    model.train()
    device = next(model.parameters()).device
    text = "The quick brown fox jumps over the lazy dog. " * 64
    enc = tok(text, return_tensors="pt")
    ids = enc["input_ids"][:, :seq_len].to(device)

    model.zero_grad(set_to_none=True)
    out = model(ids, labels=ids)
    loss = out.loss
    loss.backward()

    named_params = dict(model.named_parameters())
    score = 0.0; denom = 0.0
    for t in targets:
        wname = f"{t}.weight"
        grad_f = 0.0; m_eff = 1.0
        if wname in named_params and named_params[wname].grad is not None:
            G = named_params[wname].grad.detach()
            m, n = G.shape
            m_eff = float(min(m, n))
            grad_f = torch.norm(G, p='fro').item()
        else:
            # 回退：聚合该模块下所有叶子参数梯度的范数
            for n, p in named_params.items():
                if n.startswith(t) and p.grad is not None:
                    g = p.grad.detach()
                    grad_f += torch.norm(g, p=2).item()**2
            grad_f = math.sqrt(grad_f) if grad_f>0 else 0.0
        cap = max(1.0, m_eff)
        score += (r / cap) * grad_f
        denom += 1.0
    model.zero_grad(set_to_none=True)
    return float(score / max(1.0, denom))

def _extract_primary_metrics(results_dict):
    """
    从 lm-eval 返回的 {task: {metric: value}} 中抽取主指标，优先 acc/acc_norm/EM/F1/（word_）perplexity
    返回形如：{"wikitext.perplexity": 7.93, "hellaswag.acc_norm": 0.78, ...}
    """
    out = {}
    prefer = ("acc", "acc_norm", "exact_match", "f1", "word_perplexity", "perplexity", "ppl")
    res = (results_dict or {})
    for task, metrics in res.items():
        got = None
        for name in prefer:
            if name in metrics:
                got = (name, metrics[name]); break
            # 兼容 "word_perplexity,none" 这类键
            for k, v in metrics.items():
                if k.split(",", 1)[0] == name:
                    got = (name, v); break
            if got: break
        if got:
            out[f"{task}.{got[0]}"] = float(got[1])
    return out


def _harness_simple_eval(model_args_str, tasks, limit, batch_size, bootstrap, device, tag, out_dir, mdir=None):
    
    if mdir is None:
        final_model_args = model_args_str
    else:
        use_4bit = (not args.harness_no_4bit) and args.harness_device and not str(args.harness_device).lower().startswith("cpu")
        if use_4bit:
            final_model_args = (
                f"pretrained={mdir},trust_remote_code=True,attn_implementation=sdpa,"
                f"load_in_4bit=True,bnb_4bit_quant_type=nf4,bnb_4bit_compute_dtype=float16,"
                f"bnb_4bit_use_double_quant=True,device_map=auto"
            )
        else:
            final_model_args = (
                f"pretrained={mdir},trust_remote_code=True,attn_implementation=sdpa,device_map=auto"
            )
    
    eval_kwargs = dict(
        model="hf",
        model_args=final_model_args,
        tasks=[t.strip() for t in tasks.split(",") if t.strip()],
        batch_size=batch_size,
        limit=(None if str(limit).lower() == "none" else limit),
        bootstrap_iters=bootstrap,
        log_samples=False,
    )

    # 只在没有使用 device_map=auto 或你明确指定时，才传 device
    if device and ("device_map=auto" not in final_model_args):
        eval_kwargs["device"] = device

    print("[harness] launch:", eval_kwargs)
    results = simple_evaluate(**eval_kwargs)

    primary = _extract_primary_metrics(results.get("results", {}))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"harness_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"primary": primary, "raw": results}, f, indent=2, ensure_ascii=False, default=str)
    print(f"[harness] write -> {out_path}")
    return primary, out_path


@torch.no_grad()
def run_harness_eval_from_model(model, tok, base_model_id, args, tag):
    """
    对“当前（可能含 LoRA）模型”做 harness：
    - 若无 LoRA，可直接指向 base_model_id（速度更快）；
    - 若含 LoRA，则导出 Adapter -> 在 CPU 上合并权重 -> 保存临时“已合并”HF 模型 -> harness 评测。
    """
    if not HAS_HARNESS:
        print("[harness] lm-evaluation-harness 未安装，跳过。pip install lm-evaluation-harness")
        return {}, None

    # 无 LoRA（或 r=0）：直接用仓库/本地路径评测
    # if count_lora_params(model) == 0:
    #     model_args_str = f"pretrained={base_model_id},trust_remote_code=True,attn_implementation=sdpa"
    #     return _harness_simple_eval(
    #         model_args_str, args.harness_tasks, args.harness_limit,
    #         args.harness_batch_size, args.harness_bootstrap,
    #         args.harness_device, tag, args.harness_out_dir
    #     )
    if count_lora_params(model) == 0:
        if args.harness_device and str(args.harness_device).lower().startswith("cpu"):
            model_args_str = f"pretrained={base_model_id},trust_remote_code=True,attn_implementation=sdpa"
        else:
            model_args_str = (
                f"pretrained={base_model_id},trust_remote_code=True,attn_implementation=sdpa,"
                f"load_in_4bit=True,bnb_4bit_quant_type=nf4,bnb_4bit_compute_dtype=float16,"
                f"bnb_4bit_use_double_quant=True,device_map=auto"
            )
        return _harness_simple_eval(
            model_args_str, args.harness_tasks, args.harness_limit,
            args.harness_batch_size, args.harness_bootstrap,
            args.harness_device, tag, args.harness_out_dir
        )

    # 有 LoRA：导出 adapter -> CPU 合并 -> 保存临时 HF 模型目录
    import tempfile
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    with tempfile.TemporaryDirectory() as tmp:
        adir = os.path.join(tmp, "adapter")
        mdir = os.path.join(tmp, "merged")
        os.makedirs(mdir, exist_ok=True)

        # 1) 导出当前 LoRA（仅 Adapter 权重即可）
        model.save_pretrained(adir)

        # 2) CPU 上重建基座 + 加载 Adapter + 合并
        base_cpu = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=torch.float16, device_map="cpu",
            low_cpu_mem_usage=True, trust_remote_code=True
        )
        peft_loaded = PeftModel.from_pretrained(base_cpu, adir)
        merged = peft_loaded.merge_and_unload()  # 变成纯 HF 模型（无 PEFT 依赖）
        merged.save_pretrained(mdir)
        tok.save_pretrained(mdir)                # 同步保存分词器

        # 3) 以“已合并模型目录”调用 harness
        model_args_str = f"pretrained={mdir},trust_remote_code=True,attn_implementation=sdpa"
        return _harness_simple_eval(
            model_args_str, args.harness_tasks, args.harness_limit,
            args.harness_batch_size, args.harness_bootstrap,
            args.harness_device, tag, args.harness_out_dir, mdir
        )


# -------------------------
# Stage 1/2: 快速微调（仅 LoRA）
# -------------------------
@dataclass
class FTConfig:
    steps: int = 200
    lr: float = 1e-4
    warmup_ratio: float = 0.06
    seq_len: int = 512
    batch_size: int = 1
    grad_accum: int = 1
    eval_every: int = 10
    early_stop_patience: int = 20
    early_stop_delta: float = 1e-4
    clip_grad_norm: float = 1.0

def text_stream(tok, eval_texts, seq_len, rng=None, mode="deterministic"):
    buf = tok("\n\n".join(eval_texts), return_tensors="pt")["input_ids"][0]
    N = buf.shape[0]
    cur = 0
    while True:
        if mode == "deterministic":
            if cur + seq_len + 1 >= N: cur = 0
            ids = buf[cur:cur+seq_len+1]; cur += seq_len
        else:  # stochastic
            if rng is None:
                rng = np.random.RandomState(0)
            start = 0 if N <= seq_len+1 else rng.randint(0, N-(seq_len+1))
            ids = buf[start:start+seq_len+1]
        yield ids

# def train_lora(model, tok, train_texts, eval_texts, cfg: FTConfig, seed_mode="deterministic"):
def train_lora(model, tok, train_texts, eval_texts, cfg: FTConfig, args, seed_mode="deterministic"):

    from transformers import get_linear_schedule_with_warmup
    device = next(model.parameters()).device
    model.train()
    has_lora = freeze_base_params(model)
    if not has_lora:
        print("[info] no LoRA parameters found; skip training")
        return {"train_loss_traj": [], "steps_done": 0, "eval_ppl_per_step": []}

    lora_params = [p for n,p in model.named_parameters() if p.requires_grad]
    if len(lora_params) == 0:
        print("[info] empty trainable set; skip training")
        return {"train_loss_traj": [], "steps_done": 0, "eval_ppl_per_step": []}

    opt = AdamW(lora_params, lr=cfg.lr)
    sch = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=int(cfg.warmup_ratio * cfg.steps),
        num_training_steps=cfg.steps
    )

    rng = np.random.RandomState(torch.initial_seed() % (2**32-1))
    feeder = text_stream(tok, train_texts, cfg.seq_len, rng=rng, mode=seed_mode)
    best_loss = float('inf'); no_improve = 0
    hist = []
    hist_ppl_per_step = []

    for step in range(1, cfg.steps+1):
        total_loss = 0.0
        ppl = 100
        for _ in range(cfg.grad_accum):
            batch = [next(feeder) for __ in range(cfg.batch_size)]
            batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=tok.pad_token_id)
            batch = batch[:, :cfg.seq_len+1]
            inp = batch[:, :-1].to(device)
            lab = batch[:, :-1].to(device)

            out = model(inp, labels=lab)
            loss = out.loss / cfg.grad_accum
            loss.backward()
            total_loss += loss.item()

        if cfg.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(lora_params, cfg.clip_grad_norm)

        opt.step(); sch.step(); model.zero_grad(set_to_none=True)

        train_loss = total_loss
        if best_loss - train_loss > cfg.early_stop_delta:
            best_loss = train_loss; no_improve = 0
        else:
            no_improve += 1

        # if step % max(1, cfg.eval_every) == 0:

        #     ppl = eval_ppl(
        #         model, tok, eval_texts,
        #         max_length=args.ppl_max_len,
        #         max_eval_tokens=args.ppl_max_tokens,
        #         stride=args.ppl_stride if args.ppl_stride is not None else args.ppl_max_len
        #     )
        if step % max(1, cfg.eval_every) == 0:
            ppl = eval_ppl(model, tok, eval_texts,
                max_length=args.ppl_max_len,
                max_eval_tokens=args.ppl_max_tokens,
                stride=args.ppl_stride if args.ppl_stride is not None else args.ppl_max_len)
            
            hist_ppl_per_step.append(float(ppl))

            print(f"[FT] step {step}/{cfg.steps} | train_loss={train_loss:.4f} | best={best_loss:.4f} | no_improve={no_improve} | eval_ppl={ppl}")
        hist.append(float(train_loss))

        if no_improve >= cfg.early_stop_patience:
            print(f"[early-stop] no improvement {no_improve} steps, stop at {step}")
            break

    return {"train_loss_traj": hist, "steps_done": len(hist), "eval_ppl_per_step": hist_ppl_per_step}, min(hist_ppl_per_step)

def load_train_texts(name="wikitext", config="wikitext-2-raw-v1", split="train", n=1000):
    """加载训练数据，与评估数据分离"""
    ds = load_dataset(name, config)[split]
    texts = [x["text"] for x in ds.select(range(max(n,1))) if len(x["text"].strip())>0]
    return texts


def parse_target_modules(target_modules_arg):
    if isinstance(target_modules_arg, (list, tuple)):
        return [str(x).strip() for x in target_modules_arg if str(x).strip()]
    if target_modules_arg is None:
        return []
    if isinstance(target_modules_arg, str):
        try:
            parsed = json.loads(target_modules_arg)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
        return [x.strip() for x in target_modules_arg.split(",") if x.strip()]
    return []


def _safe_seq_id(record, idx):
    rid = record.get("id")
    if rid is None or str(rid).strip() == "":
        rid = f"sample_{idx:06d}"
    return str(rid)


def _coords_to_contact_map(coords, threshold=8.0):
    arr = np.asarray(coords)
    if arr.ndim == 2 and arr.shape[1] == 3:
        cb = arr
    elif arr.ndim == 3 and arr.shape[2] == 3:
        if arr.shape[1] > 4:
            cb = arr[:, 4, :]  # 常见原子顺序下 Cβ 在索引 4
        else:
            cb = arr[:, 1, :]  # 回退：取 Cα
    else:
        raise ValueError(f"Unsupported coords shape: {arr.shape}")
    d = cb[:, None, :] - cb[None, :, :]
    dist = np.linalg.norm(d, axis=-1)
    return (dist < float(threshold)).astype(np.uint8)


def _normalize_record(record, idx, source):
    seq = record.get("sequence", record.get("seq"))
    if isinstance(seq, bytes):
        seq = seq.decode("utf-8")
    if seq is None:
        raise ValueError("missing sequence")
    seq = str(seq).strip()
    if len(seq) == 0:
        raise ValueError("empty sequence")

    contact = record.get("contact_map", record.get("contacts"))
    if contact is None and "coords" in record:
        contact = _coords_to_contact_map(record["coords"])
    if contact is None and "atom_positions" in record:
        contact = _coords_to_contact_map(record["atom_positions"])
    if contact is None:
        raise ValueError("missing contact_map/coords")

    cm = np.asarray(contact)
    if cm.ndim != 2:
        raise ValueError(f"contact_map must be 2D, got {cm.shape}")
    L = len(seq)
    if cm.shape[0] != L or cm.shape[1] != L:
        raise ValueError(f"shape mismatch seq={L}, contact={cm.shape}")

    out = {
        "id": _safe_seq_id(record, idx),
        "sequence": seq,
        "contact_map": (cm > 0).astype(np.uint8),
        "length": L,
        "source": source,
    }
    return out


def load_structure_records(data_path, data_format="auto", max_records=None):
    data_path = Path(data_path).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"DATA_PATH not found: {data_path}")

    records = []
    dropped = []

    def _accept(rec, idx, source):
        try:
            norm = _normalize_record(rec, idx, source)
            records.append(norm)
        except Exception as e:
            dropped.append({"index": int(idx), "source": source, "reason": str(e)})

    if data_format in ("auto", "npz"):
        files = [data_path] if data_path.suffix == ".npz" else sorted(data_path.glob("*.npz"))
        for fp in files:
            arr = np.load(fp, allow_pickle=True)
            keys = list(arr.keys())
            if "records" in arr:
                items = arr["records"]
                for i, item in enumerate(items):
                    if max_records and len(records) >= max_records:
                        break
                    rec = dict(item)
                    rec.setdefault("id", f"{fp.stem}_{i}")
                    _accept(rec, i, str(fp))
            elif {"sequence", "contact_map"}.issubset(keys):
                seqs = arr["sequence"]
                cms = arr["contact_map"]
                ids = arr["id"] if "id" in keys else None
                if np.asarray(seqs).ndim == 0 and np.asarray(cms).ndim == 2:
                    rid = ids.item() if (ids is not None and np.asarray(ids).ndim == 0) else fp.stem
                    _accept({"id": rid, "sequence": seqs.item(), "contact_map": cms}, 0, str(fp))
                else:
                    for i in range(min(len(seqs), len(cms))):
                        if max_records and len(records) >= max_records:
                            break
                        rid = (
                            ids[i]
                            if (ids is not None and np.asarray(ids).ndim > 0 and len(ids) > i)
                            else f"{fp.stem}_{i}"
                        )
                        _accept({"id": rid, "sequence": seqs[i], "contact_map": cms[i]}, i, str(fp))
            elif {"seq", "contact_map"}.issubset(keys):
                seqs = arr["seq"]
                cms = arr["contact_map"]
                ids = arr["id"] if "id" in keys else None
                if np.asarray(seqs).ndim == 0 and np.asarray(cms).ndim == 2:
                    rid = ids.item() if (ids is not None and np.asarray(ids).ndim == 0) else fp.stem
                    _accept({"id": rid, "seq": seqs.item(), "contact_map": cms}, 0, str(fp))
                else:
                    for i in range(min(len(seqs), len(cms))):
                        if max_records and len(records) >= max_records:
                            break
                        rid = (
                            ids[i]
                            if (ids is not None and np.asarray(ids).ndim > 0 and len(ids) > i)
                            else f"{fp.stem}_{i}"
                        )
                        _accept({"id": rid, "seq": seqs[i], "contact_map": cms[i]}, i, str(fp))
            elif {"sequence", "coords"}.issubset(keys):
                seqs = arr["sequence"]
                cds = arr["coords"]
                ids = arr["id"] if "id" in keys else None
                if np.asarray(seqs).ndim == 0 and np.asarray(cds).ndim == 2:
                    rid = ids.item() if (ids is not None and np.asarray(ids).ndim == 0) else fp.stem
                    _accept({"id": rid, "sequence": seqs.item(), "coords": cds}, 0, str(fp))
                else:
                    for i in range(min(len(seqs), len(cds))):
                        if max_records and len(records) >= max_records:
                            break
                        rid = (
                            ids[i]
                            if (ids is not None and np.asarray(ids).ndim > 0 and len(ids) > i)
                            else f"{fp.stem}_{i}"
                        )
                        _accept({"id": rid, "sequence": seqs[i], "coords": cds[i]}, i, str(fp))
            elif "sequence" in keys and "atom_positions" in keys:
                seqs = arr["sequence"]
                aps = arr["atom_positions"]
                ids = arr["id"] if "id" in keys else None
                if np.asarray(seqs).ndim == 0 and np.asarray(aps).ndim >= 2:
                    rid = ids.item() if (ids is not None and np.asarray(ids).ndim == 0) else fp.stem
                    _accept({"id": rid, "sequence": seqs.item(), "atom_positions": aps}, 0, str(fp))
                else:
                    for i in range(min(len(seqs), len(aps))):
                        if max_records and len(records) >= max_records:
                            break
                        rid = (
                            ids[i]
                            if (ids is not None and np.asarray(ids).ndim > 0 and len(ids) > i)
                            else f"{fp.stem}_{i}"
                        )
                        _accept({"id": rid, "sequence": seqs[i], "atom_positions": aps[i]}, i, str(fp))
            else:
                dropped.append({"index": -1, "source": str(fp), "reason": f"unsupported keys: {keys}"})

    if (data_format in ("auto", "sidechainnet")) and len(records) == 0:
        try:
            import sidechainnet as scn

            dset = scn.load(casp_version=12, thinning=100)
            for split_name in ("train", "valid", "test"):
                split = dset.get(split_name, {})
                seqs = split.get("seq", [])
                coords = split.get("crd", [])
                ids = split.get("ids", [])
                for i in range(min(len(seqs), len(coords))):
                    if max_records and len(records) >= max_records:
                        break
                    _accept(
                        {
                            "id": ids[i] if i < len(ids) else f"{split_name}_{i}",
                            "seq": seqs[i],
                            "coords": np.asarray(coords[i]).reshape(-1, 14, 3)[:, 4, :],
                        },
                        i,
                        f"sidechainnet:{split_name}",
                    )
        except Exception as e:
            dropped.append({"index": -1, "source": "sidechainnet", "reason": str(e)})

    # 兜底：若仍有重复 ID，自动去重重命名，防止 split/fixed_subset 失真
    seen = {}
    dup_cnt = 0
    for rec in records:
        rid = rec["id"]
        if rid not in seen:
            seen[rid] = 0
            continue
        seen[rid] += 1
        dup_cnt += 1
        rec["id"] = f"{rid}__dup{seen[rid]:04d}"
    if dup_cnt > 0:
        print(f"[warn] deduplicated {dup_cnt} repeated record ids")

    return records, dropped


def write_dropped_records(dropped, out_csv):
    if not dropped:
        return
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "source", "reason"])
        writer.writeheader()
        for row in dropped:
            writer.writerow(row)


def _length_bin(L):
    if L < 256:
        return "<256"
    if L <= 512:
        return "256-512"
    return ">512"


def _stratified_sample_indices(records, n_samples, seed=42):
    if n_samples > len(records):
        raise ValueError(f"Requested {n_samples} > available {len(records)}")
    by_bin = {"<256": [], "256-512": [], ">512": []}
    for i, rec in enumerate(records):
        by_bin[_length_bin(rec["length"])].append(i)
    rng = np.random.RandomState(seed)
    target = {}
    for k in by_bin.keys():
        prop = len(by_bin[k]) / max(1, len(records))
        target[k] = int(round(prop * n_samples))
    # 修正四舍五入误差
    while sum(target.values()) < n_samples:
        k = max(by_bin.keys(), key=lambda x: len(by_bin[x]) - target[x])
        target[k] += 1
    while sum(target.values()) > n_samples:
        k = max(target.keys(), key=lambda x: target[x])
        if target[k] > 0:
            target[k] -= 1
    picked = []
    for k, ids in by_bin.items():
        take = min(target[k], len(ids))
        if take > 0:
            choice = rng.choice(ids, size=take, replace=False).tolist()
            picked.extend(choice)
    if len(picked) < n_samples:
        remaining = sorted(list(set(range(len(records))) - set(picked)))
        extra = rng.choice(remaining, size=(n_samples - len(picked)), replace=False).tolist()
        picked.extend(extra)
    return sorted(picked)


def _split_structure_records(records, split_manifest_path, train_n=12000, test_n=2300, seed=42):
    split_manifest_path = Path(split_manifest_path)
    if split_manifest_path.exists():
        with split_manifest_path.open("r", encoding="utf-8") as f:
            man = json.load(f)
        rid_to_idx = {r["id"]: i for i, r in enumerate(records)}
        train_idx = [rid_to_idx[x] for x in man["train_ids"] if x in rid_to_idx]
        test_idx = [rid_to_idx[x] for x in man["test_ids"] if x in rid_to_idx]
        val_idx = [rid_to_idx[x] for x in man["val_ids"] if x in rid_to_idx]
        ok = (
            len(train_idx) == train_n
            and len(test_idx) == test_n
            and len(set(train_idx).intersection(test_idx)) == 0
            and len(set(train_idx).intersection(val_idx)) == 0
            and len(set(test_idx).intersection(val_idx)) == 0
        )
        if ok:
            return train_idx, test_idx, val_idx
        print(
            "[warn] Existing split_manifest is incompatible with current records "
            f"(train={len(train_idx)}/{train_n}, test={len(test_idx)}/{test_n}). Regenerating..."
        )

    if len(records) < (train_n + test_n):
        raise RuntimeError(f"Not enough valid records ({len(records)}) for train={train_n}, test={test_n}")

    by_bin = {"<256": [], "256-512": [], ">512": []}
    for i, rec in enumerate(records):
        by_bin[_length_bin(rec["length"])].append(i)
    rng = np.random.RandomState(seed)
    for k in by_bin:
        rng.shuffle(by_bin[k])

    train_idx, test_idx, val_idx = [], [], []
    for k, ids in by_bin.items():
        total = len(ids)
        t = int(round(total * (train_n / len(records))))
        te = int(round(total * (test_n / len(records))))
        train_idx.extend(ids[:t])
        test_idx.extend(ids[t:t + te])
        val_idx.extend(ids[t + te:])

    # 调整到精确条数
    def _rebalance(target_n, bucket, pool):
        while len(bucket) < target_n and pool:
            bucket.append(pool.pop())
        while len(bucket) > target_n:
            pool.append(bucket.pop())

    _rebalance(train_n, train_idx, val_idx)
    _rebalance(test_n, test_idx, val_idx)

    train_idx = sorted(set(train_idx))
    test_idx = sorted(set(test_idx) - set(train_idx))
    val_idx = sorted(set(val_idx) - set(train_idx) - set(test_idx))

    # 兜底：若因去重产生缺口，再从 val 补齐
    rng.shuffle(val_idx)
    while len(train_idx) < train_n and val_idx:
        train_idx.append(val_idx.pop())
    while len(test_idx) < test_n and val_idx:
        test_idx.append(val_idx.pop())

    train_idx = sorted(train_idx[:train_n])
    test_idx = sorted(test_idx[:test_n])
    val_idx = sorted(set(range(len(records))) - set(train_idx) - set(test_idx))

    split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "train_ids": [records[i]["id"] for i in train_idx],
        "test_ids": [records[i]["id"] for i in test_idx],
        "val_ids": [records[i]["id"] for i in val_idx],
    }
    with split_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return train_idx, test_idx, val_idx


def _make_masked_batch(tokens, alphabet, mask_prob=0.15):
    labels = tokens.clone()
    special = torch.zeros_like(tokens, dtype=torch.bool)
    for idx in [alphabet.padding_idx, alphabet.cls_idx, alphabet.eos_idx]:
        special |= tokens.eq(idx)
    can_mask = ~special
    rand = torch.rand_like(tokens.float())
    mask_pos = (rand < mask_prob) & can_mask
    labels[~mask_pos] = -100
    inputs = tokens.clone()
    inputs[mask_pos] = alphabet.mask_idx
    return inputs, labels


@torch.no_grad()
def eval_esm_pseudo_ppl(model, records, alphabet, batch_converter, batch_size=1, mask_prob=0.15, max_eval=512, max_len=768):
    
    stats = {
    "total_seen": 0,
    "evaluated": 0,
    "skipped_too_long": 0,
    "skipped_oom": 0,
    "max_len_threshold": max_len,
    }

    if len(records) == 0:
        return float("nan"), stats

    model.eval()
    device = next(model.parameters()).device
    loss_sum, steps = 0.0, 0
    use = records[:min(max_eval, len(records))]

    for i in range(0, len(use), batch_size):
        chunk = use[i:i + batch_size]
        stats["total_seen"] += len(chunk)

        kept = [r for r in chunk if r["length"] <= max_len]
        stats["skipped_too_long"] += (len(chunk) - len(kept))
        if len(kept) == 0:
            continue

        batch = [(r["id"], r["sequence"]) for r in kept]
        _, _, toks = batch_converter(batch)
        toks = toks.to(device)
        inp, labels = _make_masked_batch(toks, alphabet, mask_prob=mask_prob)

        try:
            out = model(inp)
            loss = _esm_masked_ce_loss(out, labels)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                stats["skipped_oom"] += len(kept)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise

        loss_sum += float(loss.item())
        steps += 1
        stats["evaluated"] += len(kept)

        del toks, inp, labels, out, batch, kept, chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if steps == 0:
        return float("nan"), stats
    ppl = float(math.exp(loss_sum / steps))
    return ppl, stats


@torch.no_grad()
def eval_long_range_pl(model, records, alphabet, batch_converter, batch_size=1, long_range_sep=24, max_eval=None, max_len=768):

    stats = {
        "total_seen": 0,
        "evaluated": 0,
        "skipped_too_long": 0,
        "skipped_oom": 0,
        "max_len_threshold": max_len,
    }

    if len(records) == 0:
        return float("nan"), [], stats
    model.eval()
    device = next(model.parameters()).device
    use = records if max_eval is None else records[:min(max_eval, len(records))]
    per_item = []
    for i in range(0, len(use), batch_size):
        chunk = use[i:i + batch_size]
        stats["total_seen"] += len(chunk)

        kept = [r for r in chunk if r["length"] <= max_len]
        stats["skipped_too_long"] += (len(chunk) - len(kept))
        if len(kept) == 0:
            continue

        batch = [(r["id"], r["sequence"]) for r in kept]
        _, _, toks = batch_converter(batch)
        toks = toks.to(device)

        try:
            out = model(toks, return_contacts=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                stats["skipped_oom"] += len(kept)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
        
        contacts = out["contacts"].detach().cpu().numpy()  # [B, L, L]
        for b, rec in enumerate(kept):
            pred = contacts[b]
            L = rec["length"]
            pred = pred[:L, :L]
            gt = rec["contact_map"][:L, :L].astype(bool)
            ii, jj = np.triu_indices(L, k=1)
            long_mask = (jj - ii) >= long_range_sep
            ii = ii[long_mask]
            jj = jj[long_mask]
            if len(ii) == 0:
                per = float("nan")
            else:
                scores = pred[ii, jj]
                labels = gt[ii, jj]
                topk = min(L, len(scores))
                if topk <= 0:
                    per = float("nan")
                else:
                    top_idx = np.argpartition(-scores, topk - 1)[:topk]
                    per = float(labels[top_idx].mean())
            per_item.append({"id": rec["id"], "long_range_pl": per, "length": L})
            stats["evaluated"] += 1

        del toks, out, contacts, batch, kept, chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    vals = [x["long_range_pl"] for x in per_item if not np.isnan(x["long_range_pl"])]
    score = float(np.mean(vals)) if vals else float("nan")
    return score, per_item, stats


def train_lora_esm(model, alphabet, batch_converter, train_records, val_records, cfg: FTConfig, args):
    device = next(model.parameters()).device
    model.train()
    has_lora = freeze_base_params(model)
    if not has_lora:
        return {"train_loss_traj": [], "steps_done": 0, "eval_ppl_per_step": []}
    lora_params = [p for _, p in model.named_parameters() if p.requires_grad]
    if len(lora_params) == 0:
        return {"train_loss_traj": [], "steps_done": 0, "eval_ppl_per_step": []}

    from transformers import get_linear_schedule_with_warmup
    opt = AdamW(lora_params, lr=cfg.lr)
    sch = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=int(cfg.warmup_ratio * cfg.steps),
        num_training_steps=cfg.steps,
    )
    rng = np.random.RandomState(42)
    hist, hist_ppl = [], []
    best = float("inf")
    no_improve = 0
    for step in range(1, cfg.steps + 1):
        total = 0.0
        for _ in range(cfg.grad_accum):
            idx = rng.choice(len(train_records), size=cfg.batch_size, replace=(len(train_records) < cfg.batch_size))
            batch = [(train_records[i]["id"], train_records[i]["sequence"]) for i in idx]
            _, _, toks = batch_converter(batch)
            toks = toks.to(device)
            inp, labels = _make_masked_batch(toks, alphabet, mask_prob=args.esm_mask_prob)
            out = model(inp)
            loss = _esm_masked_ce_loss(out, labels) / cfg.grad_accum
            loss.backward()
            total += float(loss.item())
        if cfg.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(lora_params, cfg.clip_grad_norm)
        opt.step()
        sch.step()
        model.zero_grad(set_to_none=True)
        hist.append(total)
        if best - total > cfg.early_stop_delta:
            best = total
            no_improve = 0
        else:
            no_improve += 1
        if step % max(1, cfg.eval_every) == 0:
            ppl, ppl_base_stats = eval_esm_pseudo_ppl(
                model,
                val_records,
                alphabet,
                batch_converter,
                batch_size=max(1, args.bs),
                mask_prob=args.esm_mask_prob,
                max_eval=args.esm_eval_max_items,
            )
            hist_ppl.append(float(ppl))
            print(f"[ESM-FT] step {step}/{cfg.steps} | train_loss={total:.4f} | eval_pPPL={ppl:.4f}")
        if no_improve >= cfg.early_stop_patience:
            print(f"[ESM-FT early-stop] no improvement {no_improve} steps")
            break
    return {"train_loss_traj": hist, "steps_done": len(hist), "eval_ppl_per_step": hist_ppl}


def _load_esm_model(args):
    model_name = args.esm_model_name
    local_pt = Path(args.esm_local_model_pt).expanduser()
    if local_pt.exists():
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(local_pt))
    else:
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.cuda() if torch.cuda.is_available() else model
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter


def _build_or_load_fixed_subset(test_records, out_dir, subset_n=500, seed=42):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "fixed_subset_500.json"
    if manifest.exists():
        with manifest.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        wanted = set(payload.get("ids", []))
        subset = [r for r in test_records if r["id"] in wanted]
        if len(subset) == len(wanted):
            return subset, manifest
    idx = _stratified_sample_indices(test_records, subset_n, seed=seed)
    subset = [test_records[i] for i in idx]
    payload = {"seed": seed, "ids": [x["id"] for x in subset], "size": len(subset)}
    with manifest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return subset, manifest


def run_job_esm(job, out_dir, args):
    os.makedirs(out_dir, exist_ok=True)

    records, dropped = load_structure_records(
        data_path=args.esm_data_path,
        data_format=args.esm_data_format,
        max_records=args.esm_max_records if args.esm_max_records > 0 else None,
    )
    write_dropped_records(dropped, Path(out_dir) / "dropped_samples.csv")

    split_manifest = Path(out_dir) / "split_manifest.json"
    train_idx, test_idx, val_idx = _split_structure_records(
        records,
        split_manifest_path=split_manifest,
        train_n=args.esm_train_size,
        test_n=args.esm_test_size,
        seed=args.esm_split_seed,
    )
    train_records = [records[i] for i in train_idx]
    test_records = [records[i] for i in test_idx]
    val_records = [records[i] for i in val_idx]

    model, alphabet, batch_converter = _load_esm_model(args)

    target_modules = job.get("target_modules", args.esm_target_modules)
    target_modules = parse_target_modules(target_modules)
    rank = int(job.get("rank", job.get("r", args.esm_rank)))
    seed = int(job.get("seed", args.SEED))
    set_all_seeds(seed)

    if rank > 0:
        model, native_lora_meta = build_native_esm_lora(
            model,
            target_modules=target_modules,
            r=rank,
            lora_alpha=rank,
            lora_dropout=0.05,
        )
    else:
        native_lora_meta = {"injected": 0, "matched": [], "missing": []}

    # -------------------------
    # Fixed subset
    # -------------------------
    fixed_subset, subset_manifest = _build_or_load_fixed_subset(
        test_records,
        out_dir=Path(out_dir),
        subset_n=args.esm_fixed_subset_size,
        seed=args.esm_split_seed,
    )

    # -------------------------
    # Base eval
    # -------------------------
    ppl_base, ppl_base_stats = eval_esm_pseudo_ppl(
        model,
        test_records,
        alphabet,
        batch_converter,
        batch_size=max(1, args.bs),
        mask_prob=args.esm_mask_prob,
        max_eval=args.esm_eval_max_items,
    )
    clear_cuda("after ppl_base")

    pl_full_base, per_full_base, pl_full_base_stats = eval_long_range_pl(
        model,
        test_records,
        alphabet,
        batch_converter,
        batch_size=args.esm_contact_batch_size,
    )
    clear_cuda("after pl_full_base")

    pl_fixed_base, _, pl_fixed_base_stats = eval_long_range_pl(
        model,
        fixed_subset,
        alphabet,
        batch_converter,
        batch_size=args.esm_contact_batch_size,
    )
    clear_cuda("after pl_fixed_base")

    # -------------------------
    # Stage 1
    # -------------------------
    ft1 = FTConfig(
        steps=args.s1_steps,
        lr=args.s1_lr,
        seq_len=args.seq_len,
        batch_size=args.bs,
        grad_accum=args.ga,
        eval_every=args.eval_every,
        early_stop_patience=args.es_patience,
        early_stop_delta=args.es_delta,
    )
    ft1_log = train_lora_esm(model, alphabet, batch_converter, train_records, val_records, ft1, args)

    ppl_s1, ppl_s1_stats = eval_esm_pseudo_ppl(
        model,
        test_records,
        alphabet,
        batch_converter,
        batch_size=max(1, args.bs),
        mask_prob=args.esm_mask_prob,
        max_eval=args.esm_eval_max_items,
    )
    clear_cuda("after ppl_s1")

    pl_full_s1, per_full_s1, pl_full_s1_stats = eval_long_range_pl(
        model,
        test_records,
        alphabet,
        batch_converter,
        batch_size=args.esm_contact_batch_size,
    )
    clear_cuda("after pl_full_s1")

    pl_fixed_s1, _, pl_fixed_s1_stats = eval_long_range_pl(
        model,
        fixed_subset,
        alphabet,
        batch_converter,
        batch_size=args.esm_contact_batch_size,
    )
    clear_cuda("after pl_fixed_s1")

    # -------------------------
    # Stage 2
    # -------------------------
    go_s2 = (ppl_base - ppl_s1) >= args.s2_gate_delta if args.s2_gate_delta is not None else True

    ft2_log = {"train_loss_traj": [], "steps_done": 0, "eval_ppl_per_step": []}
    ppl_s2, pl_full_s2, pl_fixed_s2 = ppl_s1, pl_full_s1, pl_fixed_s1
    per_full_s2 = per_full_s1

    # 默认沿用 s1 的 stats
    ppl_s2_stats = ppl_s1_stats
    pl_full_s2_stats = pl_full_s1_stats
    pl_fixed_s2_stats = pl_fixed_s1_stats

    if go_s2 and args.s2_steps > 0:
        ft2 = FTConfig(
            steps=args.s2_steps,
            lr=args.s2_lr,
            seq_len=args.seq_len,
            batch_size=args.bs,
            grad_accum=args.ga,
            eval_every=args.eval_every,
            early_stop_patience=args.es_patience,
            early_stop_delta=args.es_delta,
        )
        ft2_log = train_lora_esm(model, alphabet, batch_converter, train_records, val_records, ft2, args)

        ppl_s2, ppl_s2_stats = eval_esm_pseudo_ppl(
            model,
            test_records,
            alphabet,
            batch_converter,
            batch_size=max(1, args.bs),
            mask_prob=args.esm_mask_prob,
            max_eval=args.esm_eval_max_items,
        )
        clear_cuda("after ppl_s2")

        pl_full_s2, per_full_s2, pl_full_s2_stats = eval_long_range_pl(
            model,
            test_records,
            alphabet,
            batch_converter,
            batch_size=args.esm_contact_batch_size,
        )
        clear_cuda("after pl_full_s2")

        pl_fixed_s2, _, pl_fixed_s2_stats = eval_long_range_pl(
            model,
            fixed_subset,
            alphabet,
            batch_converter,
            batch_size=args.esm_contact_batch_size,
        )
        clear_cuda("after pl_fixed_s2")

    rec = {
        **job,
        "mode": "esm",
        "seed": seed,
        "target_modules": target_modules,
        "rank": rank,
        "native_lora": native_lora_meta,
        "dataset": {
            "data_path": str(args.esm_data_path),
            "train_size": len(train_records),
            "test_size": len(test_records),
            "val_size": len(val_records),
            "split_manifest": str(split_manifest),
            "fixed_subset_manifest": str(subset_manifest),
        },
        "ppl_base": ppl_base,
        "ppl_s1": ppl_s1,
        "ppl_s2": ppl_s2,
        "delta_ppl_s1": ppl_s1 - ppl_base,
        "delta_ppl_s2": ppl_s2 - ppl_base,
        "long_range_pl": {
            "full_test_base": pl_full_base,
            "full_test_s1": pl_full_s1,
            "full_test_s2": pl_full_s2,
            "fixed_500_base": pl_fixed_base,
            "fixed_500_s1": pl_fixed_s1,
            "fixed_500_s2": pl_fixed_s2,
            "delta_full_s2": pl_full_s2 - pl_full_base,
            "delta_fixed_s2": pl_fixed_s2 - pl_fixed_base,
        },
        "s1_log": ft1_log,
        "s2_log": ft2_log,
        "lora_params": count_lora_params(model),
        "tokens_trained": (ft1_log["steps_done"] + ft2_log["steps_done"]) * args.bs * args.ga,
        "eval_stats": {
            "ppl_base": ppl_base_stats,
            "ppl_s1": ppl_s1_stats,
            "ppl_s2": ppl_s2_stats,
            "long_range_full_base": pl_full_base_stats,
            "long_range_full_s1": pl_full_s1_stats,
            "long_range_full_s2": pl_full_s2_stats,
            "long_range_fixed_base": pl_fixed_base_stats,
            "long_range_fixed_s1": pl_fixed_s1_stats,
            "long_range_fixed_s2": pl_fixed_s2_stats,
        },
    }

    fname = f"esm_single_seed{seed}_r{rank}.json"
    if "name" in job:
        safe_name = re.sub(r"[^a-zA-Z0-9_\\-]+", "_", str(job["name"]))
        fname = f"{safe_name}_seed{seed}_r{rank}.json"

    out_json = os.path.join(out_dir, fname)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)

    per_item_path = os.path.join(out_dir, fname.replace(".json", "_per_item_s2.csv"))
    with open(per_item_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "long_range_pl", "length"])
        writer.writeheader()
        for row in per_full_s2:
            writer.writerow(row)

    rec["long_range_pl"]["full_test_s2_per_item_csv"] = per_item_path

    # 如果你希望 JSON 里也带上 per-item csv 路径，可以再写一遍
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)

    hard_free(model, alphabet, batch_converter, tag="run_job_esm end")
    model = alphabet = batch_converter = None
    return rec

# def extract_gsm_num(text):
#     m = re.search(r"####\s*(\d+)", text)
#     if not m: return None
#     try:
#         return int(m.group(1).replace(",", ""))
#     except Exception:
#         return None
def extract_gsm_num(text):
   nums = re.findall(r"[-+]?\d[\d,]*", text)
   if not nums: 
       return None
   try:
       return int(nums[-1].replace(",", ""))
   except Exception:
       return None

TEMPLATE = (
    "You are a math problem solver. Solve the problem step by step, "
    "then output ONLY the final integer answer on the LAST line in the exact format:\n"
    "#### <answer>\n\n"
    "Question: {q}\n\n"
    "Solution:"
)

@torch.no_grad()
# def eval_gsm8k_accuracy(model, tokenizer, data_root="./data/gsm8k/main",
#                         split="test", max_prompt_len=768, gen_len=512,
#                         bs=8, do_sample=False, temperature=0.8, top_p=0.95):
def eval_gsm8k_accuracy(model, tokenizer, data_root="./data/gsm8k/main",
                        split="test", max_prompt_len=768, gen_len=128,
                        bs=32, do_sample=False, temperature=0.8, top_p=0.95,
                        limit=512):
    
    if os.path.isdir(data_root):
        ds = load_from_disk(data_root)[split]        # 读取本地磁盘格式
    else:
        ds = load_dataset("openai/gsm8k", "main", split=split)  # 兜底走 Hub
    
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    # 后续按你的 prompt/生成/抽取逻辑即可（期望列名就是 question/answer）
    # 组 prompt + 参考答案
    X, Y = [], []
    for q, a in zip(ds["question"], ds["answer"]):
        X.append(TEMPLATE.format(q=q) + " ")
        # 提前把参考答案抽出成数字
        y = extract_gsm_num(a)
        Y.append(y if y is not None else -10**9)

    orig_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc = tokenizer(
        X, return_tensors="pt", max_length=max_prompt_len,
        padding="max_length", truncation=True, return_token_type_ids=False
    )
    ids, am = enc["input_ids"], enc["attention_mask"]
    device = next(model.parameters()).device

    acc, i = 0, 0
    while i < ids.size(0):
        j = min(i + bs, ids.size(0))
        # out = model.generate(
        #     ids[i:j].to(device),
        #     attention_mask=am[i:j].to(device),
        #     return_dict_in_generate=True,
        #     output_scores=False, max_new_tokens=gen_len,
        #     eos_token_id=tokenizer.eos_token_id,
        #     top_p=top_p, temperature=temperature
        # )
        sampling_kw = {"do_sample": True, "temperature": temperature, "top_p": top_p} if do_sample else {}
        out = model.generate(
            ids[i:j].to(device),
            attention_mask=am[i:j].to(device),
            return_dict_in_generate=True,
            output_scores=False, max_new_tokens=gen_len,
            eos_token_id=tokenizer.eos_token_id,
            **sampling_kw
        )
        texts = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
        preds = [extract_gsm_num(t) for t in texts]
        for p, y in zip(preds, Y[i:j]):
            acc += int(p == y)
        i = j

    tokenizer.padding_side = orig_pad_side

    print("\033[92m[ok]\033[0m")
    pprint({
        "acc": acc,
        "len Y": len(Y),
    })
    return acc / len(Y)


# -------------------------
# 任务运行
# -------------------------
def run_job(job, base_model_id, out_dir, args):
    os.makedirs(out_dir, exist_ok=True)
    if getattr(args, "model_family", "causal_lm") == "esm2":
        return run_job_esm(job, out_dir, args)

    # === 加载模型/分词器（沿用你原脚本） ===
    cache_dir = os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME")
    model_path = resolve_model_path(base_model_id, cache_dir=cache_dir)
    
    common_kwargs = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     base_model_id,
    #     device_map=("auto" if not args.force_cuda0 else {"": "cuda:0"}),
    #     **common_kwargs
    # )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=("auto" if not args.force_cuda0 else {"": "cuda:0"}),
        local_files_only=True,
        **common_kwargs,
    )
    # tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    # if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    if hasattr(model.config, "use_cache"): model.config.use_cache = False
    model.gradient_checkpointing_enable()

    gsm_acc_base = None
    if os.path.exists("./data/gsm8k/main"):
        print("\033[92m[ok]\033[0m")
        gsm_acc_base = eval_gsm8k_accuracy(model, tok)  # 这里

    # === 评测集（沿用你原脚本的加载函数） ===
    train_texts = load_train_texts(name=args.train_dataset, config=args.train_config, split="train", n=args.train_count)
    eval_texts = load_eval_texts(name=args.eval_dataset, config=args.eval_config, split=args.eval_split, n=args.eval_count)

    # baseline ppl/lat（使用优化后的评估函数）
    ppl_base = eval_ppl(
        model, tok, train_texts,
        max_length=args.ppl_max_len,
        max_eval_tokens=args.ppl_max_tokens,
        stride=args.ppl_stride if args.ppl_stride is not None else args.ppl_max_len
    )

    val_ppl_base = eval_ppl(
        model, tok, eval_texts,
        max_length=args.ppl_max_len,
        max_eval_tokens=args.ppl_max_tokens,
        stride=args.ppl_stride if args.ppl_stride is not None else args.ppl_max_len
    )

    lat_base, mem_base = measure_step_latency(model, tok, seq_len=args.seq_len)

    harness_base = {}
    harness_base_path = None
    if args.harness_enable and args.harness_when in ("baseline", "all"):
        tag = f"baseline_L{job['layer']}_{job.get('pos', job.get('pos1','NA'))}_seed{job['seed']}"
        harness_base, harness_base_path = run_harness_eval_from_model(model, tok, base_model_id, args, tag)

    # === 选择 target modules（沿用你的 select_targets） ===
    if job["type"] == "single":
        targets = select_targets(model, job["layer"], [job["pos"]])
        model, status, msg = build_peft_loraga(
            model, tok, eval_texts, targets, job["rank"],
            ga_seq_len=args.ga_seq_len, ga_bs=args.ga_bs, ga_samples=args.ga_samples
        )
        meta = {"targets": targets, "rank": job["rank"]}
    else:
        targets = select_targets(model, job["layer"], [job["pos1"], job["pos2"]])
        model, status, msg = build_peft_loraga(
            model, tok, eval_texts, [targets[0]], job["rank1"],
            ga_seq_len=args.ga_seq_len, ga_bs=args.ga_bs, ga_samples=args.ga_samples
        )
        model, status, msg = build_peft_loraga(
            model, tok, eval_texts, [targets[1]], job["rank2"],
            ga_seq_len=args.ga_seq_len, ga_bs=args.ga_bs, ga_samples=args.ga_samples
        )
        meta = {"targets": targets, "rank1": job["rank1"], "rank2": job["rank2"]}
    # === A补丁：若无可训练 LoRA 参数（r=0 或未命中），跳过 S1/S2 ===
    trainable = count_lora_params(model)
    if trainable == 0:
        print("[info] no LoRA params to train (rank=0 or no targets). Skip fine-tuning.")
        
        ppl_s1 = ppl_base
        ppl_s2 = ppl_base
        val_ppl = val_ppl_base

        lat, mem = measure_step_latency(model, tok, seq_len=args.seq_len)
        rec = {
            **job, **meta,
            "ppl_base": ppl_base, "ppl_s1": ppl_s1, "ppl_s2": ppl_s2, "val_ppl":val_ppl,
            "delta_ppl_s1": (ppl_s1 - ppl_base), "delta_ppl_s2": (ppl_s2 - ppl_base), "delta_val_ppl": (val_ppl - val_ppl_base),
            "gsm_acc_base": gsm_acc_base,
            "lat_base": lat_base, "lat": lat, "delta_lat": (lat - lat_base),
            "mem_base_mb": mem_base, "mem_mb": mem,
            "s1_log": {"train_loss_traj": [], "steps_done": 0, "eval_ppl_per_step": []},
            "s2_log": {"train_loss_traj": [], "steps_done": 0, "eval_ppl_per_step": []},
            "lora_params": count_lora_params(model),
            "tokens_trained": 0,
        }
        def _mk_fname(job):
            if job["type"] == "single":
                return f"single_L{job['layer']}_{job['pos']}_r{job['rank']}_seed{job['seed']}.json"
            else:
                return (f"pair_L{job['layer']}_{job['pos1']}_{job['pos2']}"
                        f"_r{job['rank1']}-{job['rank2']}_seed{job['seed']}.json")

        fname = _mk_fname(job)
        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump(rec, f, indent=2)
        
        def unload_model(model, tokenizer=None):
            if tokenizer is not None:
                del tokenizer
            if model is not None:
                del model

            # 2) 清理 PyTorch 与 Python 引用
            gc.collect()

            # 3) 等GPU上所有kernel结束（否则有"正在用的内存"无法释放）
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()      # 释放 PyTorch 缓存到 CUDA driver
                torch.cuda.ipc_collect()      # 回收跨进程句柄
        
        def print_mem(tag=""):
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024**2
                reserv = torch.cuda.memory_reserved() / 1024**2
                print(f"[{tag}] allocated={alloc:.1f}MB reserved={reserv:.1f}MB")

        unload_model(model)
        print_mem("after unload")
        return rec

    # Stage 1
    ft1 = FTConfig(steps=args.s1_steps, lr=args.s1_lr, seq_len=args.seq_len,
                   batch_size=args.bs, grad_accum=args.ga, eval_every=args.eval_every,
                   early_stop_patience=args.es_patience, early_stop_delta=args.es_delta)
    # ft1_log, val_ppl_s1 = train_lora(model, tok, train_texts, eval_texts, ft1, seed_mode=args.seed_mode)
    ft1_log, val_ppl_s1 = train_lora(model, tok, train_texts, eval_texts, ft1, args, seed_mode=args.seed_mode)
    
    ppl_s1 = eval_ppl(model, tok, train_texts,
                  max_length=args.ppl_max_len,
                  max_eval_tokens=args.ppl_max_tokens,
                  stride=args.ppl_stride if args.ppl_stride is not None else args.ppl_max_len)

    # harness_s1 = {}
    # harness_s1_path = None
    # if args.harness_enable and (args.harness_when in ("s1", "s1,s2", "all")):
    #     tag = f"s1_L{job['layer']}_{job.get('pos', job.get('pos1','NA'))}_seed{job['seed']}"
    #     harness_s1, harness_s1_path = run_harness_eval_from_model(model, tok, base_model_id, args, tag)

    # 进入 Stage 2 的门槛
    go_s2 = (ppl_base - ppl_s1) >= args.s2_gate_delta if args.s2_gate_delta is not None else True
    
    val_ppl_s2 = val_ppl_s1
    ppl_s2 = ppl_s1
    ft2_log = {"train_loss_traj": [], "steps_done": 0, "eval_ppl_per_step": []}
    if go_s2 and args.s2_steps > 0:
        ft2 = FTConfig(steps=args.s2_steps, lr=args.s2_lr, seq_len=args.seq_len,
                       batch_size=args.bs, grad_accum=args.ga, eval_every=args.eval_every,
                       early_stop_patience=args.es_patience, early_stop_delta=args.es_delta)
        # ft2_log, val_ppl_s2 = train_lora(model, tok, train_texts, eval_texts, ft2, seed_mode=args.seed_mode)
        ft2_log, val_ppl_s2 = train_lora(model, tok, train_texts, eval_texts, ft2, args, seed_mode=args.seed_mode)

        ppl_s2 = eval_ppl(model, tok, train_texts,
                  max_length=args.ppl_max_len,
                  max_eval_tokens=args.ppl_max_tokens,
                  stride=args.ppl_stride if args.ppl_stride is not None else args.ppl_max_len)

    lat, mem = measure_step_latency(model, tok, seq_len=args.seq_len)

    harness_s2 = {}
    harness_s2_path = None
    if args.harness_enable and (args.harness_when in ("s2", "s1,s2", "all")):
        tag = f"s2_L{job['layer']}_{job.get('pos', job.get('pos1','NA'))}_seed{job['seed']}"
        harness_s2, harness_s2_path = run_harness_eval_from_model(model, tok, base_model_id, args, tag)

    gsm_acc_s2 = None
    if os.path.exists("./data/gsm8k/main"):
        print("\033[92m[ok]\033[0m")
        gsm_acc_s2 = eval_gsm8k_accuracy(model, tok)

    # del model
    gc.collect()
    torch.cuda.empty_cache()

    rec = {
        **job, **meta,
        "ppl_base": ppl_base, "ppl_s1": ppl_s1, "ppl_s2": ppl_s2, "val_ppl": val_ppl_s2,
        "delta_ppl_s1": (ppl_s1 - ppl_base), "delta_ppl_s2": (ppl_s2 - ppl_base),
        "val_ppl_base": val_ppl_base,
        "delta_val_ppl": (val_ppl_s2 - val_ppl_base),
        "lat_base": lat_base, "lat": lat, "delta_lat": (lat - lat_base),
        "mem_base_mb": mem_base, "mem_mb": mem,
        "s1_log": ft1_log, "s2_log": ft2_log,
        "lora_params": count_lora_params(model),
        "tokens_trained": (ft1_log["steps_done"] + ft2_log["steps_done"]) * args.bs * args.seq_len * args.ga,

        # === 新增：harness 结果聚合 ===
        "harness": {
            "base": {"metrics": harness_base, "json": harness_base_path},
            # "s1":   {"metrics": harness_s1,   "json": harness_s1_path},
            "s2":   {"metrics": harness_s2,   "json": harness_s2_path},
        },

        "gsm8k": {
        "base_acc": gsm_acc_base,
        "s2_acc": gsm_acc_s2,
        "delta_acc": (None if (gsm_acc_base is None or gsm_acc_s2 is None)
                      else (gsm_acc_s2 - gsm_acc_base)),
        },
    }

    def _get_ppl_from(primary_dict):
        # 兼容 "wikitext.word_perplexity" / "wikitext.perplexity" / "ppl"
        if not primary_dict: return None
        for k, v in primary_dict.items():
            kk = k.lower()
            if "perplexity" in kk or kk.endswith(".ppl") or kk == "ppl":
                return float(v)
        return None

    # 在 run_job() 末尾、构造 rec 之后追加（或并入 rec 构造处）
    h_base = _get_ppl_from(harness_base)
    h_s2   = _get_ppl_from(harness_s2)
    if h_base is not None and h_s2 is not None:
        rec["harness"]["delta_ppl"] = h_s2 - h_base
        rec["harness"]["rel_improve"] = (h_base - h_s2) / max(h_base, 1e-12)

    def _mk_fname(job):
        if job["type"] == "single":
            return f"single_L{job['layer']}_{job['pos']}_r{job['rank']}_seed{job['seed']}.json"
        else:
            return (f"pair_L{job['layer']}_{job['pos1']}_{job['pos2']}"
                    f"_r{job['rank1']}-{job['rank2']}_seed{job['seed']}.json")

    fname = _mk_fname(job)
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump(rec, f, indent=2)

    def unload_model(model, tokenizer=None):
        if tokenizer is not None:
            del tokenizer
        if model is not None:
            del model

        # 2) 清理 PyTorch 与 Python 引用
        gc.collect()

        # 3) 等GPU上所有kernel结束（否则有"正在用的内存"无法释放）
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()      # 释放 PyTorch 缓存到 CUDA driver
            torch.cuda.ipc_collect()      # 回收跨进程句柄

    def print_mem(tag=""):
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**2
            reserv = torch.cuda.memory_reserved() / 1024**2
            print(f"[{tag}] allocated={alloc:.1f}MB reserved={reserv:.1f}MB")

    unload_model(model)
    print_mem("after unload")
    
    return rec

def make_lora_ft(cfg_path, model_id=None, out_dir=None):
    
    start_time = time.time()

    here = Path(__file__).resolve().parent
    configs_path = here / cfg_path

    rec=None

    ap = argparse.ArgumentParser()
    
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B") # Qwen/Qwen2.5-1.5B / meta-llama/Llama-3.1-8B
    # ap.add_argument("--configs_path", type=str, required=True) # <---
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--model_family", type=str, default="causal_lm", choices=["causal_lm", "esm2"])

    # ESM2 / offline protein settings
    ap.add_argument("--esm_model_name", type=str, default="esm2_t36_3B_UR50D")
    ap.add_argument("--esm_local_model_pt", type=str, default="./assets/models/esm2_t36_3B_UR50D/esm2_t36_3B_UR50D.pt")
    ap.add_argument("--esm_data_path", type=str, default="./assets/data/tr_rosetta")
    ap.add_argument("--esm_data_format", type=str, default="auto", choices=["auto", "npz", "sidechainnet"])
    ap.add_argument("--esm_train_size", type=int, default=12000)
    ap.add_argument("--esm_test_size", type=int, default=2300)
    ap.add_argument("--esm_split_seed", type=int, default=42)
    ap.add_argument("--esm_fixed_subset_size", type=int, default=500)
    ap.add_argument("--esm_max_records", type=int, default=0, help="0=use all")
    ap.add_argument("--esm_mask_prob", type=float, default=0.15)
    ap.add_argument("--esm_eval_max_items", type=int, default=512)
    ap.add_argument("--esm_contact_batch_size", type=int, default=1)
    ap.add_argument("--esm_rank", type=int, default=8)
    ap.add_argument(
        "--esm_target_modules",
        type=str,
        default='["attention.self.query","attention.self.key","attention.self.value","attention.output.dense","intermediate.dense","output.dense"]',
    )

    # 设备/内存策略
    ap.add_argument("--force_cuda0", action="store_true", help="禁用 auto offload，强制整机放到 cuda:0（显存足够时使用）")

    # 数据/评测

    ap.add_argument("--train_dataset", type=str, default="wikitext")
    ap.add_argument("--train_config", type=str, default="wikitext-2-raw-v1")  
    ap.add_argument("--train_count", type=int, default=1000)

    ap.add_argument("--eval_dataset", type=str, default="wikitext")
    ap.add_argument("--eval_config", type=str, default="wikitext-2-raw-v1")
    ap.add_argument("--eval_split", type=str, default="test")
    ap.add_argument("--eval_count", type=int, default=500)

    # Stage 0
    ap.add_argument("--s0_seq_len", type=int, default=256)

    # Stage 1
    ap.add_argument("--s1_steps", type=int, default=100)
    ap.add_argument("--s1_lr", type=float, default=1e-4)

    # Stage 2
    ap.add_argument("--s2_steps", type=int, default=1000)
    ap.add_argument("--s2_lr", type=float, default=8e-5)
    ap.add_argument("--s2_gate_delta", type=float, default=-0.02, help="进入S2的门槛：ppl_base - ppl_s1 >= gate")

    # 训练 & 早停 & 批配置
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--ga", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--es_patience", type=int, default=100000)
    ap.add_argument("--es_delta", type=float, default=1e-4) # 0.0001

    ap.add_argument("--ga_seq_len", type=int, default=256, help="LoRA-GA 梯度估计的序列长度")
    ap.add_argument("--ga_bs", type=int, default=2, help="LoRA-GA 梯度估计的batch size")
    ap.add_argument("--ga_samples", type=int, default=16, help="用于估计梯度的样本数上限")

    # 任务采样
    ap.add_argument("--max_items", type=int, default=1)

    ap.add_argument("--seed_mode", type=str, default="deterministic",
                choices=["deterministic","stochastic"])
    
    ap.add_argument("--harness_enable", action="store_true",
                help="开启 lm-evaluation-harness 评测（baseline/S1/S2 可选）")
    ap.add_argument("--harness_when", type=str, default="s2",
                    choices=["baseline", "s1", "s2", "s1,s2", "all"],
                    help="在哪些阶段跑 harness：baseline / s1 / s2 / s1,s2 / all")
    ap.add_argument("--harness_tasks", type=str, default="wikitext",
                    help="逗号分隔任务列表，如 wikitext,hellaswag,piqa")
    ap.add_argument("--harness_limit", type=int, default=None,
                    help="每个任务取样上限（None=全量，调试建议 <=200）")
    ap.add_argument("--harness_batch_size", type=int, default=1)
    ap.add_argument("--harness_bootstrap", type=int, default=0)
    ap.add_argument("--harness_device", type=str, default=None,
                    help="harness 的 device（如 cuda / cpu）。为空则使用默认放置策略")
    ap.add_argument("--harness_out_dir", type=str, default="harness_runs",
                    help="harness 结果 JSON 输出目录")
    
    # 评测窗口策略（新增）
    ap.add_argument("--ppl_max_tokens", type=int, default=32768,
                help="参与PPL评测的最大token数（从拼接后的文本前缀截取）")
    ap.add_argument("--ppl_max_len", type=int, default=1024,
                help="评测窗口的最大长度（每个chunk的token数上限）")
    ap.add_argument("--ppl_stride", type=int, default=512,
                help="滑窗步长；默认None表示与ppl_max_len相同（不重叠）")
    
    ap.add_argument("--harness_no_4bit", action="store_true",
                help="评测时禁用 4bit 量化，与 eval_ppl 的精度一致")
    
    ap.add_argument("--ALGO", type=str, default="Coflex")

    ap.add_argument("--SEED", type=int, default=42)

    ap.add_argument('--MODEL_ID', type=str, default='Qwen/Qwen2.5-1.5B')
    
    args = ap.parse_args()

    if model_id is not None:
        args.model_id = model_id

    if args.model_family == "esm2":
        global_task_type = TaskType.FEATURE_EXTRACTION
    else:
        global_task_type = TaskType.CAUSAL_LM
    
    print(f"[info] global_task_type = {global_task_type}", flush=True)

    # Qwen/Qwen2.5-1.5B / meta-llama/Llama-3.1-8B
    if args.model_id == 'Qwen/Qwen2.5-1.5B':
        args.s2_steps = 1000
    elif args.model_id == 'meta-llama/Llama-3.1-8B':
        # args.s1_steps = 30
        # args.s2_steps = 400
        # args.seq_len = 256
        # args.ga_seq_len = 128
        # args.ga_samples = 8
        # args.eval_every = 20
        # args.train_count = min(args.train_count, 400)
        # args.eval_count = min(args.eval_count, 200)
        # args.s1_lr = 2e-4
        # args.s2_lr = 1e-4
        args.s2_steps = 300
    elif args.model_id == 'meta-llama/Llama-3.2-3B':
        args.s2_steps = 500

    with open(configs_path) as f:
        jobs = json.load(f)
    if args.max_items and args.max_items > 0:
        jobs = jobs[:args.max_items]

    print(f"\033[92m[ok]\033[0m loaded {len(jobs)} job(s) from: {configs_path}")
    print(json.dumps(jobs, indent=2, ensure_ascii=False))

    # args.out_dir
    if out_dir is not None:
        args.out_dir = out_dir
    
    print("\033[92m[ok]\033[0m")
    pprint({
        "model_id": args.model_id,
        "out_dir": args.out_dir,
    })  

    rec = {"val_ppl": float("nan"), "delta_val_ppl": float("nan")}
    for i, job in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] run: {job}")
        # set_all_seeds(job['seed'])
        try:

            print("\033[92m[ok]\033[0m")
            pprint({
                "s1_steps": args.s1_steps,
                "s2_steps": args.s2_steps,
            })   

            rec = run_job(job, args.model_id, args.out_dir, args)
            if args.model_family == "esm2":
                print(
                    "[ok] ppl_s2=", rec.get("ppl_s2"),
                    "| Δ=", rec.get("delta_ppl_s2"),
                    "| long_range_full=", rec.get("long_range_pl", {}).get("full_test_s2"),
                    "| long_range_fixed=", rec.get("long_range_pl", {}).get("fixed_500_s2"),
                )
                print(
                    "[eval-summary] long_range_full:",
                    rec.get("eval_stats", {}).get("long_range_full_s2"),
                    flush=True,
                )
            else:
                print(
                    "[ok] ppl_s2=", rec.get("ppl_s2"),
                    "| Δ=", rec.get("delta_ppl_s2"),
                    "| val_ppl=", rec.get("val_ppl"),
                    "| val_Δ=", rec.get("delta_val_ppl"),
                )
        except RuntimeError as e:
            print("[fail][RuntimeError]", e)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return 11, 0.0
        except Exception as e:
            if _is_hf_network_error(e):
                print("[info] Hugging Face network error ignored:", e)
                continue
            print("[fail]", e)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return 11, 0.0
        
    print(f"\033[1;31m[ Time used: --------------------------------------------------- {time.time() - start_time} --------------------------------------------------- ]\033[0m")

    if args.model_family == "esm2":
        return rec.get("ppl_s2"), rec.get("delta_ppl_s2")
    return rec.get("val_ppl"), rec.get("delta_val_ppl")

# -------------------------
# main
# -------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--configs_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--model_family", type=str, default="causal_lm", choices=["causal_lm", "esm2"])

    # ESM2 / offline protein settings
    ap.add_argument("--esm_model_name", type=str, default="esm2_t36_3B_UR50D")
    ap.add_argument("--esm_local_model_pt", type=str, default="./assets/models/esm2_t36_3B_UR50D/esm2_t36_3B_UR50D.pt")
    ap.add_argument("--esm_data_path", type=str, default="./assets/data/tr_rosetta")
    ap.add_argument("--esm_data_format", type=str, default="auto", choices=["auto", "npz", "sidechainnet"])
    ap.add_argument("--esm_train_size", type=int, default=12000)
    ap.add_argument("--esm_test_size", type=int, default=2300)
    ap.add_argument("--esm_split_seed", type=int, default=42)
    ap.add_argument("--esm_fixed_subset_size", type=int, default=500)
    ap.add_argument("--esm_max_records", type=int, default=0, help="0=use all")
    ap.add_argument("--esm_mask_prob", type=float, default=0.15)
    ap.add_argument("--esm_eval_max_items", type=int, default=512)
    ap.add_argument("--esm_contact_batch_size", type=int, default=1)
    ap.add_argument("--esm_rank", type=int, default=8)
    ap.add_argument(
        "--esm_target_modules",
        type=str,
        default='["attention.self.query","attention.self.key","attention.self.value","attention.output.dense","intermediate.dense","output.dense"]',
    )

    # 设备/内存策略
    ap.add_argument("--force_cuda0", action="store_true", help="禁用 auto offload，强制整机放到 cuda:0（显存足够时使用）")

    # 数据/评测

    ap.add_argument("--train_dataset", type=str, default="wikitext")
    ap.add_argument("--train_config", type=str, default="wikitext-2-raw-v1")  
    ap.add_argument("--train_count", type=int, default=1000)

    ap.add_argument("--eval_dataset", type=str, default="wikitext")
    ap.add_argument("--eval_config", type=str, default="wikitext-2-raw-v1")
    ap.add_argument("--eval_split", type=str, default="test")
    ap.add_argument("--eval_count", type=int, default=500)

    # Stage 0
    ap.add_argument("--s0_seq_len", type=int, default=256)

    # Stage 1
    ap.add_argument("--s1_steps", type=int, default=50)
    ap.add_argument("--s1_lr", type=float, default=1e-4)

    # Stage 2
    ap.add_argument("--s2_steps", type=int, default=300)
    ap.add_argument("--s2_lr", type=float, default=8e-5)
    ap.add_argument("--s2_gate_delta", type=float, default=-0.02, help="进入S2的门槛：ppl_base - ppl_s1 >= gate")

    # 训练 & 早停 & 批配置
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--ga", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--es_patience", type=int, default=100000)
    ap.add_argument("--es_delta", type=float, default=1e-4)

    ap.add_argument("--ga_seq_len", type=int, default=256, help="LoRA-GA 梯度估计的序列长度")
    ap.add_argument("--ga_bs", type=int, default=1, help="LoRA-GA 梯度估计的batch size")
    ap.add_argument("--ga_samples", type=int, default=16, help="用于估计梯度的样本数上限")

    # 任务采样
    ap.add_argument("--max_items", type=int, default=0)

    ap.add_argument("--seed_mode", type=str, default="deterministic",
                choices=["deterministic","stochastic"])
    
    ap.add_argument("--harness_enable", action="store_true",
                help="开启 lm-evaluation-harness 评测（baseline/S1/S2 可选）")
    ap.add_argument("--harness_when", type=str, default="s2",
                    choices=["baseline", "s1", "s2", "s1,s2", "all"],
                    help="在哪些阶段跑 harness：baseline / s1 / s2 / s1,s2 / all")
    ap.add_argument("--harness_tasks", type=str, default="wikitext",
                    help="逗号分隔任务列表，如 wikitext,hellaswag,piqa")
    ap.add_argument("--harness_limit", type=int, default=None,
                    help="每个任务取样上限（None=全量，调试建议 <=200）")
    ap.add_argument("--harness_batch_size", type=int, default=1)
    ap.add_argument("--harness_bootstrap", type=int, default=0)
    ap.add_argument("--harness_device", type=str, default=None,
                    help="harness 的 device（如 cuda / cpu）。为空则使用默认放置策略")
    ap.add_argument("--harness_out_dir", type=str, default="harness_runs",
                    help="harness 结果 JSON 输出目录")
    
    # 评测窗口策略（新增）
    ap.add_argument("--ppl_max_tokens", type=int, default=4096,
                help="参与PPL评测的最大token数（从拼接后的文本前缀截取）")
    ap.add_argument("--ppl_max_len", type=int, default=1024,
                help="评测窗口的最大长度（每个chunk的token数上限）")
    ap.add_argument("--ppl_stride", type=int, default=None,
                help="滑窗步长；默认None表示与ppl_max_len相同（不重叠）")
    
    ap.add_argument("--ALGO", type=str, default="Coflex")

    ap.add_argument("--SEED", type=int, default=42)

    ap.add_argument('--MODEL_ID', type=str, default='Qwen/Qwen2.5-1.5B')
    
    ap.add_argument("--harness_no_4bit", action="store_true",
                help="评测时禁用 4bit 量化，与 eval_ppl 的精度一致")
    ap.add_argument("--fast_debug", action="store_true")

    args = ap.parse_args()

    if args.fast_debug:
        args.esm_train_size = min(args.esm_train_size, 200)
        args.esm_test_size = min(args.esm_test_size, 40)
        args.esm_fixed_subset_size = min(args.esm_fixed_subset_size, 20)
        args.esm_eval_max_items = min(args.esm_eval_max_items, 20)
        args.s1_steps = 0
        args.s2_steps = 0
        print("[info] fast_debug enabled: using reduced ESM eval sizes", flush=True)
    
    if args.model_family == "esm2":
        global_task_type = TaskType.FEATURE_EXTRACTION
    else:
        global_task_type = TaskType.CAUSAL_LM
    
    print(f"[info] global_task_type = {global_task_type}", flush=True)

    if args.model_id == "meta-llama/Llama-3.1-8B":
        # args.s1_steps = 30
        # args.s2_steps = 400
        # args.seq_len = 256
        # args.ga_seq_len = 128
        # args.ga_samples = 8
        # args.eval_every = 20
        # args.train_count = min(args.train_count, 400)
        # args.eval_count = min(args.eval_count, 200)
        # args.s1_lr = 2e-4
        # args.s2_lr = 1e-4
        args.s2_steps = 300

    with open(args.configs_path) as f:
        jobs = json.load(f)
    if args.max_items and args.max_items > 0:
        jobs = jobs[:args.max_items]

    for i, job in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] run: {job}")
        
        # set_all_seeds(job['seed'])
        
        try:
            rec = run_job(job, args.model_id, args.out_dir, args)
            if args.model_family == "esm2":
                print(
                    "[ok] ppl_s2=", rec.get("ppl_s2"),
                    "| Δ=", rec.get("delta_ppl_s2"),
                    "| long_range_full=", rec.get("long_range_pl", {}).get("full_test_s2"),
                    "| long_range_fixed=", rec.get("long_range_pl", {}).get("fixed_500_s2"),
                )
                print(
                    "[eval-summary] long_range_full:",
                    rec.get("eval_stats", {}).get("long_range_full_s2"),
                    flush=True,
                )
            else:
                print(
                    "[ok] ppl_s2=", rec.get("ppl_s2"),
                    "| Δ=", rec.get("delta_ppl_s2"),
                    "| val_ppl=", rec.get("val_ppl"),
                    "| val_Δ=", rec.get("delta_val_ppl"),
                )
        except RuntimeError as e:
            print("[fail][RuntimeError]", e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            if _is_hf_network_error(e):
                print("[info] Hugging Face network error ignored:", e)
                continue
            print("[fail]", e)
