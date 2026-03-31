#!/usr/bin/env python3
"""KVTC Perplexity Benchmark -- Prove cosine similarity translates to real quality.

Measures actual perplexity delta between:
1. Original model (no compression)
2. KVTC-compressed KV cache at various configs

Uses WikiText-2 test set for standardized comparison.

Usage:
    py benchmark_perplexity.py [--model MODEL] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import common
from common import CalibrationData, PCAEntry
import entropy
import pca
import quantize
import gpu_ops
from pipeline_fast import KVTCCompressorFast
from adaptive_budget import apply_adaptive_budgets
from ans_entropy import compress_best, decompress_best


def get_vram_gb():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total = (getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)) / 1024**3
        alloc = torch.cuda.memory_allocated() / 1024**3
        return total, alloc
    return 0, 0


def load_model(name, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading {name}...")
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    try:
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map="auto", trust_remote_code=True)
    except:
        name = "Qwen/Qwen2.5-7B-Instruct"
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map="auto", trust_remote_code=True)
    m.eval()
    t, a = get_vram_gb()
    print(f"  OK: {a:.1f}/{t:.1f} GB VRAM")
    return m, tok, name


def get_wikitext_samples(tokenizer, max_length=512, n_samples=50):
    """Get WikiText-2 test samples for perplexity evaluation."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 100]
    except Exception as e:
        print(f"  WikiText load failed ({e}), using built-in corpus")
        texts = [
            "The tower of London is a historic castle located on the north bank of the River Thames in central London. It was founded towards the end of 1066 as part of the Norman Conquest. The White Tower, which gives the entire castle its name, was built by William the Conqueror in 1078 and was a resented symbol of oppression inflicted upon London by the new ruling elite. " * 3,
            "In mathematics, a group is a set equipped with an operation that combines any two elements of the set to produce a third element of the set, in such a way that the operation is associative, an identity element exists and every element has an inverse. These three conditions, called group axioms, hold for number systems and many other mathematical structures. " * 3,
            "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves. The process begins with observations or data, such as examples, direct experience, or instruction. " * 3,
            "The human brain is the central organ of the human nervous system, and with the spinal cord makes up the central nervous system. The brain consists of the cerebrum, the brainstem and the cerebellum. It controls most of the activities of the body, processing, integrating, and coordinating the information it receives from the sense organs, and making decisions as to the instructions sent to the rest of the body. " * 3,
            "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented and functional programming. It was conceived in the late 1980s by Guido van Rossum. " * 3,
        ] * 10

    samples = []
    for text in texts[:n_samples * 2]:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        if tokens["input_ids"].shape[1] >= 50:
            samples.append(tokens)
        if len(samples) >= n_samples:
            break
    
    print(f"  Got {len(samples)} samples for perplexity eval")
    return samples


def compute_perplexity_original(model, samples, device="cuda"):
    """Compute perplexity with the original (uncompressed) model."""
    print(f"  Computing baseline perplexity...")
    total_loss = 0
    total_tokens = 0
    
    for i, sample in enumerate(samples):
        input_ids = sample["input_ids"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
        
        n_tokens = input_ids.shape[1] - 1  # exclude first token
        total_loss += loss * n_tokens
        total_tokens += n_tokens
        
        if i % 10 == 0:
            print(f"    {i}/{len(samples)}...")
    
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    print(f"  Baseline PPL: {ppl:.2f}")
    return ppl


def compute_perplexity_kvtc(model, tokenizer, samples, calibration, key_bits, value_bits, use_adaptive=True, device="cuda"):
    """Compute perplexity with KVTC-compressed KV cache.
    
    Strategy: for each sample, do a forward pass to get KV cache,
    compress it, decompress it, then use the reconstructed KV cache
    to compute next-token predictions and measure loss.
    """
    dim = None
    for e in calibration.entries.values():
        dim = e.eigenvectors.shape[0]; break
    
    if use_adaptive:
        apply_adaptive_budgets(calibration, key_bits, value_bits, strength=1.0)
    else:
        for (li, gi, kind), entry in calibration.entries.items():
            entry.bit_budget = int(dim * (key_bits if kind == "keys" else value_bits))
    
    compressor = KVTCCompressorFast(calibration, device=device)
    
    total_loss = 0
    total_tokens = 0
    
    for i, sample in enumerate(samples):
        input_ids = sample["input_ids"].to(device)
        seq_len = input_ids.shape[1]
        
        if seq_len < 10:
            continue
        
        # Split: use first half as context (compress KV), second half for perplexity
        split = seq_len // 2
        context_ids = input_ids[:, :split]
        eval_ids = input_ids[:, split:]
        
        # Forward pass on context to get KV cache
        with torch.no_grad():
            context_out = model(context_ids, use_cache=True)
        
        past_kv = context_out.past_key_values
        
        # Extract and compress KV cache
        kl, vl = [], []
        for lkv in past_kv:
            kl.append(lkv[0].squeeze(0).permute(1, 0, 2))
            vl.append(lkv[1].squeeze(0).permute(1, 0, 2))
        
        keys = torch.stack(kl, dim=0).float()
        values = torch.stack(vl, dim=0).float()
        positions = torch.arange(split, dtype=torch.long, device=device)
        
        kv_cache = {"keys": keys, "values": values}
        
        # Compress and decompress
        compressed = compressor.compress(kv_cache, positions)
        reconstructed = compressor.decompress(compressed)
        
        # Rebuild past_key_values as DynamicCache (HF transformers 5.x format)
        from transformers import DynamicCache
        recon_cache = DynamicCache()
        for l in range(reconstructed["keys"].shape[0]):
            rk = reconstructed["keys"][l].to(dtype=torch.float16, device=device).unsqueeze(0).permute(0, 2, 1, 3)  # [1, heads, seq, dim]
            rv = reconstructed["values"][l].to(dtype=torch.float16, device=device).unsqueeze(0).permute(0, 2, 1, 3)
            recon_cache.update(rk, rv, l)
        recon_past = recon_cache
        
        # Forward pass on eval tokens using reconstructed KV cache
        with torch.no_grad():
            eval_out = model(eval_ids, past_key_values=recon_past, labels=eval_ids)
            loss = eval_out.loss.item()
        
        n_tokens = eval_ids.shape[1] - 1
        total_loss += loss * n_tokens
        total_tokens += n_tokens
        
        del context_out, past_kv, compressed, reconstructed, recon_past
        torch.cuda.empty_cache()
        
        if i % 10 == 0:
            print(f"    {i}/{len(samples)}...")
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 100))  # Cap to avoid overflow
    return ppl


def main():
    parser = argparse.ArgumentParser(description="KVTC Perplexity Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()

    print(f"  {'='*70}")
    print(f"  KVTC Perplexity Benchmark")
    print(f"  {'='*70}")

    model, tok, actual = load_model(args.model, args.device)
    
    # Get test samples
    samples = get_wikitext_samples(tok, max_length=512, n_samples=args.samples)
    
    # Baseline perplexity
    baseline_ppl = compute_perplexity_original(model, samples, args.device)
    
    # Load or compute calibration
    cp = Path(__file__).parent / f"calibration_v4_{actual.replace('/','_')}.pt"
    if cp.exists():
        print(f"  Loading calibration from {cp}")
        calib = torch.load(cp, weights_only=False)
    else:
        # Quick calibration
        from benchmark_v4 import calibrate
        calib = calibrate(model, tok, 30, args.device)
        torch.save(calib, cp)
    
    # Test configs
    configs = [
        ("K1V3", 1, 3, False),
        ("K1V3-adaptive", 1, 3, True),
        ("K1V4-adaptive", 1, 4, True),
        ("K2V4", 2, 4, False),
        ("K2V4-adaptive", 2, 4, True),
        ("K3V4-adaptive", 3, 4, True),
        ("K4V4-adaptive", 4, 4, True),
        ("K4V6-adaptive", 4, 6, True),
    ]
    
    results = [{"name": "baseline (f16)", "ppl": baseline_ppl, "ppl_delta": 0, "ppl_pct": 0}]
    
    for name, kb, vb, adaptive in configs:
        print(f"\n  [{name}] K={kb}b V={vb}b {'adaptive' if adaptive else 'uniform'}:")
        ppl = compute_perplexity_kvtc(model, tok, samples, calib, kb, vb, adaptive, args.device)
        delta = ppl - baseline_ppl
        pct = (delta / baseline_ppl) * 100
        
        results.append({
            "name": name, "ppl": ppl, "ppl_delta": delta, "ppl_pct": pct,
            "key_bits": kb, "value_bits": vb, "adaptive": adaptive,
        })
        print(f"  PPL: {ppl:.2f} (delta: {delta:+.2f}, {pct:+.1f}%)")
    
    # Save results
    jp = Path(__file__).parent / "perplexity_results.json"
    with open(jp, "w", encoding="utf-8") as f:
        json.dump({"model": actual, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": results}, f, indent=2)
    
    # Print summary
    print(f"\n  {'='*70}")
    print(f"  PERPLEXITY RESULTS -- {actual}")
    print(f"  {'='*70}")
    print(f"  {'Config':<25s} | {'PPL':>8s} | {'Delta':>8s} | {'% Change':>8s}")
    print(f"  {'-'*25:s}-+-{'-'*8:s}-+-{'-'*8:s}-+-{'-'*8:s}")
    for r in results:
        print(f"  {r['name']:<25s} | {r['ppl']:8.2f} | {r.get('ppl_delta',0):+8.2f} | {r.get('ppl_pct',0):+7.1f}%")
    
    print(f"\n  Results saved to {jp}")


if __name__ == "__main__":
    main()
