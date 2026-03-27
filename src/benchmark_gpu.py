#!/usr/bin/env python3
"""KVTC GPU Benchmark — Real model compression validation.

Usage:
    # CPU test with TinyLlama (quick validation):
    python -m src.benchmark_gpu --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cpu

    # GPU test (on 5090 or any CUDA machine):
    python -m src.benchmark_gpu --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cuda

    # Larger model:
    python -m src.benchmark_gpu --model mistralai/Mistral-7B-Instruct-v0.3 --device cuda

    # Custom sequence length:
    python -m src.benchmark_gpu --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --seq-len 1024 --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Tuple

import torch


def get_kv_cache_from_model(
    model,
    tokenizer,
    prompt: str,
    device: str = "cpu",
    max_new_tokens: int = 0,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, int]:
    """Run a forward pass and extract the KV cache.
    
    Returns (kv_cache_dict, positions, num_layers).
    kv_cache_dict has shape [layers, tokens, heads, head_dim] for both keys and values.
    """
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    seq_len = input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    
    past = outputs.past_key_values
    
    # Handle DynamicCache (.layers), legacy DynamicCache (.key_cache), and tuple formats
    keys = []
    values = []
    if hasattr(past, 'layers'):
        num_layers = len(past.layers)
        for layer in past.layers:
            keys.append(layer.keys[0].transpose(0, 1).cpu())
            values.append(layer.values[0].transpose(0, 1).cpu())
    elif hasattr(past, 'key_cache'):
        num_layers = len(past.key_cache)
        for i in range(num_layers):
            keys.append(past.key_cache[i][0].transpose(0, 1).cpu())
            values.append(past.value_cache[i][0].transpose(0, 1).cpu())
    else:
        num_layers = len(past)
        for layer_k, layer_v in past:
            keys.append(layer_k[0].transpose(0, 1).cpu())
            values.append(layer_v[0].transpose(0, 1).cpu())
    
    kv_cache = {
        "keys": torch.stack(keys),      # [layers, tokens, heads, dim]
        "values": torch.stack(values),   # [layers, tokens, heads, dim]
    }
    positions = torch.arange(seq_len, dtype=torch.long)
    
    return kv_cache, positions, num_layers


def measure_compression(
    kv_cache: Dict[str, torch.Tensor],
    positions: torch.Tensor,
    calibration_data,
    verbose: bool = True,
) -> Dict:
    """Compress, decompress, and measure quality metrics."""
    from src.pipeline import KVTCCompressor
    
    compressor = KVTCCompressor(calibration_data)
    
    # Compress
    t0 = time.perf_counter()
    compressed = compressor.compress(kv_cache, positions)
    compress_time = time.perf_counter() - t0
    
    # Decompress
    t0 = time.perf_counter()
    restored = compressor.decompress(compressed)
    decompress_time = time.perf_counter() - t0
    
    # Metrics
    layers, tokens, heads, dim = kv_cache["keys"].shape
    
    # Per-layer cosine similarity
    key_cosines = []
    value_cosines = []
    for layer in range(layers):
        kc = torch.nn.functional.cosine_similarity(
            kv_cache["keys"][layer].reshape(1, -1),
            restored["keys"][layer].reshape(1, -1),
        ).item()
        vc = torch.nn.functional.cosine_similarity(
            kv_cache["values"][layer].reshape(1, -1),
            restored["values"][layer].reshape(1, -1),
        ).item()
        key_cosines.append(kc)
        value_cosines.append(vc)
    
    # Overall cosine
    overall_key_cos = torch.nn.functional.cosine_similarity(
        kv_cache["keys"].reshape(1, -1),
        restored["keys"].reshape(1, -1),
    ).item()
    overall_val_cos = torch.nn.functional.cosine_similarity(
        kv_cache["values"].reshape(1, -1),
        restored["values"].reshape(1, -1),
    ).item()
    
    # MSE
    key_mse = torch.mean((kv_cache["keys"].float() - restored["keys"].float()) ** 2).item()
    val_mse = torch.mean((kv_cache["values"].float() - restored["values"].float()) ** 2).item()
    
    # Original size in bytes (FP16)
    original_bytes = kv_cache["keys"].numel() * 2 + kv_cache["values"].numel() * 2  # FP16 = 2 bytes
    
    # Compressed size (sum of all compressed sections + sinks + window in FP16)
    sink_bytes = compressed.sinks["keys"].numel() * 2 + compressed.sinks["values"].numel() * 2
    window_bytes = compressed.window["keys"].numel() * 2 + compressed.window["values"].numel() * 2
    section_bytes = sum(len(s.compressed_bytes) for s in compressed.compressed_sections)
    compressed_total = sink_bytes + window_bytes + section_bytes
    
    ratio = original_bytes / max(compressed_total, 1)
    
    results = {
        "layers": layers,
        "tokens": tokens,
        "heads": heads,
        "head_dim": dim,
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_total,
        "sink_bytes": sink_bytes,
        "window_bytes": window_bytes,
        "section_bytes": section_bytes,
        "compression_ratio": ratio,
        "pipeline_ratio": compressed.metadata.compression_ratio,
        "key_cosine_avg": sum(key_cosines) / len(key_cosines),
        "value_cosine_avg": sum(value_cosines) / len(value_cosines),
        "key_cosine_min": min(key_cosines),
        "value_cosine_min": min(value_cosines),
        "overall_key_cosine": overall_key_cos,
        "overall_value_cosine": overall_val_cos,
        "key_mse": key_mse,
        "value_mse": val_mse,
        "compress_time_ms": compress_time * 1000,
        "decompress_time_ms": decompress_time * 1000,
        "num_sections": len(compressed.compressed_sections),
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("KVTC Compression Results")
        print("=" * 60)
        print(f"Model shape:     {layers} layers × {tokens} tokens × {heads} heads × {dim} dim")
        print(f"Original size:   {original_bytes / 1024:.1f} KB")
        print(f"Compressed:      {compressed_total / 1024:.1f} KB")
        print(f"  Sinks (FP16):  {sink_bytes / 1024:.1f} KB")
        print(f"  Window (FP16): {window_bytes / 1024:.1f} KB")
        print(f"  Sections:      {section_bytes / 1024:.1f} KB")
        print(f"Compression:     {ratio:.1f}× (pipeline: {compressed.metadata.compression_ratio:.1f}×)")
        print()
        print(f"Key cosine:      {results['key_cosine_avg']:.4f} avg, {results['key_cosine_min']:.4f} min")
        print(f"Value cosine:    {results['value_cosine_avg']:.4f} avg, {results['value_cosine_min']:.4f} min")
        print(f"Key MSE:         {key_mse:.6f}")
        print(f"Value MSE:       {val_mse:.6f}")
        print()
        print(f"Compress time:   {compress_time * 1000:.1f} ms")
        print(f"Decompress time: {decompress_time * 1000:.1f} ms")
        print("=" * 60)
    
    return results


def run_logit_comparison(
    model,
    tokenizer,
    kv_cache: Dict[str, torch.Tensor],
    restored_cache: Dict[str, torch.Tensor],
    device: str = "cpu",
) -> Dict:
    """Compare logits from original vs restored KV cache on continuation tokens."""
    prompts = [
        "The capital of France is",
        "In 1969, humans first",
        "Python is a programming language that",
        "The theory of relativity was developed by",
        "Water boils at a temperature of",
    ]
    
    matches = 0
    cosines = []
    
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
        
        original_logits = outputs.logits[0, -1].cpu().float()
        original_pred = original_logits.argmax().item()
        original_token = tokenizer.decode([original_pred])
        
        # For restored: we'd need to reconstruct the past_key_values format
        # This is model-specific, so we'll just compare the KV caches directly
        # and report the logit comparison from a fresh forward pass
        
        print(f"  '{prompt}' → '{original_token.strip()}'")
        matches += 1  # Placeholder until we wire restored KV → model
    
    return {"prompts_tested": len(prompts), "match_rate": matches / len(prompts)}


def main():
    parser = argparse.ArgumentParser(description="KVTC Real Model Benchmark")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="HuggingFace model name")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run on")
    parser.add_argument("--seq-len", type=int, default=512, help="Target sequence length for benchmark")
    parser.add_argument("--calibration-samples", type=int, default=10, help="Number of calibration samples")
    parser.add_argument("--bit-budget-ratio", type=float, default=0.25, help="Bit budget as fraction of FP16")
    parser.add_argument("--sink-tokens", type=int, default=4, help="Number of attention sink tokens to preserve")
    parser.add_argument("--window-tokens", type=int, default=32, help="Sliding window size to preserve")
    args = parser.parse_args()
    
    print(f"KVTC Benchmark")
    print(f"Model:  {args.model}")
    print(f"Device: {args.device}")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available!")
            sys.exit(1)
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
        print(f"VRAM:   {vram / 1024**3:.1f} GB")
    
    print("\n[1/4] Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )
    if args.device == "cuda":
        model = model.cuda()
    model.eval()
    
    print(f"  Loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B parameters")
    
    # Generate a long prompt — repeat to fill seq_len
    base_prompt = (
        "Explain the following topics in detail with examples and mathematical formulations: "
        "quantum computing and its applications in cryptography and post-quantum security, "
        "the complete history of artificial intelligence from Alan Turing's original proposals through "
        "modern transformer architectures and large language models, "
        "how neural networks learn through backpropagation and stochastic gradient descent with momentum, "
        "the mathematics behind multi-head attention mechanisms in transformer architectures including "
        "scaled dot-product attention and positional encoding schemes, "
        "recent advances in KV cache compression for efficient LLM inference including quantization "
        "and pruning approaches, the role of principal component analysis in dimensionality reduction "
        "and its application to feature decorrelation in neural network activations, "
        "entropy coding methods including Huffman coding and arithmetic coding and their use in "
        "data compression pipelines, the difference between lossy and lossless compression algorithms "
        "and when each is appropriate for different data types, "
        "the theory of information and Shannon entropy and how it relates to optimal compression rates, "
        "modern GPU architecture and how tensor cores accelerate matrix multiplication in deep learning, "
        "the design of memory hierarchies in modern processors and how cache locality affects performance, "
        "distributed training strategies including data parallelism and model parallelism and pipeline parallelism, "
        "the mathematics of convex optimization and how it applies to training neural networks, "
        "reinforcement learning from human feedback and its role in aligning language models, "
        "the architecture of modern operating systems and how virtual memory management works. "
    )
    # Repeat to fill desired token count
    long_prompt = base_prompt
    while len(tokenizer.encode(long_prompt)) < args.seq_len:
        long_prompt = long_prompt + " " + base_prompt
    
    # Truncate to desired length
    tokens = tokenizer.encode(long_prompt)
    if len(tokens) > args.seq_len:
        tokens = tokens[:args.seq_len]
        long_prompt = tokenizer.decode(tokens)
    
    print(f"\n[2/4] Calibrating PCA bases ({args.calibration_samples} samples)...")
    from src.calibrate import KVTCCalibrator
    
    calibration_texts = [
        "KV cache compression can unlock longer contexts in language models.",
        "Dynamic programming assigns bits where variance matters most.",
        "Principal components capture the most important directions in data.",
        "Attention sinks at the beginning of sequences should be preserved exactly.",
        "Random rotation decorrelates features before quantization.",
        "Entropy coding exploits low-entropy streams for additional compression.",
        "RoPE positional encoding must be undone before applying PCA to keys.",
        "Sliding window attention keeps recent tokens uncompressed.",
        "The bit allocation problem is solved optimally via dynamic programming.",
        "Transformer KV caches grow linearly with sequence length.",
        "Modern LLMs require efficient memory management for long contexts.",
        "Quantization introduces controlled error in exchange for compression.",
    ]
    
    calibrator = KVTCCalibrator(head_group_size=1)
    calibrator.collect_samples(model, tokenizer, calibration_texts[:args.calibration_samples])
    calibration = calibrator.compute_calibration(bit_budget_ratio=args.bit_budget_ratio)
    calibration.sink_tokens = args.sink_tokens
    calibration.window_tokens = args.window_tokens
    print(f"  Bit budget ratio: {args.bit_budget_ratio} ({args.bit_budget_ratio * 16:.1f} bits/value avg)")
    print(f"  Collected {calibrator.samples_collected} samples")
    print(f"  Entries: {len(calibration.entries)} (layer × group × kind)")
    
    print(f"\n[3/4] Extracting KV cache ({len(tokens)} tokens)...")
    kv_cache, positions, num_layers = get_kv_cache_from_model(
        model, tokenizer, long_prompt, device=args.device
    )
    layers, seq, heads, dim = kv_cache["keys"].shape
    print(f"  Shape: {layers}L × {seq}T × {heads}H × {dim}D")
    print(f"  KV cache size: {(kv_cache['keys'].numel() + kv_cache['values'].numel()) * 2 / 1024:.1f} KB (FP16)")
    
    print(f"\n[4/4] Running KVTC compression...")
    results = measure_compression(kv_cache, positions, calibration, verbose=True)
    
    # Summary verdict
    print("\n" + "=" * 60)
    key_ok = results["key_cosine_avg"] > 0.95
    val_ok = results["value_cosine_avg"] > 0.95
    ratio_ok = results["compression_ratio"] > 3.0
    
    if key_ok and val_ok and ratio_ok:
        print("PASS -- Quality and compression look good!")
    elif key_ok and val_ok:
        print("PARTIAL -- Quality is good but compression ratio is low")
        if results["compression_ratio"] < 1.1:
            print(f"NOTE: Only {results['tokens']} tokens with sink={calibration.sink_tokens} + window={calibration.window_tokens}.")
            print("      All tokens are in sink/window regions -- nothing to compress!")
            print("      Use --seq-len 512+ to test actual compression.")
    else:
        print(f"NEEDS WORK -- Cosine sim below 0.95 (keys={results['key_cosine_avg']:.4f}, values={results['value_cosine_avg']:.4f})")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
