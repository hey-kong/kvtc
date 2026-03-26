# KVTC — KV Cache Transform Coding

[![Tests](https://img.shields.io/github/actions/workflow/status/OnlyTerp/kvtc/test.yml?label=tests)](https://github.com/OnlyTerp/kvtc/actions/workflows/test.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2511.01815-b31b1b.svg)](https://arxiv.org/abs/2511.01815)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OnlyTerp/kvtc/blob/master/notebooks/demo.ipynb)

**Compress your LLM's KV cache by 8–32× with near-zero accuracy loss.** PCA decorrelation + adaptive quantization + entropy coding.

> First open-source implementation of [NVIDIA's KVTC](https://arxiv.org/abs/2511.01815) (ICLR 2026). No model changes required — compress, store, decompress, resume.

## Why KVTC?

KV caches grow linearly with context length and can consume multiple gigabytes for long conversations. KVTC compresses them for compact storage using the same ideas behind JPEG: decorrelate, quantize, entropy-code.

**Key insight:** KV cache vectors have strong low-rank structure. PCA exposes this structure, then a DP algorithm allocates bits only where they matter. The rest gets pruned to zero — free dimensionality reduction.

### Paper Results (NVIDIA, ICLR 2026)

| Method | Compression | Accuracy Retention | Approach |
|--------|:-----------:|:------------------:|----------|
| **KVTC** | **20×** | **< 1% loss** | PCA + DP quantization + DEFLATE |
| KIVI | 2.6× | Moderate | 2-bit asymmetric quantization |
| GEAR | 4× | Good | Low-rank + quantization |
| H2O | 4–8× | Task-dependent | Token eviction |
| xKV | 8–16× | Strong | SVD-based compression |

### Our Results (Mistral-7B-Instruct-v0.3, 4-bit quantized)

| Mode | Key Cosine | Value Cosine | Compression | Use Case |
|------|:----------:|:------------:|:-----------:|----------|
| High Quality | **0.9996** | **0.9982** | **7.9×** | Production serving |
| Balanced | **0.9950** | **0.9818** | **16.2×** | Memory-constrained |
| Aggressive | **0.9895** | **0.9654** | **31.7×** | Maximum compression |

Tested on 32 layers × 8 KV heads × head_dim=128. Attention sinks (first 4 tokens) and sliding window (last 128 tokens) preserved exactly — only middle tokens are compressed.

## Architecture

```
Input: KV Cache [layers, tokens, heads, dim]
         │
         ├── Sinks (first 4 tokens) ──────────────── stored exactly
         ├── Window (last 128 tokens) ────────────── stored exactly
         │
         └── Middle tokens
              │
              ▼
    ┌─────────────────┐
    │  Undo RoPE      │  (keys only — exposes low-rank structure)
    │  on keys         │
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │  PCA Transform   │  Calibrated offline, one-time per model
    │  decorrelate     │  V^T · (x - μ) → principal components
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │  DP Quantize     │  Optimal bit allocation per component
    │  0–16 bits/comp  │  0 bits = component pruned entirely
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │  Bit Pack +      │  Variable-width packing → zlib DEFLATE
    │  DEFLATE         │  Lossless entropy coding
    └────────┬────────┘
             ▼
    Output: CompressedKVCache (bytes + metadata)
```

Decompression reverses the pipeline: DEFLATE → unpack → dequantize → PCA inverse → reapply RoPE → concatenate with sinks and window.

## Quick Start

```bash
git clone https://github.com/hey-kong/kvtc.git
cd kvtc
uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .
(uv pip install pytest)
pytest src/test_kvtc.py  # 40 tests
```

```python
import torch
from src.pca import PCACalibrator
from src.pipeline import KVTCCompressor

# Simulate KV cache: [layers, tokens, heads, dim]
kv_cache = {
    "keys": torch.randn(2, 256, 4, 64),
    "values": torch.randn(2, 256, 4, 64),
}
positions = torch.arange(256)

# Step 1: Calibrate PCA (one-time, per model)
calibrator = PCACalibrator(head_group_size=1)
for layer_idx in range(2):
    calibrator.collect(layer_idx, "keys", kv_cache["keys"][layer_idx], positions)
    calibrator.collect(layer_idx, "values", kv_cache["values"][layer_idx])
calibration = calibrator.compute(bit_budget_ratio=0.12)  # 12% = ~16x compression

# Step 2: Compress
compressor = KVTCCompressor(calibration)
compressed = compressor.compress(kv_cache, positions)
print(f"Compression ratio: {compressed.metadata.compression_ratio:.1f}x")

# Step 3A: Full decompress (baseline)
restored_full = compressor.decompress(compressed)

# Step 3B: Layer-by-layer decompress + check
num_layers = kv_cache["keys"].shape[0]
for layer_idx in range(num_layers):
    restored_layer = compressor.decompress_layer(compressed, layer_idx)

    # shape check: [tokens, heads, dim]
    assert restored_layer["keys"].shape == kv_cache["keys"][layer_idx].shape
    assert restored_layer["values"].shape == kv_cache["values"][layer_idx].shape

    # consistency check vs full decompress
    assert torch.allclose(restored_layer["keys"], restored_full["keys"][layer_idx], atol=1e-6)
    assert torch.allclose(restored_layer["values"], restored_full["values"][layer_idx], atol=1e-6)

print("decompress_layer test passed.")
```

### Compression Modes

| `bit_budget_ratio` | Compression | Quality | When to use |
|:-------------------:|:-----------:|:-------:|-------------|
| `0.25` | ~8× | Excellent (0.999+ cosine) | Production, quality-critical |
| `0.12` | ~16× | Very good (0.995 cosine) | Balanced memory/quality |
| `0.06` | ~32× | Good (0.989 cosine) | Maximum savings, long context |

## Limitations

- **Reference implementation** — Pure PyTorch on CPU. Not optimized for production throughput.
- **Entropy coding is CPU-only** — Uses `zlib` DEFLATE. The paper uses NVIDIA's `nvCOMP` for GPU-accelerated DEFLATE.
- **DP quantization is O(d × B × 16)** — Fast enough for reference use, but production would need optimized kernels.
- **Compression is slow (~30-60s per prompt on CPU)** — The DP + PCA transform dominates. GPU acceleration would fix this.
- **Tested on 2 models** — TinyLlama-1.1B and Mistral-7B. More model validation welcome.
- **Not affiliated with NVIDIA** — Independent implementation from the public paper.

## Algorithm Details

### Stage 1: PCA Feature Decorrelation

- Collect KV cache samples from a calibration dataset (10 texts, ~2 seconds)
- **Undo RoPE on keys** before computing PCA — RoPE rotation hides low-rank structure
- Compute SVD per `(layer, head, key/value)` to get eigenvectors and eigenvalues
- At compression time: project vectors into PCA space via matrix multiply
- Eigenvalues sorted descending — first components capture most variance

### Stage 2: Adaptive Quantization (Dynamic Programming)

- DP algorithm over eigenvalues and bit budget minimizes total reconstruction error
- Error model: `λᵢ / 4^bᵢ` — each bit halves quantization step, reducing MSE by 4×
- Components assigned 0 bits are **pruned entirely** (dimensionality reduction for free)
- Uniform affine quantization within each bit width: `scale = (max - min) / (2^b - 1)`

### Stage 3: Entropy Coding (DEFLATE)

- Pack variable-width quantized indices into compact byte stream
- Apply zlib DEFLATE for lossless compression of statistical redundancy
- Typically adds 1.2–1.5× additional compression beyond quantization alone

### Token Protection

- **Attention sinks** (first 4 tokens): Never compressed. These receive disproportionate attention weight regardless of content.
- **Sliding window** (last 128 tokens): Never compressed. Most relevant context for next-token generation.
- Ablation studies in the paper show compressing these tokens collapses accuracy at high compression ratios.

## Project Structure

```
kvtc/
├── src/
│   ├── __init__.py          # Package exports
│   ├── pca.py               # PCA calibration, RoPE undo/reapply
│   ├── quantize.py          # DP bit allocation, uniform quantization
│   ├── entropy.py           # Bit packing, zlib DEFLATE
│   ├── pipeline.py          # Full KVTCCompressor (compress/decompress)
│   ├── cache.py             # HuggingFace DynamicCache wrapper
│   ├── calibrate.py         # Model calibration utilities
│   ├── common.py            # Shared dataclasses
│   ├── test_kvtc.py         # 38 unit tests
│   └── test_real_model.py   # Optional TinyLlama integration test
├── notebooks/
│   └── demo.ipynb           # Colab notebook
├── deploy/
│   ├── Dockerfile
│   └── run.sh
├── .github/workflows/test.yml
├── IMPLEMENTATION_NOTES.md  # Detailed algorithm documentation
├── CONTRIBUTING.md          # How to contribute
├── BENCHMARKS.md            # Full benchmark results
├── LICENSE                  # MIT
├── README.md
└── setup.py
```

## Citation

```bibtex
@inproceedings{staniszewski2026kvtc,
  title={KV Cache Transform Coding for Compact Storage in LLM Inference},
  author={Staniszewski, Konrad and {\L}a{\'n}cucki, Adrian},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## Credits & Attribution

This is an **independent open-source implementation** of the KVTC algorithm. All credit for the algorithm design and research belongs to the paper authors at NVIDIA.

- **Paper:** [KV Cache Transform Coding for Compact Storage in LLM Inference](https://arxiv.org/abs/2511.01815) — Accepted at **ICLR 2026**
- **Authors:** Konrad Staniszewski, Adrian Łańcucki (NVIDIA)
- **Implementation:** [Terp AI Labs](https://github.com/OnlyTerp)

Not affiliated with or endorsed by NVIDIA. Built from the public paper to make KVTC accessible to the open-source community.

## License

MIT — see [LICENSE](LICENSE) for details.
