# Research Notes — Other TurboQuant Implementations (Mar 26, 2026)

## Key Finding: QJL Residual Is Unnecessary
Multiple independent implementations confirmed the paper's QJL residual stage adds complexity without meaningful quality improvement. Skip it.

## TheTom/turboquant_plus (llama.cpp integration)
- TurboQuant running in llama.cpp on Apple Silicon
- 4.6× KV compression, 102% of q8_0 speed (FASTER because less memory bandwidth)
- PPL within 1.3% of baseline
- Graph-side WHT rotation was the key speedup (3.72× over naive fp32 rotation)
- "coherent text output means nothing. always run perplexity" — use quantitative metrics

## dhawalc/turboQuantDC (Python, real model validation)
- Qwen2.5-3B: 0.9959 cosine sim, 91.7% top-5 match
- Qwen2.5-14B: 0.9964 cosine sim, 95.3% top-5 match  
- Qwen3.5-27B: 0.9932 cosine sim, 100% top-5 match
- "The bigger the model, the better it works" — more redundancy in larger KV caches
- "the scaling trend is the key finding here, bigger models have more redundancy in the KV cache so the rotation maps to a tighter distribution"

## 0xSero/turboquant (vLLM integration, our reference for the backend)
- Working vLLM monkey-patch with Triton decode kernels
- Tested on Qwen3.5-27B with 4× RTX 3090, 2× context capacity
- 30GB KV freed across 4 GPUs after prefill
- 6 iterations to get real VRAM savings (not just theoretical compression)

## Implications for KVTC vLLM Integration
1. Skip QJL/residual — simpler is better for the decode kernel
2. Larger models = better compression quality (great news for TerpBot Pro 30B)
3. The decode kernel should fuse dequant + PCA inverse + attention (like 0xSero's 3-kernel approach)
4. ALWAYS measure with perplexity, not "looks coherent"
5. The VRAM savings come from freeing the paged cache AFTER prefill — this is the critical step
