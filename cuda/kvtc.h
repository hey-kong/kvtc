/*
 * KVTC — KV-Cache Tensor Compression CUDA Kernels
 * 
 * Header for the CUDA encode/decode kernels that implement:
 * 1. PCA transform (matmul with pre-computed eigenvectors)
 * 2. Adaptive quantization (per-component bit widths)
 * 3. Bit packing into compressed byte stream
 * 
 * These kernels are designed to be called from llama.cpp's KV cache
 * write/read paths, replacing the FP16 KV storage with compressed
 * KVTC format.
 * 
 * Copyright (c) 2026 Terp AI Labs / @OnlyTerp
 * License: MIT
 */

#ifndef KVTC_H
#define KVTC_H

#include <cuda_fp16.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Calibration data (per layer, per head group, per kind) ──────── */

typedef struct {
    float *eigenvectors;     /* [dim, dim] PCA basis (row-major) */
    float *eigenvalues;      /* [dim] eigenvalue magnitudes */
    float *mean;             /* [dim] PCA mean vector */
    int    dim;              /* head dimension (e.g. 128) */
    int    bit_budget;       /* total bits for this group */
    float  rope_theta;       /* RoPE base frequency */
} kvtc_calibration_t;

/* ─── Compressed block metadata ──────────────────────────────────── */

typedef struct {
    uint8_t *data;           /* compressed byte stream */
    int      data_bytes;     /* size of compressed data */
    int      num_rows;       /* number of KV vectors compressed */
    int      dim;            /* head dimension */
    float   *scales;         /* [dim] quantization scales */
    float   *zero_points;    /* [dim] quantization zero points */
    int8_t  *bit_widths;     /* [dim] bits per component (0-16) */
} kvtc_compressed_t;

/* ─── Context (holds all calibration data for one model) ─────────── */

typedef struct {
    kvtc_calibration_t *entries;  /* array of calibration entries */
    int                 n_entries;
    int                 n_layers;
    int                 n_heads;
    int                 head_dim;
    int                 head_group_size;
    int                 sink_tokens;     /* default: 4 */
    int                 window_tokens;   /* default: 128 */
} kvtc_ctx_t;

/* ─── API ────────────────────────────────────────────────────────── */

/**
 * Initialize KVTC context from a calibration file.
 * Returns NULL on failure.
 */
kvtc_ctx_t *kvtc_init(const char *calibration_path);

/**
 * Free KVTC context and all associated GPU memory.
 */
void kvtc_free(kvtc_ctx_t *ctx);

/**
 * Encode (compress) a block of KV cache vectors.
 * 
 * @param ctx       KVTC context with calibration data
 * @param src       Source FP16 KV vectors [num_rows, dim] (device memory)
 * @param num_rows  Number of KV vectors to compress
 * @param layer_idx Layer index (for calibration lookup)
 * @param group_idx Head group index
 * @param is_key    1 for keys (applies RoPE inverse), 0 for values
 * @param positions Position indices [num_rows] (device memory, for RoPE)
 * @param dst       Output compressed block (allocated by caller)
 * @param stream    CUDA stream for async execution
 * 
 * @return 0 on success, nonzero on error
 */
int kvtc_encode(
    const kvtc_ctx_t     *ctx,
    const half           *src,
    int                   num_rows,
    int                   layer_idx,
    int                   group_idx,
    int                   is_key,
    const int            *positions,
    kvtc_compressed_t    *dst,
    void                 *stream
);

/**
 * Decode (decompress) a block back to FP16 KV vectors.
 * 
 * @param ctx       KVTC context
 * @param src       Compressed block
 * @param layer_idx Layer index
 * @param group_idx Head group index
 * @param is_key    1 for keys (reapplies RoPE), 0 for values
 * @param positions Position indices [num_rows] (device memory)
 * @param dst       Output FP16 vectors [num_rows, dim] (device memory)
 * @param stream    CUDA stream
 * 
 * @return 0 on success, nonzero on error
 */
int kvtc_decode(
    const kvtc_ctx_t          *ctx,
    const kvtc_compressed_t   *src,
    int                        layer_idx,
    int                        group_idx,
    int                        is_key,
    const int                 *positions,
    half                      *dst,
    void                      *stream
);

/**
 * Allocate a compressed block on the GPU.
 * Estimates maximum compressed size from num_rows and dim.
 */
kvtc_compressed_t *kvtc_alloc_compressed(int num_rows, int dim);

/**
 * Free a compressed block.
 */
void kvtc_free_compressed(kvtc_compressed_t *block);

/* ─── Individual kernel launchers (for testing/benchmarking) ─────── */

/**
 * PCA transform: centered = (src - mean), pca = centered @ Vh^T
 * Fused into a single kernel with the mean subtraction.
 */
void kvtc_pca_transform(
    const float *src,        /* [rows, dim] */
    const float *eigenvectors, /* [dim, dim] */
    const float *mean,       /* [dim] */
    float       *dst,        /* [rows, dim] output PCA values */
    int          rows,
    int          dim,
    void        *stream
);

/**
 * PCA inverse: restored = pca @ Vh + mean
 */
void kvtc_pca_inverse(
    const float *src,        /* [rows, dim] PCA values */
    const float *eigenvectors, /* [dim, dim] */
    const float *mean,       /* [dim] */
    float       *dst,        /* [rows, dim] output */
    int          rows,
    int          dim,
    void        *stream
);

/**
 * Adaptive quantize: per-component quantization with variable bit widths.
 * Computes scales/zero_points from data, then quantizes in one pass.
 */
void kvtc_quantize(
    const float  *pca_values,  /* [rows, dim] */
    const int8_t *bit_widths,  /* [dim] */
    float        *scales,      /* [dim] output */
    float        *zero_points, /* [dim] output */
    int32_t      *indices,     /* [rows, dim] output quantized indices */
    int           rows,
    int           dim,
    void         *stream
);

/**
 * Adaptive dequantize: reverse of kvtc_quantize.
 */
void kvtc_dequantize(
    const int32_t *indices,     /* [rows, dim] */
    const int8_t  *bit_widths,  /* [dim] */
    const float   *scales,      /* [dim] */
    const float   *zero_points, /* [dim] */
    float         *dst,         /* [rows, dim] output */
    int            rows,
    int            dim,
    void          *stream
);

/**
 * Bit packing: pack variable-width quantized indices into byte stream.
 */
void kvtc_pack_bits(
    const int32_t *indices,    /* [rows, dim] */
    const int8_t  *bit_widths, /* [dim] */
    uint8_t       *dst,        /* output byte stream */
    int           *dst_bytes,  /* output: actual bytes written */
    int            rows,
    int            dim,
    void          *stream
);

/**
 * Bit unpacking: reverse of kvtc_pack_bits.
 */
void kvtc_unpack_bits(
    const uint8_t *src,        /* packed byte stream */
    int            src_bytes,
    const int8_t  *bit_widths, /* [dim] */
    int32_t       *dst,        /* [rows, dim] output */
    int            rows,
    int            dim,
    void          *stream
);

/**
 * RoPE inverse: undo rotary position embeddings on key vectors.
 */
void kvtc_rope_inverse(
    const float *src,       /* [rows, dim] keys with RoPE */
    const int   *positions, /* [rows] position indices */
    float       *dst,       /* [rows, dim] keys without RoPE */
    int          rows,
    int          dim,
    float        rope_theta,
    void        *stream
);

/**
 * RoPE forward: reapply rotary position embeddings after decompression.
 */
void kvtc_rope_forward(
    const float *src,       /* [rows, dim] keys without RoPE */
    const int   *positions, /* [rows] position indices */
    float       *dst,       /* [rows, dim] keys with RoPE */
    int          rows,
    int          dim,
    float        rope_theta,
    void        *stream
);

/**
 * Greedy bit allocation: given eigenvalues and budget, compute optimal
 * per-component bit widths.
 */
void kvtc_bit_allocation(
    const float *eigenvalues, /* [dim] */
    int          bit_budget,
    int8_t      *bit_widths,  /* [dim] output */
    int          dim,
    int          max_bits,    /* default: 16 */
    void        *stream
);

#ifdef __cplusplus
}
#endif

#endif /* KVTC_H */
