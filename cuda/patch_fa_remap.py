"""Remap GGML_TYPE_KVTC to Q8_0 at the FA entry points.

Instead of adding KVTC cases everywhere, we intercept at the top of each
FA function and remap the K/V tensor types from KVTC to Q8_0. This way
all existing Q8_0 template instantiations work automatically.
"""

import os

FATTN = r"T:\ik-llama\llama-cpp-turboquant\ggml\src\ggml-cuda\fattn.cu"

content = open(FATTN, "r").read()

if "KVTC_REMAP" in content:
    print("Already patched")
    exit(0)

# The remap code to inject at the top of each FA function
REMAP_CODE = """
    // KVTC_REMAP: treat KVTC as Q8_0 for flash attention (stub mode)
    if (K->type == GGML_TYPE_KVTC) { const_cast<ggml_tensor*>(K)->type = GGML_TYPE_Q8_0; }
    if (V->type == GGML_TYPE_KVTC) { const_cast<ggml_tensor*>(V)->type = GGML_TYPE_Q8_0; }
"""

RESTORE_CODE = """
    // KVTC_REMAP: restore original types (in case tensors are reused)
    // (handled by the caller if needed)
"""

# Inject into ggml_cuda_flash_attn_ext_vec (the main vec dispatch)
old = """static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];"""

new = """static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];
""" + REMAP_CODE

content = content.replace(old, new)

# Also inject into the MMA dispatch
old2 = """static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {"""
if old2 in content:
    # Find the Q/K/V variable declarations after this
    idx = content.find(old2)
    # Find the next blank line or the first FATTN call after src declarations
    after = content.find("V = dst->src[2];", idx)
    if after > 0:
        insert_point = content.find("\n", after) + 1
        content = content[:insert_point] + REMAP_CODE + content[insert_point:]
        print("Patched mma_f16")

# Also inject into the tile dispatch if it exists
old3 = "static void ggml_cuda_flash_attn_ext_tile(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {"
if old3 in content:
    idx = content.find(old3)
    after = content.find("V = dst->src[2];", idx)
    if after > 0:
        insert_point = content.find("\n", after) + 1
        content = content[:insert_point] + REMAP_CODE + content[insert_point:]
        print("Patched tile")

# Also need to patch ggml_cuda_flash_attn_ext_supported to accept KVTC
old4 = "return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0 || t == GGML_TYPE_Q8_0;"
new4 = "return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0 || t == GGML_TYPE_Q8_0 || t == GGML_TYPE_KVTC;"
content = content.replace(old4, new4)

# Patch the supports check
old5 = """bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}"""
new5 = """bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    // KVTC_REMAP: temporarily remap KVTC->Q8_0 for support check
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];
    ggml_type orig_k = K->type, orig_v = V->type;
    if (K->type == GGML_TYPE_KVTC) const_cast<ggml_tensor*>(K)->type = GGML_TYPE_Q8_0;
    if (V->type == GGML_TYPE_KVTC) const_cast<ggml_tensor*>(V)->type = GGML_TYPE_Q8_0;
    bool result = ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
    const_cast<ggml_tensor*>(K)->type = orig_k;
    const_cast<ggml_tensor*>(V)->type = orig_v;
    return result;
}"""
content = content.replace(old5, new5)

open(FATTN, "w").write(content)
count = content.count("KVTC")
print(f"Patched fattn.cu: {count} KVTC references")
print("Done!")
