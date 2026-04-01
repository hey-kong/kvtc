"""
KVTC llama.cpp Integration Script
Adds GGML_TYPE_KVTC as a stub KV cache type in the TurboQuant fork.
Phase 1: Stub that uses q8_0 internally (proves the plumbing works).
Phase 2: Wire in actual KVTC CUDA kernels.

Run on the 5090:
    cd T:\ik-llama\llama-cpp-turboquant
    py T:\kvtc-bench\integrate_kvtc_llamacpp.py
"""

import os
import sys
import re
import shutil
from pathlib import Path

SRC = Path(r"T:\ik-llama\llama-cpp-turboquant")

def backup_file(path):
    bak = str(path) + ".kvtc-bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"  Backed up {path.name}")

def patch_file(path, patches):
    """Apply a list of (old_text, new_text) patches to a file."""
    backup_file(path)
    content = path.read_text(encoding="utf-8", errors="replace")
    for old, new in patches:
        if old not in content:
            print(f"  WARNING: patch target not found in {path.name}: {old[:60]}...")
            continue
        content = content.replace(old, new, 1)
        print(f"  Patched {path.name}: {old[:50].strip()}...")
    path.write_text(content, encoding="utf-8")

def main():
    print("KVTC llama.cpp Integration — Phase 1 (Stub)")
    print("=" * 60)
    
    # ─── 1. ggml.h: Add GGML_TYPE_KVTC ───────────────────────
    print("\n1. Adding GGML_TYPE_KVTC to ggml.h...")
    patch_file(SRC / "ggml" / "include" / "ggml.h", [
        (
            "GGML_TYPE_TURBO2_0 = 43, // TurboQuant 2-bit KV cache: 2-bit PolarQuant (no QJL)",
            "GGML_TYPE_TURBO2_0 = 43, // TurboQuant 2-bit KV cache: 2-bit PolarQuant (no QJL)\n"
            "        GGML_TYPE_KVTC     = 44, // KVTC: PCA + DP-optimal quantization (Terp AI Labs)"
        ),
    ])
    
    # ─── 2. arg.cpp: Add "kvtc" to allowed cache types ───────
    print("\n2. Adding 'kvtc' to arg.cpp cache types...")
    patch_file(SRC / "common" / "arg.cpp", [
        (
            "GGML_TYPE_TURBO2_0,\n"
            "        GGML_TYPE_TURBO3_0,\n"
            "        GGML_TYPE_TURBO4_0,",
            "GGML_TYPE_TURBO2_0,\n"
            "        GGML_TYPE_TURBO3_0,\n"
            "        GGML_TYPE_TURBO4_0,\n"
            "        GGML_TYPE_KVTC,"
        ),
    ])
    
    # ─── 3. ggml.c: Add KVTC type traits ─────────────────────
    # For the stub, we use q8_0's block size and quantize functions
    # This means KVTC behaves exactly like q8_0 until we wire in real kernels
    print("\n3. Adding KVTC type traits to ggml.c...")
    
    ggml_c = SRC / "ggml" / "src" / "ggml.c"
    backup_file(ggml_c)
    content = ggml_c.read_text(encoding="utf-8", errors="replace")
    
    # Find the TURBO2 type_traits entry and add KVTC after it
    # We need to find the closing brace of the TURBO2 entry
    turbo2_pattern = "[GGML_TYPE_TURBO2_0] = {"
    idx = content.find(turbo2_pattern)
    if idx == -1:
        print("  WARNING: Could not find TURBO2 type_traits entry")
    else:
        # Find the closing }, of this entry
        brace_count = 0
        i = idx
        while i < len(content):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found the closing brace, find the comma after it
                    end = content.index(',', i) + 1
                    break
            i += 1
        
        # Insert KVTC entry after TURBO2
        kvtc_entry = """
    [GGML_TYPE_KVTC] = {
        .type_name                = "kvtc",
        .blck_size                = QK8_0,
        .type_size                = sizeof(block_q8_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q8_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_0_ref,
    },"""
        content = content[:end] + kvtc_entry + content[end:]
        ggml_c.write_text(content, encoding="utf-8")
        print("  Added KVTC type_traits entry (stub using q8_0 functions)")
    
    # Also add KVTC to the quantize switch statement
    content = ggml_c.read_text(encoding="utf-8", errors="replace")
    turbo2_quant = "case GGML_TYPE_TURBO2_0: result = quantize_turbo2_0"
    idx = content.find(turbo2_quant)
    if idx != -1:
        line_end = content.index('\n', idx)
        kvtc_case = "\n            case GGML_TYPE_KVTC:     result = quantize_q8_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;"
        content = content[:line_end] + kvtc_case + content[line_end:]
        ggml_c.write_text(content, encoding="utf-8")
        print("  Added KVTC quantize case (stub using q8_0)")
    
    # ─── 4. llama-kv-cache.cpp: Handle KVTC type ─────────────
    print("\n4. Adding KVTC handling to llama-kv-cache.cpp...")
    patch_file(SRC / "src" / "llama-kv-cache.cpp", [
        (
            "const bool is_turbo = (type_k == GGML_TYPE_TURBO3_0 || type_k == GGML_TYPE_TURBO4_0 || type_k == GGML_TYPE_TURBO2_0);",
            "const bool is_turbo = (type_k == GGML_TYPE_TURBO3_0 || type_k == GGML_TYPE_TURBO4_0 || type_k == GGML_TYPE_TURBO2_0);\n"
            "    const bool is_kvtc = (type_k == GGML_TYPE_KVTC || type_v == GGML_TYPE_KVTC);"
        ),
    ])
    
    # ─── 5. CUDA dispatch: Add KVTC to supported types ───────
    print("\n5. Checking CUDA dispatch tables...")
    cuda_path = SRC / "ggml" / "src" / "ggml-cuda"
    
    # Check if there's a file that lists supported quantization types
    fattn = cuda_path / "fattn-common.cuh"
    if fattn.exists():
        content = fattn.read_text(encoding="utf-8", errors="replace")
        if "GGML_TYPE_TURBO2_0" in content and "GGML_TYPE_KVTC" not in content:
            # Add KVTC alongside TURBO2 in fattn support
            content = content.replace(
                "case GGML_TYPE_TURBO2_0:",
                "case GGML_TYPE_TURBO2_0:\n                case GGML_TYPE_KVTC:"
            )
            backup_file(fattn)
            fattn.write_text(content, encoding="utf-8")
            print("  Added KVTC to fattn dispatch")
        else:
            print("  fattn-common.cuh: already has KVTC or TURBO2 not found")
    
    print("\n" + "=" * 60)
    print("Phase 1 complete!")
    print("KVTC is registered as a stub using q8_0 quantization.")
    print("This means -ctk kvtc -ctv kvtc will work but gives q8_0 compression.")
    print("Phase 2 will replace q8_0 stubs with actual KVTC PCA+quant kernels.")
    print("\nTo build:")
    print("  cd T:\\ik-llama\\llama-cpp-turboquant\\build")
    print("  cmake --build . --config Release")
    print("\nTo test:")
    print("  llama-server.exe -m model.gguf -ctk kvtc -ctv kvtc -fa on -ngl 99 -c 32768")

if __name__ == "__main__":
    main()
