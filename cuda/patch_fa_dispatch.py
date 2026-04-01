"""Patch Flash Attention files to support GGML_TYPE_KVTC.

Strategy: Wherever GGML_TYPE_Q8_0 appears in a type-switch or if-chain
for KV cache types, add GGML_TYPE_KVTC as an additional case that uses
the same code path as Q8_0.

This is the stub approach — KVTC uses q8_0 encoding internally, so
FA can treat it identically to q8_0.
"""

import os
import re

CUDA_DIR = r"T:\ik-llama\llama-cpp-turboquant\ggml\src\ggml-cuda"

def patch_file(filepath, description):
    """Add KVTC alongside Q8_0 in type switches."""
    content = open(filepath, "r").read()
    original = content
    changes = 0
    
    # Pattern 1: "case GGML_TYPE_Q8_0:" — add "case GGML_TYPE_KVTC:" before or after
    # But only in FA-related contexts (near other KV cache types)
    
    # Pattern 2: "type == GGML_TYPE_Q8_0" in if conditions — add "|| type == GGML_TYPE_KVTC"
    # But be careful not to double-patch
    
    if "GGML_TYPE_KVTC" in content:
        print(f"  {os.path.basename(filepath)}: already patched, skipping")
        return 0
    
    # Strategy: find all "case GGML_TYPE_Q8_0:" and add KVTC case right after
    pattern = r"(case GGML_TYPE_Q8_0:)"
    replacement = r"case GGML_TYPE_KVTC:\n                \1"
    new_content = re.sub(pattern, replacement, content)
    if new_content != content:
        changes += content.count("case GGML_TYPE_Q8_0:")
        content = new_content
    
    # Also handle "== GGML_TYPE_Q8_0" in if/ternary expressions
    # Add "|| X == GGML_TYPE_KVTC" patterns
    # Be more targeted: only in lines that also mention other KV types
    lines = content.split("\n")
    new_lines = []
    for line in lines:
        if "GGML_TYPE_Q8_0" in line and ("GGML_TYPE_TURBO" in line or "GGML_TYPE_Q4_0" in line or "GGML_TYPE_F16" in line):
            if "GGML_TYPE_KVTC" not in line:
                # Add KVTC alongside Q8_0 in multi-type checks
                line = line.replace("GGML_TYPE_Q8_0", "GGML_TYPE_Q8_0 || op->type == GGML_TYPE_KVTC", 1)
                changes += 1
        new_lines.append(line)
    content = "\n".join(new_lines)
    
    if changes > 0:
        open(filepath, "w").write(content)
        print(f"  {os.path.basename(filepath)}: {changes} patches applied")
    else:
        print(f"  {os.path.basename(filepath)}: no changes needed")
    
    return changes

def patch_fattn_common():
    """Specific patches for fattn-common.cuh template specializations."""
    f = os.path.join(CUDA_DIR, "fattn-common.cuh")
    content = open(f, "r").read()
    
    if "GGML_TYPE_KVTC" in content:
        print("  fattn-common.cuh: already has KVTC")
        return
    
    # The key pattern: template specializations for vec_dot functions
    # Add KVTC as an alias for Q8_0 in the type dispatch
    
    # Find "case GGML_TYPE_Q8_0:" and add KVTC
    content = content.replace(
        "case GGML_TYPE_Q8_0:",
        "case GGML_TYPE_KVTC:\n                case GGML_TYPE_Q8_0:"
    )
    
    open(f, "w").write(content)
    count = content.count("GGML_TYPE_KVTC")
    print(f"  fattn-common.cuh: added {count} KVTC references")

def patch_fattn_vec():
    """Specific patches for fattn-vec.cuh."""
    f = os.path.join(CUDA_DIR, "fattn-vec.cuh")
    content = open(f, "r").read()
    
    if "GGML_TYPE_KVTC" in content:
        print("  fattn-vec.cuh: already has KVTC")
        return
    
    content = content.replace(
        "case GGML_TYPE_Q8_0:",
        "case GGML_TYPE_KVTC:\n                case GGML_TYPE_Q8_0:"
    )
    
    open(f, "w").write(content)
    count = content.count("GGML_TYPE_KVTC")
    print(f"  fattn-vec.cuh: added {count} KVTC references")

def patch_fattn_cu():
    """Specific patches for fattn.cu."""
    f = os.path.join(CUDA_DIR, "fattn.cu")
    content = open(f, "r").read()
    
    if "GGML_TYPE_KVTC" in content:
        print("  fattn.cu: already has KVTC")
        return
    
    content = content.replace(
        "case GGML_TYPE_Q8_0:",
        "case GGML_TYPE_KVTC:\n            case GGML_TYPE_Q8_0:"
    )
    
    open(f, "w").write(content)
    count = content.count("GGML_TYPE_KVTC")
    print(f"  fattn.cu: added {count} KVTC references")

def patch_convert():
    """Patch convert.cu for KVTC type conversion."""
    f = os.path.join(CUDA_DIR, "convert.cu")
    content = open(f, "r").read()
    
    if "GGML_TYPE_KVTC" in content:
        print("  convert.cu: already has KVTC")
        return
    
    content = content.replace(
        "case GGML_TYPE_Q8_0:",
        "case GGML_TYPE_KVTC:\n            case GGML_TYPE_Q8_0:"
    )
    
    open(f, "w").write(content)
    count = content.count("GGML_TYPE_KVTC")
    print(f"  convert.cu: added {count} KVTC references")

def patch_dequantize():
    """Patch dequantize.cuh."""
    f = os.path.join(CUDA_DIR, "dequantize.cuh")
    content = open(f, "r").read()
    
    if "GGML_TYPE_KVTC" in content:
        print("  dequantize.cuh: already has KVTC")
        return
    
    content = content.replace(
        "case GGML_TYPE_Q8_0:",
        "case GGML_TYPE_KVTC:\n        case GGML_TYPE_Q8_0:"
    )
    
    open(f, "w").write(content)
    count = content.count("GGML_TYPE_KVTC")
    print(f"  dequantize.cuh: added {count} KVTC references")

def patch_ggml_cuda_supports():
    """Patch ggml-cuda.cu supports_op for additional ops."""
    f = os.path.join(CUDA_DIR, "ggml-cuda.cu")
    content = open(f, "r").read()
    
    # Check CPY op — might need KVTC support
    if "GGML_TYPE_KVTC" not in content or content.count("GGML_TYPE_KVTC") < 2:
        # Add KVTC to CPY support (F32 <-> KVTC)
        old = 'if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {'
        new = 'if (src0_type == GGML_TYPE_F32 && (src1_type == GGML_TYPE_Q8_0 || src1_type == GGML_TYPE_KVTC)) {'
        content = content.replace(old, new)
        
        old2 = 'if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {'
        new2 = 'if ((src0_type == GGML_TYPE_Q8_0 || src0_type == GGML_TYPE_KVTC) && src1_type == GGML_TYPE_F32) {'
        content = content.replace(old2, new2)
        
        open(f, "w").write(content)
        print(f"  ggml-cuda.cu: added KVTC to CPY support")
    else:
        print(f"  ggml-cuda.cu: KVTC already present")

def main():
    print("Patching Flash Attention for KVTC support")
    print("=" * 50)
    
    patch_fattn_common()
    patch_fattn_vec()
    patch_fattn_cu()
    patch_convert()
    patch_dequantize()
    patch_ggml_cuda_supports()
    
    print("\nDone! Rebuild with: cmake --build . --config Release")

if __name__ == "__main__":
    main()
