"""Patch set-rows.cu to add KVTC support."""
import sys

f = r"T:\ik-llama\llama-cpp-turboquant\ggml\src\ggml-cuda\set-rows.cu"
content = open(f, "r").read()

target = 'GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));'
idx = content.find(target)
print(f"Found ABORT at index {idx}")

if idx < 0:
    print("ERROR: target not found")
    sys.exit(1)

# Find the '} else {' before it
else_idx = content.rfind("} else {", 0, idx)
print(f"Found else at index {else_idx}")

kvtc_case = """} else if (dst->type == GGML_TYPE_KVTC) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    """

content = content[:else_idx] + kvtc_case + content[else_idx:]
open(f, "w").write(content)
print("Patched set-rows.cu!")

# Verify
count = content.count("GGML_TYPE_KVTC")
print(f"KVTC references in file: {count}")
