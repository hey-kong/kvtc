"""Unit tests for the KVTC implementation."""

from __future__ import annotations

import pytest
import torch

from src.cache import KVTCCache
from src.calibrate import KVTCCalibrator
from src.entropy import compress, decompress, pack_bits, unpack_bits
from src.pca import PCACalibrator, apply_rope, apply_rope_inverse, pca_inverse, pca_transform
from src.pipeline import KVTCCompressor
from src.quantize import compute_quant_params, dp_bit_allocation, uniform_dequantize, uniform_quantize


def _synthetic_kv(
    layers: int = 2,
    tokens: int = 192,
    heads: int = 4,
    dim: int = 8,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    positions = torch.arange(tokens, dtype=torch.long)
    base = torch.randn(layers, tokens, heads, 1, generator=generator)
    mix = torch.linspace(2.5, 0.1, dim).view(1, 1, 1, dim)
    keys = base * mix + 0.01 * torch.randn(layers, tokens, heads, dim, generator=generator)
    values = 1.2 * base * mix + 0.01 * torch.randn(layers, tokens, heads, dim, generator=generator)
    return {"keys": keys, "values": values}, positions


def _calibration_from_cache(kv_cache: dict[str, torch.Tensor], positions: torch.Tensor):
    calibrator = PCACalibrator(head_group_size=2)
    for layer_idx in range(kv_cache["keys"].shape[0]):
        calibrator.collect(layer_idx, "keys", kv_cache["keys"][layer_idx], positions)
        calibrator.collect(layer_idx, "values", kv_cache["values"][layer_idx])
    return calibrator.compute(bit_budget_ratio=0.12)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()


def test_pca_roundtrip_preserves_variance() -> None:
    data = torch.randn(512, 8)
    mean = data.mean(dim=0)
    centered = data - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    projected = pca_transform(centered, vh.transpose(0, 1))
    restored = pca_inverse(projected, vh.transpose(0, 1))
    total_var = centered.var(dim=0).sum()
    error_var = (centered - restored).var(dim=0).sum()
    preserved = 1.0 - error_var / total_var
    assert preserved > 0.97


def test_pca_with_partial_components_preserves_majority_variance() -> None:
    generator = torch.Generator().manual_seed(0)
    data = torch.randn(1024, 8, generator=generator) @ torch.diag(torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.25, 0.1]))
    mean = data.mean(dim=0)
    centered = data - mean
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    eigenvalues = singular_values.square() / (centered.shape[0] - 1)
    projected = pca_transform(centered, vh.transpose(0, 1))
    projected[:, 4:] = 0
    restored = pca_inverse(projected, vh.transpose(0, 1))
    preserved = 1.0 - ((centered - restored).var(dim=0).sum() / centered.var(dim=0).sum())
    assert preserved > 0.97
    assert torch.all(eigenvalues[:-1] >= eigenvalues[1:])


def test_rope_roundtrip_is_correct() -> None:
    keys = torch.randn(32, 2, 8)
    positions = torch.arange(32)
    rotated = apply_rope(keys, positions, head_dim=8)
    restored = apply_rope_inverse(rotated, positions, head_dim=8)
    assert torch.allclose(restored, keys, atol=1e-5)


def test_rope_inverse_handles_single_token() -> None:
    keys = torch.randn(1, 1, 8)
    positions = torch.tensor([3])
    restored = apply_rope_inverse(apply_rope(keys, positions, head_dim=8), positions, head_dim=8)
    assert torch.allclose(restored, keys, atol=1e-5)


def test_dp_allocates_zero_bits_to_low_variance_components() -> None:
    eigenvalues = torch.tensor([10.0, 8.0, 0.001, 0.0001])
    widths = dp_bit_allocation(eigenvalues, bit_budget=16, max_bits=8)
    assert widths[-1].item() == 0


def test_dp_respects_budget_constraint() -> None:
    eigenvalues = torch.tensor([4.0, 3.0, 2.0, 1.0])
    budget = 10
    widths = dp_bit_allocation(eigenvalues, bit_budget=budget, max_bits=4)
    assert int(widths.sum().item()) <= budget


def test_dp_various_budgets_produce_valid_outputs() -> None:
    eigenvalues = torch.tensor([9.0, 4.0, 1.0, 0.5, 0.1])
    for budget in [0, 4, 8, 20]:
        widths = dp_bit_allocation(eigenvalues, bit_budget=budget, max_bits=8)
        assert widths.shape[0] == eigenvalues.shape[0]
        assert torch.all(widths >= 0)


@pytest.mark.parametrize("bits", [1, 2, 4, 8, 16])
def test_uniform_quantize_dequantize_roundtrip(bits: int) -> None:
    values = torch.linspace(-1.0, 1.0, 33)
    scale = 2.0 / ((1 << bits) - 1)
    zero_point = (1 << (bits - 1)) - 0.5 if bits > 1 else 0.0
    quantized = uniform_quantize(values, bits, scale, zero_point)
    restored = uniform_dequantize(quantized, bits, scale, zero_point)
    assert torch.max(torch.abs(values - restored)) <= scale + 1e-5


def test_zero_bit_quantization_prunes_component() -> None:
    values = torch.tensor([1.0, -2.0, 3.0])
    quantized = uniform_quantize(values, 0, 1.0, 0.0)
    restored = uniform_dequantize(quantized, 0, 1.0, 0.0)
    assert torch.equal(quantized, torch.zeros_like(quantized))
    assert torch.equal(restored, torch.zeros_like(restored))


def test_compute_quant_params_outputs_shapes() -> None:
    pca_values = torch.randn(16, 8)
    bit_widths = torch.tensor([4] * 8)
    params = compute_quant_params(pca_values, bit_widths)
    assert params.scales.shape == (8,)
    assert params.zero_points.shape == (8,)


@pytest.mark.parametrize("bit_widths", [[1], [2, 2], [4, 4], [8], [16]])
def test_bit_packing_roundtrip(bit_widths: list[int]) -> None:
    generator = torch.Generator().manual_seed(sum(bit_widths))
    indices = []
    lengths = []
    for width in bit_widths:
        size = 11
        indices.append(torch.randint(0, 1 << width, (size,), generator=generator, dtype=torch.int64))
        lengths.append(size)
    packed = pack_bits(indices, bit_widths)
    restored = unpack_bits(packed, bit_widths, lengths)
    assert all(torch.equal(a, b) for a, b in zip(indices, restored))


def test_bit_packing_handles_zero_width_segments() -> None:
    indices = [torch.tensor([1, 2, 3]), torch.zeros(3, dtype=torch.int64)]
    packed = pack_bits(indices, [2, 0])
    restored = unpack_bits(packed, [2, 0], [3, 3])
    assert torch.equal(restored[1], torch.zeros(3, dtype=torch.int64))


def test_entropy_roundtrip() -> None:
    payload = b"\x00\x01\x02" * 64
    compressed, ratio = compress(payload)
    restored = decompress(compressed, len(payload))
    assert restored == payload
    assert ratio > 1.0


def test_entropy_empty_payload() -> None:
    compressed, ratio = compress(b"")
    restored = decompress(compressed, 0)
    assert compressed == b""
    assert restored == b""
    assert ratio == 1.0


def test_full_pipeline_roundtrip() -> None:
    kv_cache, positions = _synthetic_kv()
    calibration = _calibration_from_cache(kv_cache, positions)
    compressor = KVTCCompressor(calibration)
    compressed = compressor.compress(kv_cache, positions)
    restored = compressor.decompress(compressed)
    assert restored["keys"].shape == kv_cache["keys"].shape
    assert restored["values"].shape == kv_cache["values"].shape


def test_decompress_layer_matches_full_decompress() -> None:
    kv_cache, positions = _synthetic_kv(layers=3)
    calibration = _calibration_from_cache(kv_cache, positions)
    compressor = KVTCCompressor(calibration)
    compressed = compressor.compress(kv_cache, positions)
    restored_full = compressor.decompress(compressed)
    for layer_idx in range(kv_cache["keys"].shape[0]):
        restored_layer = compressor.decompress_layer(compressed, layer_idx)
        assert torch.allclose(restored_layer["keys"], restored_full["keys"][layer_idx])
        assert torch.allclose(restored_layer["values"], restored_full["values"][layer_idx])


def test_decompress_layer_rejects_invalid_index() -> None:
    kv_cache, positions = _synthetic_kv(layers=2)
    calibration = _calibration_from_cache(kv_cache, positions)
    compressor = KVTCCompressor(calibration)
    compressed = compressor.compress(kv_cache, positions)
    with pytest.raises(ValueError):
        compressor.decompress_layer(compressed, layer_idx=2)


def test_attention_sinks_preserved_exactly() -> None:
    kv_cache, positions = _synthetic_kv(tokens=160)
    calibration = _calibration_from_cache(kv_cache, positions)
    compressor = KVTCCompressor(calibration)
    restored = compressor.decompress(compressor.compress(kv_cache, positions))
    assert torch.equal(restored["keys"][:, :4], kv_cache["keys"][:, :4])
    assert torch.equal(restored["values"][:, :4], kv_cache["values"][:, :4])


def test_sliding_window_preserved_exactly() -> None:
    kv_cache, positions = _synthetic_kv(tokens=300)
    calibration = _calibration_from_cache(kv_cache, positions)
    compressor = KVTCCompressor(calibration)
    restored = compressor.decompress(compressor.compress(kv_cache, positions))
    assert torch.equal(restored["keys"][:, -128:], kv_cache["keys"][:, -128:])
    assert torch.equal(restored["values"][:, -128:], kv_cache["values"][:, -128:])


def test_pipeline_compression_ratio_exceeds_five_on_synthetic_data() -> None:
    kv_cache, positions = _synthetic_kv(dim=8)
    calibration = _calibration_from_cache(kv_cache, positions)
    compressed = KVTCCompressor(calibration).compress(kv_cache, positions)
    assert compressed.metadata is not None
    assert compressed.metadata.compression_ratio > 5.0


def test_pipeline_cosine_similarity_exceeds_threshold() -> None:
    kv_cache, positions = _synthetic_kv()
    calibration = _calibration_from_cache(kv_cache, positions)
    compressor = KVTCCompressor(calibration)
    restored = compressor.decompress(compressor.compress(kv_cache, positions))
    cosine = _cosine_similarity(kv_cache["keys"], restored["keys"])
    assert cosine > 0.95


def test_eigenvalues_descending() -> None:
    kv_cache, positions = _synthetic_kv()
    calibration = _calibration_from_cache(kv_cache, positions)
    for entry in calibration.entries.values():
        assert torch.all(entry.eigenvalues[:-1] >= entry.eigenvalues[1:])


def test_empty_cache_edge_case() -> None:
    kv_cache = {"keys": torch.zeros(1, 0, 2, 8), "values": torch.zeros(1, 0, 2, 8)}
    calibration = PCACalibrator(head_group_size=1).compute(bit_budget_ratio=0.25)
    compressor = KVTCCompressor(calibration)
    compressed = compressor.compress(kv_cache, torch.zeros(0, dtype=torch.long))
    restored = compressor.decompress(compressed)
    assert restored["keys"].shape[1] == 0


def test_single_token_edge_case() -> None:
    kv_cache, positions = _synthetic_kv(tokens=1)
    calibration = _calibration_from_cache(kv_cache, positions)
    compressor = KVTCCompressor(calibration)
    restored = compressor.decompress(compressor.compress(kv_cache, positions))
    assert torch.equal(restored["keys"], kv_cache["keys"])


def test_all_zeros_edge_case() -> None:
    kv_cache = {"keys": torch.zeros(1, 200, 2, 8), "values": torch.zeros(1, 200, 2, 8)}
    positions = torch.arange(200)
    calibration = _calibration_from_cache(kv_cache, positions)
    compressor = KVTCCompressor(calibration)
    restored = compressor.decompress(compressor.compress(kv_cache, positions))
    assert torch.equal(restored["keys"][:, :4], kv_cache["keys"][:, :4])
    assert torch.equal(restored["values"][:, -128:], kv_cache["values"][:, -128:])


def test_cache_wrapper_roundtrip() -> None:
    kv_cache, positions = _synthetic_kv(layers=1)
    calibration = _calibration_from_cache(kv_cache, positions)
    cache = KVTCCache(KVTCCompressor(calibration))
    cache.update(0, kv_cache["keys"][0], kv_cache["values"][0])
    cache.evict_to_compressed(0, positions)
    assert cache.is_compressed(0)
    restored = cache.restore_layer(0)
    assert restored["keys"].shape == kv_cache["keys"][0].shape


def test_calibrator_save_and_load(tmp_path) -> None:
    kv_cache, positions = _synthetic_kv()
    calibration = _calibration_from_cache(kv_cache, positions)
    calibrator = KVTCCalibrator()
    path = tmp_path / "calibration.pt"
    calibrator.save(path, calibration)
    restored = calibrator.load(path)
    assert restored.head_group_size == calibration.head_group_size
    assert restored.entries.keys() == calibration.entries.keys()


def test_pipeline_handles_no_middle_region() -> None:
    kv_cache, positions = _synthetic_kv(tokens=100)
    calibration = _calibration_from_cache(kv_cache, positions)
    compressed = KVTCCompressor(calibration).compress(kv_cache, positions)
    assert compressed.metadata is not None
    assert compressed.metadata.middle_len == 0


def test_grouped_head_calibration_entries_exist() -> None:
    kv_cache, positions = _synthetic_kv(heads=4)
    calibration = _calibration_from_cache(kv_cache, positions)
    assert (0, 0, "keys") in calibration.entries
    assert (0, 1, "values") in calibration.entries


def test_quantization_error_decreases_with_more_bits() -> None:
    values = torch.linspace(-2.0, 2.0, 129)
    errors = []
    for bits in [2, 4, 8]:
        scale = 4.0 / ((1 << bits) - 1)
        zero_point = (1 << (bits - 1)) - 0.5
        quantized = uniform_quantize(values, bits, scale, zero_point)
        restored = uniform_dequantize(quantized, bits, scale, zero_point)
        errors.append(torch.mean((values - restored) ** 2).item())
    assert errors[2] < errors[1] < errors[0]


def test_decompress_rejects_wrong_size() -> None:
    with pytest.raises(ValueError):
        decompress(compress(b"abc")[0], 99)


def test_rope_requires_even_head_dim() -> None:
    with pytest.raises(ValueError):
        apply_rope(torch.randn(2, 1, 7), torch.arange(2), head_dim=7)
