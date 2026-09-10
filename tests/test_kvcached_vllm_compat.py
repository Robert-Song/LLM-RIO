from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from llm_rio.kvcached_vllm_compat import (
    _install_shared_pool_race_shim,
    _is_shared_pool_capacity_race,
    install,
)

torch = pytest.importorskip("torch", reason="optional engine compatibility tests")


def test_shared_pool_capacity_race_becomes_retry_signal() -> None:
    class PoolExhausted(RuntimeError):
        pass

    class FakeElasticBlockPool:
        def get_num_free_blocks(self) -> int:
            return 0

        def get_new_blocks(self, num_blocks: int) -> list[Any]:
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

    class FakeElasticBlockPoolPatch:
        def inject_elastic_block_pool(self, block_pool_mod: Any) -> bool:
            block_pool_mod.ElasticBlockPool = FakeElasticBlockPool
            return True

    messages: list[str] = []
    fake_kvp = SimpleNamespace(
        ElasticBlockPoolPatch=FakeElasticBlockPoolPatch,
        KVCachePoolExhausted=PoolExhausted,
    )

    _install_shared_pool_race_shim(fake_kvp, SimpleNamespace(info=messages.append))
    module = SimpleNamespace()
    assert FakeElasticBlockPoolPatch().inject_elastic_block_pool(module)

    try:
        module.ElasticBlockPool().get_new_blocks(1)
    except PoolExhausted as exc:
        assert "requested=1, available=0" in str(exc)
    else:
        raise AssertionError("shared-pool capacity race was not translated")

    assert messages == ["Installed LLM-RIO shared KV-pool race compatibility shim"]
    assert _is_shared_pool_capacity_race(ValueError("Cannot get 4 free blocks from the pool"))
    assert not _is_shared_pool_capacity_race(ValueError("unrelated invariant"))


def test_vllm026_packed_mha_uses_combined_kv_pages(monkeypatch: Any) -> None:
    from kvcached.integration.vllm import interfaces as kvi  # type: ignore[import-untyped]
    from kvcached.integration.vllm import patches as kvp

    calls: list[tuple[tuple[int, ...], str, bool]] = []
    original_build_calls = 0

    def fake_alloc(
        kvcache_shape: tuple[int, ...],
        block_size: int,
        dtype: torch.dtype,
        device: str,
        num_layers: int,
        attention_type: str = "MHA",
        kv_layout: str = "NHD",
        group_id: int = 0,
        kernel_block_size: int | None = None,
        return_meta: bool = False,
    ) -> Any:
        del device, kv_layout, group_id, kernel_block_size
        calls.append((kvcache_shape, attention_type, return_meta))
        if attention_type != "HYBRID_LINEAR":
            return "original-layout"
        requested_blocks, _, _, num_heads, head_size = kvcache_shape
        packed_head_size = 2 * head_size
        elements = requested_blocks * block_size * num_heads * packed_head_size
        buffers = [
            torch.empty(elements * dtype.itemsize, dtype=torch.uint8) for _ in range(num_layers)
        ]
        raw_info = {
            "buffers": buffers,
            "num_blocks": requested_blocks,
            "is_contiguous": False,
        }
        meta = {
            "raw_kv_tensors": buffers,
            "num_blocks_per_layer": requested_blocks,
            "gpu_mem_bytes_per_layer_k_or_v": elements * dtype.itemsize // 2,
            "num_layers": num_layers,
            "dtype": dtype,
        }
        if return_meta:
            return [], raw_info, meta
        return [], raw_info

    def fake_build(*args: Any, **kwargs: Any) -> Any:
        nonlocal original_build_calls
        original_build_calls += 1
        return args, kwargs

    def fake_params(
        kv_cache_spec: Any, block_size: int, attention_type: str = "MHA"
    ) -> tuple[int, int]:
        del kv_cache_spec, block_size, attention_type
        return -1, -1

    monkeypatch.setattr(kvi, "alloc_kv_cache", fake_alloc)
    monkeypatch.setattr(kvi, "build_kv_views", fake_build)
    monkeypatch.setattr(kvp, "_get_kv_cache_params", fake_params)
    install()

    views = kvi.alloc_kv_cache(
        (5, 4, 16, 256),
        16,
        torch.float32,
        "cpu",
        2,
        attention_type="MHA",
    )

    assert calls == [((5, 2, 16, 4, 128), "HYBRID_LINEAR", False)]
    assert len(views) == 2
    assert views[0].shape == (5, 4, 16, 256)
    assert views[0].stride() == (16384, 256, 1024, 1)

    profiled_views, meta = kvi.alloc_kv_cache(
        (5, 4, 16, 256),
        16,
        torch.float32,
        "cpu",
        2,
        attention_type="MHA",
        return_meta=True,
    )
    assert calls[-1] == ((5, 2, 16, 4, 128), "HYBRID_LINEAR", True)
    assert profiled_views[0].shape == (5, 4, 16, 256)
    assert meta["num_blocks_per_layer"] == 5

    hybrid_views, raw_info = kvi.alloc_kv_cache(
        (5, 4, 16, 256),
        16,
        torch.float32,
        "cpu",
        2,
        attention_type="HYBRID_LINEAR",
    )
    assert calls[-1] == ((5, 2, 16, 4, 128), "HYBRID_LINEAR", False)
    assert hybrid_views[0].shape == (5, 4, 16, 256)
    assert raw_info["num_blocks"] == 5

    spec = SimpleNamespace(page_size_bytes=16 * 4 * 256 * torch.float32.itemsize)
    assert kvp._get_kv_cache_params(spec, 16, "MHA") == (4096, 1)

    hetero_views, page_size = kvi.build_kv_views(
        [view.view(torch.uint8) for view in views],
        (5, 2, 32, 128),
        32,
        torch.float32,
        "MHA",
        5,
        32768,
        2,
    )
    assert original_build_calls == 0
    assert hetero_views[0].shape == (5, 2, 32, 128)
    assert hetero_views[0].stride() == (8192, 128, 256, 1)
    assert page_size == 32768

    assert (
        kvi.alloc_kv_cache(
            (5, 2, 16, 4, 128),
            16,
            torch.float32,
            "cpu",
            2,
            attention_type="MHA",
        )
        == "original-layout"
    )
