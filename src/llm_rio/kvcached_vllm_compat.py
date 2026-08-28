from __future__ import annotations

from functools import wraps
from typing import Any, cast

_INSTALL_MARKER = "_llm_rio_vllm026_packed_kv"
_ELASTIC_POOL_INSTALL_MARKER = "_llm_rio_vllm026_shared_pool_race"
_ELASTIC_POOL_METHOD_MARKER = "_llm_rio_vllm026_shared_pool_retry"
_ALLOCATE_SLOTS_INSTALL_MARKER = "_llm_rio_vllm026_allocate_slots_rollback"
_ALLOCATE_SLOTS_METHOD_MARKER = "_llm_rio_vllm026_allocate_slots_transactional"


def _is_shared_pool_capacity_race(exc: BaseException) -> bool:
    """Identify kvcached's stale shared-capacity snapshot guard."""
    message = str(exc)
    return (
        isinstance(exc, ValueError)
        and message.startswith("Cannot get ")
        and message.endswith(" free blocks from the pool")
    )


def _install_shared_pool_race_shim(kvp: Any, logger: Any) -> None:
    """Translate a stale free-count guard into kvcached's retry signal.

    ElasticBlockPool.get_num_free_blocks() observes a device-wide pool shared
    by every colocated engine. A peer can consume the last physical page after
    vLLM's scheduler capacity check but before get_new_blocks() repeats that
    check. The pinned kvcached revision raises ValueError in that early path,
    although its later alloc() race raises KVCachePoolExhausted. Its existing
    allocate_slots patch catches only KVCachePoolExhausted and returns the
    scheduler's normal "not now" signal, so normalize the early path too.
    """
    patch_class = kvp.ElasticBlockPoolPatch
    original_inject = patch_class.inject_elastic_block_pool
    if getattr(original_inject, _ELASTIC_POOL_INSTALL_MARKER, False):
        return

    @wraps(original_inject)
    def inject_elastic_block_pool(self: Any, block_pool_mod: Any) -> bool:
        installed = bool(original_inject(self, block_pool_mod))
        if not installed:
            return False

        pool_class = getattr(block_pool_mod, "ElasticBlockPool", None)
        if pool_class is None:
            return False
        original_get_new_blocks = pool_class.get_new_blocks
        if getattr(original_get_new_blocks, _ELASTIC_POOL_METHOD_MARKER, False):
            return True

        @wraps(original_get_new_blocks)
        def get_new_blocks(pool: Any, num_blocks: int) -> Any:
            try:
                return original_get_new_blocks(pool, num_blocks)
            except ValueError as exc:
                if not _is_shared_pool_capacity_race(exc):
                    raise
                available = pool.get_num_free_blocks()
                raise kvp.KVCachePoolExhausted(
                    "Shared physical KV capacity changed after the scheduler "
                    f"check; requested={num_blocks}, available={available}"
                ) from exc

        setattr(get_new_blocks, _ELASTIC_POOL_METHOD_MARKER, True)
        pool_class.get_new_blocks = get_new_blocks
        logger.info("Installed LLM-RIO shared KV-pool race compatibility shim")
        return True

    setattr(inject_elastic_block_pool, _ELASTIC_POOL_INSTALL_MARKER, True)
    patch_class.inject_elastic_block_pool = inject_elastic_block_pool


def _request_status_name(request: Any) -> str:
    status = getattr(request, "status", None)
    return str(getattr(status, "name", status)).upper()


def _install_allocate_slots_rollback_shim(kvp: Any, logger: Any) -> None:
    """Roll back a waiting request when a hybrid allocation loses a race.

    kvcached turns physical-pool exhaustion into ``allocate_slots() is None``.
    With more than one vLLM KV-cache group, however, an earlier group can have
    allocated blocks before a later group raises. vLLM treats None as a
    side-effect-free capacity miss, leaves a waiting request in place, and can
    therefore strand those blocks indefinitely. Running requests need no
    special handling because vLLM's normal None path preempts and frees them.
    """
    patch_class = kvp.KVCacheManagerAllocateSlotsPatch
    original_patch = patch_class.patch_allocate_slots
    if getattr(original_patch, _ALLOCATE_SLOTS_INSTALL_MARKER, False):
        return

    @wraps(original_patch)
    def patch_allocate_slots(self: Any, manager_module: Any) -> bool:
        installed = bool(original_patch(self, manager_module))
        if not installed:
            return False

        manager_class = getattr(manager_module, "KVCacheManager", None)
        if manager_class is None:
            return False
        original_allocate_slots = manager_class.allocate_slots
        if getattr(original_allocate_slots, _ALLOCATE_SLOTS_METHOD_MARKER, False):
            return True

        @wraps(original_allocate_slots)
        def allocate_slots(manager: Any, *args: Any, **kwargs: Any) -> Any:
            request = args[0] if args else kwargs.get("request")
            coordinator = getattr(manager, "coordinator", None)
            group_managers = tuple(getattr(coordinator, "single_type_managers", ()))
            if request is None or not group_managers:
                return original_allocate_slots(manager, *args, **kwargs)

            request_id = request.request_id
            block_counts = tuple(
                len(group.req_to_blocks.get(request_id, ())) for group in group_managers
            )
            new_id_counts = tuple(
                len(getattr(group, "new_block_ids", ())) for group in group_managers
            )
            cow_counts = tuple(
                len(getattr(group, "_pending_cow_copies", ())) for group in group_managers
            )

            result = original_allocate_slots(manager, *args, **kwargs)
            if result is not None or _request_status_name(request) not in {
                "WAITING",
                "PREEMPTED",
            }:
                return result

            current_counts = tuple(
                len(group.req_to_blocks.get(request_id, ())) for group in group_managers
            )
            if not any(
                after > before for before, after in zip(block_counts, current_counts, strict=True)
            ):
                return result

            # No model step was scheduled, so none of the newly recorded IDs
            # can have reached the worker. Restore those side ledgers before
            # freeing the request's partially constructed group state.
            for group, new_id_count, cow_count in zip(
                group_managers, new_id_counts, cow_counts, strict=True
            ):
                new_ids = getattr(group, "new_block_ids", None)
                if isinstance(new_ids, list):
                    del new_ids[new_id_count:]
                pending_cows = getattr(group, "_pending_cow_copies", None)
                if isinstance(pending_cows, list):
                    del pending_cows[cow_count:]
            manager.free(request)
            logger.warning(
                "Rolled back partial shared KV allocation after a scheduling "
                "miss: request=%s groups_before=%s groups_after=%s",
                request_id,
                block_counts,
                current_counts,
            )
            return None

        setattr(allocate_slots, _ALLOCATE_SLOTS_METHOD_MARKER, True)
        manager_class.allocate_slots = allocate_slots
        logger.info("Installed LLM-RIO transactional KV allocation rollback shim")
        return True

    setattr(patch_allocate_slots, _ALLOCATE_SLOTS_INSTALL_MARKER, True)
    patch_class.patch_allocate_slots = patch_allocate_slots


def install() -> None:
    """Teach the pinned kvcached revision about vLLM 0.26 packed K/V tensors.

    vLLM 0.26 represents CUDA attention cache tensors as
    (blocks, heads, tokens, 2 * head_size). The pinned kvcached revision
    already supports vLLM's hybrid Mamba/GDN groups, but its allocator still
    recognizes only the older shapes with a standalone K/V dimension. This
    adapter expands the shape for kvcached's physical allocation, uses one
    combined physical K/V page per block, and then rebuilds the exact
    blocks-first NHD view expected by vLLM 0.26.

    The bootstrap that calls this function is enabled only for the accepted
    vLLM 0.26 compatibility tuple. Explicit-K/V and MLA tensor layouts keep
    kvcached's original implementation.
    """
    import torch
    from kvcached.integration.vllm import interfaces as kvi  # type: ignore[import-untyped]
    from kvcached.integration.vllm import patches as kvp
    from kvcached.utils import get_kvcached_logger  # type: ignore[import-untyped]

    logger = get_kvcached_logger()
    _install_shared_pool_race_shim(kvp, logger)
    _install_allocate_slots_rollback_shim(kvp, logger)

    if getattr(kvi.alloc_kv_cache, _INSTALL_MARKER, False):
        return

    original = kvi.alloc_kv_cache
    original_build_kv_views = kvi.build_kv_views
    original_get_kv_cache_params = kvp._get_kv_cache_params

    def _validate_packed_shape(
        kvcache_shape: tuple[int, ...], block_size: int, kv_layout: str
    ) -> tuple[int, int, int]:
        requested_blocks, num_heads, shape_block_size, packed_head_size = kvcache_shape
        if kv_layout != "NHD":
            raise ValueError(f"LLM-RIO packed KV shim requires NHD, got {kv_layout}")
        if shape_block_size != block_size:
            raise ValueError(
                "LLM-RIO packed KV shim received mismatched block sizes: "
                f"{shape_block_size} != {block_size}"
            )
        if packed_head_size % 2:
            raise ValueError("LLM-RIO packed KV shim requires an even packed head dimension")
        return requested_blocks, num_heads, packed_head_size

    def _packed_views(
        buffers: list[torch.Tensor],
        *,
        dtype: torch.dtype,
        num_blocks: int,
        num_heads: int,
        block_size: int,
        packed_head_size: int,
        kernel_block_size: int | None,
    ) -> list[torch.Tensor]:
        kernel_size = kernel_block_size or block_size
        if block_size % kernel_size:
            raise ValueError(
                f"block_size ({block_size}) must be a multiple of kernel_block_size ({kernel_size})"
            )
        ratio = block_size // kernel_size
        packed_shape = (num_blocks * ratio, num_heads, kernel_size, packed_head_size)
        kernel_page_elements = num_heads * kernel_size * packed_head_size
        packed_strides = (
            kernel_page_elements,
            packed_head_size,
            num_heads * packed_head_size,
            1,
        )
        return [
            torch.as_strided(buffer.view(dtype), packed_shape, packed_strides) for buffer in buffers
        ]

    def get_kv_cache_params(
        kv_cache_spec: Any,
        block_size: int,
        attention_type: str = "MHA",
    ) -> tuple[int, int]:
        if attention_type in {"MHA", "GQA"}:
            # vLLM 0.26's attention backends expose K and V packed into one
            # content dimension. The PageAllocator must manage the same
            # combined page that alloc_kv_cache() creates below.
            return kv_cache_spec.page_size_bytes // block_size, 1
        return cast(
            tuple[int, int],
            original_get_kv_cache_params(kv_cache_spec, block_size, attention_type),
        )

    def alloc_kv_cache(
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
        packed_attention = (
            attention_type in {"MHA", "GQA", "HYBRID_LINEAR"} and len(kvcache_shape) == 4
        )
        if not packed_attention:
            return original(
                kvcache_shape,
                block_size,
                dtype,
                device,
                num_layers,
                attention_type=attention_type,
                kv_layout=kv_layout,
                group_id=group_id,
                kernel_block_size=kernel_block_size,
                return_meta=return_meta,
            )

        requested_blocks, num_heads, packed_head_size = _validate_packed_shape(
            kvcache_shape, block_size, kv_layout
        )
        explicit_shape = (
            requested_blocks,
            2,
            block_size,
            num_heads,
            packed_head_size // 2,
        )
        result = original(
            explicit_shape,
            block_size,
            dtype,
            device,
            num_layers,
            # HYBRID_LINEAR selects kvcached's unified, block-interleaved K/V
            # allocation. That is the physical layout represented by vLLM
            # 0.26's packed final dimension even for ordinary MHA/GQA models.
            attention_type="HYBRID_LINEAR",
            kv_layout=kv_layout,
            group_id=group_id,
            kernel_block_size=kernel_block_size,
            return_meta=return_meta,
        )
        if return_meta:
            _, raw_info, meta = result
        else:
            _, raw_info = result
            meta = None

        if raw_info["is_contiguous"]:
            raise RuntimeError(
                "LLM-RIO vLLM 0.26 packed KV support requires KVCACHED_CONTIGUOUS_LAYOUT=false"
            )
        kernel_size = kernel_block_size or block_size
        views = _packed_views(
            raw_info["buffers"],
            dtype=dtype,
            num_blocks=int(raw_info["num_blocks"]),
            num_heads=num_heads,
            block_size=block_size,
            packed_head_size=packed_head_size,
            kernel_block_size=kernel_size,
        )
        if attention_type == "HYBRID_LINEAR":
            logger.info(
                "LLM-RIO packed hybrid KV shim active: shape=%s kernel_block_size=%d",
                kvcache_shape,
                kernel_size,
            )
            if return_meta:
                return views, raw_info, meta
            return views, raw_info

        logger.info(
            "LLM-RIO packed %s KV shim active: shape=%s kernel_block_size=%d",
            attention_type,
            kvcache_shape,
            kernel_size,
        )
        if return_meta:
            return views, meta
        return views

    def build_kv_views(
        raw_kv_tensors: list[torch.Tensor],
        kvcache_shape: tuple[int, ...],
        block_size: int,
        dtype: torch.dtype,
        attention_type: str,
        num_blocks_per_layer: int,
        gpu_mem_bytes_per_layer_k_or_v: int,
        num_layers: int,
        kernel_block_size: int | None = None,
    ) -> tuple[list[torch.Tensor], int]:
        packed_attention = attention_type in {"MHA", "GQA"} and len(kvcache_shape) == 4
        if not packed_attention:
            return cast(
                tuple[list[torch.Tensor], int],
                original_build_kv_views(
                    raw_kv_tensors,
                    kvcache_shape,
                    block_size,
                    dtype,
                    attention_type,
                    num_blocks_per_layer,
                    gpu_mem_bytes_per_layer_k_or_v,
                    num_layers,
                    kernel_block_size=kernel_block_size,
                ),
            )

        _, num_heads, packed_head_size = _validate_packed_shape(kvcache_shape, block_size, "NHD")
        views = _packed_views(
            raw_kv_tensors,
            dtype=dtype,
            num_blocks=num_blocks_per_layer,
            num_heads=num_heads,
            block_size=block_size,
            packed_head_size=packed_head_size,
            kernel_block_size=kernel_block_size,
        )
        page_size_bytes = block_size * num_heads * packed_head_size * dtype.itemsize
        return views, page_size_bytes

    setattr(alloc_kv_cache, _INSTALL_MARKER, True)
    kvi.alloc_kv_cache = alloc_kv_cache
    kvi.build_kv_views = build_kv_views
    kvp._get_kv_cache_params = get_kv_cache_params
    logger.info("Installed LLM-RIO vLLM 0.26 packed KV compatibility shim")
