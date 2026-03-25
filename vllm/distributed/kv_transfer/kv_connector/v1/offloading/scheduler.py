# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, ClassVar, NamedTuple

from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    ReqId,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec, SlidingWindowSpec
from vllm.v1.kv_offload.abstract import (
    OffloadingManager,
    OffloadKey,
    get_offload_block_hash,
    make_offload_key,
)
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import TransferSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

logger = init_logger(__name__)


class GroupOffloadSpec(NamedTuple):
    group_idx: int
    gpu_block_size: int
    offloaded_block_size: int
    hash_block_size_factor: int
    # None below means full attention
    sliding_window_size_in_blocks: int | None


@dataclass
class RequestGroupState:
    offload_keys: list[OffloadKey] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    # index of next block (of size offloaded_block_size) to offload
    next_stored_block_idx: int = 0
    # number of offloaded blocks hit (including GPU prefix cache)
    # when the request first started
    num_hit_blocks: int = 0


@dataclass(slots=True)
class RequestOffloadState:
    # initialized in OffloadingConnectorScheduler.__init__
    KV_GROUP_CONFIGS: ClassVar[tuple[GroupOffloadSpec, ...]] = ()
    BLOCK_SIZE_FACTOR: ClassVar[int] = 1

    req: Request
    group_states: tuple[RequestGroupState, ...] = field(
        default_factory=lambda: tuple(
            RequestGroupState() for _ in RequestOffloadState.KV_GROUP_CONFIGS
        )
    )

    # number of hits in the GPU cache
    num_locally_computed_tokens: int = 0

    @classmethod
    def init_class(cls, spec: OffloadingSpec) -> None:
        cls.BLOCK_SIZE_FACTOR = spec.block_size_factor

        # build per-group offloading config
        group_configs: list[GroupOffloadSpec] = []
        for idx, group in enumerate(spec.kv_cache_config.kv_cache_groups):
            kv_cache_spec = group.kv_cache_spec
            gpu_block_size = kv_cache_spec.block_size
            offloaded_block_size = gpu_block_size * spec.block_size_factor
            hash_block_size_factor = offloaded_block_size // spec.hash_block_size

            sliding_window_size_in_blocks: int | None
            if isinstance(kv_cache_spec, SlidingWindowSpec):
                assert kv_cache_spec.sliding_window > 0
                sliding_window_size_in_blocks = cdiv(
                    kv_cache_spec.sliding_window, offloaded_block_size
                )
            elif isinstance(kv_cache_spec, MambaSpec):
                # Mamba depends on a single state
                sliding_window_size_in_blocks = 1
            else:
                assert isinstance(kv_cache_spec, AttentionSpec)
                sliding_window_size_in_blocks = None

            group_configs.append(
                GroupOffloadSpec(
                    group_idx=idx,
                    gpu_block_size=gpu_block_size,
                    offloaded_block_size=offloaded_block_size,
                    hash_block_size_factor=hash_block_size_factor,
                    sliding_window_size_in_blocks=sliding_window_size_in_blocks,
                )
            )
        cls.KV_GROUP_CONFIGS = tuple(group_configs)

    def update_offload_keys(self) -> None:
        for group_config, group_state in zip(
            RequestOffloadState.KV_GROUP_CONFIGS, self.group_states
        ):
            for req_block_hash in islice(
                self.req.block_hashes,
                group_config.hash_block_size_factor * len(group_state.offload_keys)
                + group_config.hash_block_size_factor
                - 1,
                None,
                group_config.hash_block_size_factor,
            ):
                group_state.offload_keys.append(
                    make_offload_key(req_block_hash, group_config.group_idx)
                )

    def update_block_id_groups(
        self, new_block_id_groups: tuple[list[int], ...] | None
    ) -> None:
        if new_block_id_groups is None:
            return

        assert len(new_block_id_groups) == len(self.group_states)
        for group_state, new_blocks in zip(self.group_states, new_block_id_groups):
            group_state.block_ids.extend(new_blocks)

    def advance_stored_idx(self, num_offloadable_tokens: int) -> None:
        for group_config, group_state in zip(
            RequestOffloadState.KV_GROUP_CONFIGS, self.group_states
        ):
            num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
            group_state.next_stored_block_idx = num_blocks

    def update_num_hit_blocks(self, num_cached_tokens: int) -> None:
        for group_config, group_state in zip(
            RequestOffloadState.KV_GROUP_CONFIGS, self.group_states
        ):
            group_state.num_hit_blocks = (
                num_cached_tokens // group_config.offloaded_block_size
            )


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, spec: OffloadingSpec):
        RequestOffloadState.init_class(spec)
        self.manager: OffloadingManager = spec.get_manager()

        full_attention_groups: list[int] = []
        sliding_window_groups: list[int] = []
        for group_config in RequestOffloadState.KV_GROUP_CONFIGS:
            if group_config.sliding_window_size_in_blocks is None:
                full_attention_groups.append(group_config.group_idx)
            else:
                sliding_window_groups.append(group_config.group_idx)

        # sort sliding window groups by window size in decreasing order
        def _sliding_window_sort_key(i: int) -> int:
            val = RequestOffloadState.KV_GROUP_CONFIGS[i].sliding_window_size_in_blocks
            assert val is not None
            return val

        sliding_window_groups.sort(key=_sliding_window_sort_key, reverse=True)

        # used by _lookup
        self.full_attention_groups: tuple[int, ...] = tuple(full_attention_groups)
        self.sliding_window_groups: tuple[int, ...] = tuple(sliding_window_groups)
        self.lookup_groups = self.full_attention_groups + self.sliding_window_groups

        self._req_status: dict[ReqId, RequestOffloadState] = {}
        # requests to load for the current scheduler step
        self._reqs_to_load: dict[ReqId, TransferSpec] = {}
        # if GPU prefix caching is enabled,
        # track loaded blocks to avoid redundant loads
        self._blocks_being_loaded: set[OffloadKey] | None = (
            set() if spec.vllm_config.cache_config.enable_prefix_caching else None
        )

        # request ID -> set(offload keys being stored/loaded)
        self._reqs_being_stored = defaultdict[ReqId, set[OffloadKey]](set)
        self._reqs_being_loaded = defaultdict[ReqId, set[OffloadKey]](set)

    def _maximal_prefix_lookup(self, keys: Iterable[OffloadKey]) -> int | None:
        """Find the length of the maximal prefix of offloaded blocks."""
        hit_count = 0
        defer_lookup = False
        for key in keys:
            result = self.manager.lookup(key)
            if result is None:
                defer_lookup = True
                # continue lookup to allow manager to kick-off async lookups
                # for all blocks (until a miss is detected)
                result = True
            if not result:
                break
            hit_count += 1
        return hit_count if not defer_lookup else None

    def _sliding_window_lookup(
        self, keys: Sequence[OffloadKey], sliding_window_size: int
    ) -> int | None:
        """Find the maximal ending position of consecutive offloaded blocks
        within a sliding window."""
        defer_lookup = False
        consecutive_hits = 0
        for idx in range(len(keys) - 1, -1, -1):
            result = self.manager.lookup(keys[idx])
            if result is None:
                defer_lookup = True
                # continue lookup to allow manager to kick-off async lookups
                # for all blocks (until a hit is detected)
                result = False
            if not result:
                consecutive_hits = 0
            else:
                consecutive_hits += 1
                if consecutive_hits == sliding_window_size:
                    return idx + sliding_window_size if not defer_lookup else None
        return consecutive_hits if not defer_lookup else None

    def _touch(self, req_status: RequestOffloadState):
        for group_config, group_state in zip(
            RequestOffloadState.KV_GROUP_CONFIGS,
            req_status.group_states,
        ):
            if group_config.sliding_window_size_in_blocks is None:
                self.manager.touch(group_state.offload_keys)
            else:
                # we aim to keep just blocks that are necessary to hit
                # the original request (+ decoded blocks)
                blocks_to_skip = max(
                    0,
                    group_state.num_hit_blocks
                    - group_config.sliding_window_size_in_blocks,
                )
                self.manager.touch(group_state.offload_keys[blocks_to_skip:])

    def _lookup(self, req_status: RequestOffloadState) -> int | None:
        num_computed_tokens = req_status.num_locally_computed_tokens
        max_hit_size_tokens: int = req_status.req.num_tokens
        if self.sliding_window_groups:
            # the last prompt token has to be recomputed to get the logprobs
            # for sliding window attention, we must reduce by 1 to make sure
            # we still have a hit after reduction
            max_hit_size_tokens -= 1
        num_hit_tokens: int = 0
        defer_lookup = False
        # start by looking up all groups (full attention first, then sliding window)
        lookup_groups = self.lookup_groups
        while lookup_groups:
            looked_up_sliding_window: bool = False
            groups_iter = iter(lookup_groups)
            lookup_groups = ()
            for group_idx in groups_iter:
                group_config: GroupOffloadSpec = RequestOffloadState.KV_GROUP_CONFIGS[
                    group_idx
                ]
                group_state: RequestGroupState = req_status.group_states[group_idx]
                offloaded_block_size = group_config.offloaded_block_size
                offload_keys = group_state.offload_keys

                assert (
                    len(offload_keys)
                    >= req_status.req.num_tokens // offloaded_block_size
                )

                max_hit_size_tokens = min(
                    max_hit_size_tokens, len(offload_keys) * offloaded_block_size
                )
                if max_hit_size_tokens - num_computed_tokens < offloaded_block_size:
                    # we can only load less than a block, better skip
                    return 0

                num_blocks = min(
                    cdiv(max_hit_size_tokens, offloaded_block_size), len(offload_keys)
                )
                start_block_idx = num_computed_tokens // offloaded_block_size
                offload_keys = offload_keys[start_block_idx:num_blocks]
                sliding_window_size_in_blocks = (
                    group_config.sliding_window_size_in_blocks
                )

                block_hits: int | None
                if sliding_window_size_in_blocks is None:
                    # full attention
                    block_hits = self._maximal_prefix_lookup(offload_keys)
                else:
                    # sliding window
                    block_hits = self._sliding_window_lookup(
                        offload_keys, sliding_window_size_in_blocks
                    )
                if block_hits == 0:
                    return 0

                if block_hits is None:
                    defer_lookup = True
                else:
                    max_hit_size_tokens = min(
                        max_hit_size_tokens,
                        offloaded_block_size * (start_block_idx + block_hits),
                    )

                new_num_hit_tokens = max_hit_size_tokens - num_computed_tokens
                if new_num_hit_tokens < offloaded_block_size:
                    # we can only load less than a block, better skip
                    return 0

                if new_num_hit_tokens < num_hit_tokens:
                    if defer_lookup:
                        # make another iteration on all groups to check
                        # if we still need to defer lookup
                        defer_lookup = False
                        lookup_groups = self.lookup_groups
                    elif looked_up_sliding_window and not lookup_groups:
                        # we need another iteration to confirm previously looked up
                        # sliding window works with the new_num_hit_tokens
                        lookup_groups = self.sliding_window_groups

                looked_up_sliding_window |= sliding_window_size_in_blocks is not None
                num_hit_tokens = new_num_hit_tokens

        if defer_lookup:
            logger.debug(
                "Offloading manager delayed request %s",
                req_status.req.request_id,
            )
            return None

        # possibly delay request if any of the hit blocks is already being loaded
        if self._blocks_being_loaded:
            for group_config, group_state in zip(
                RequestOffloadState.KV_GROUP_CONFIGS, req_status.group_states
            ):
                offloaded_block_size = group_config.offloaded_block_size
                sliding_window_size_in_blocks = (
                    group_config.sliding_window_size_in_blocks
                )
                offload_keys = group_state.offload_keys
                num_blocks = cdiv(
                    num_computed_tokens + num_hit_tokens, offloaded_block_size
                )
                start_block_idx = num_computed_tokens // offloaded_block_size
                offload_keys = offload_keys[start_block_idx:num_blocks]
                if sliding_window_size_in_blocks is not None:
                    offload_keys = offload_keys[-sliding_window_size_in_blocks:]
                if any(key in self._blocks_being_loaded for key in offload_keys):
                    # hit blocks are being loaded, delay request
                    logger.debug(
                        "Delaying request %s since some of its"
                        " blocks are already being loaded",
                        req_status.req.request_id,
                    )
                    return None

        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            req_status.req.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )

        return num_hit_tokens

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded beyond the
        num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded beyond what is
                  already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).
        """
        req_status = self._req_status.get(request.request_id)
        is_new_request = True
        if req_status is None:
            req_status = RequestOffloadState(request)
            req_status.update_offload_keys()
            self._req_status[request.request_id] = req_status
        else:
            is_new_request = False
            # make sure block IDs are cleared
            for gs in req_status.group_states:
                gs.block_ids.clear()
        assert req_status is not None
        req_status.num_locally_computed_tokens = num_computed_tokens

        num_hit_tokens = self._lookup(req_status)
        if is_new_request:
            req_status.update_num_hit_blocks(
                num_computed_tokens + (num_hit_tokens or 0)
            )

        self._touch(req_status)

        return num_hit_tokens, bool(num_hit_tokens)

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        if num_external_tokens == 0:
            return

        req_status = self._req_status[request.request_id]

        num_locally_computed_tokens = req_status.num_locally_computed_tokens
        num_cached_tokens = num_locally_computed_tokens + num_external_tokens

        keys_to_load: list[OffloadKey] = []
        dst_block_ids: list[int] = []
        # per group
        group_sizes: list[int] = []
        block_indices: list[int] = []
        for group_config, group_state, group_blocks in zip(
            RequestOffloadState.KV_GROUP_CONFIGS,
            req_status.group_states,
            blocks.blocks,
        ):
            gpu_block_size = group_config.gpu_block_size
            offloaded_block_size = group_config.offloaded_block_size
            offload_keys = group_state.offload_keys
            num_gpu_blocks = cdiv(num_cached_tokens, gpu_block_size)

            assert len(group_blocks) >= num_gpu_blocks
            num_locally_computed_gpu_blocks = next(
                (
                    i
                    for i, block in enumerate(group_blocks)
                    if not block.is_null and block.block_hash is None
                ),
                num_gpu_blocks,
            )
            assert (
                num_locally_computed_tokens
                <= num_locally_computed_gpu_blocks * gpu_block_size
            )
            num_pending_gpu_blocks = num_gpu_blocks - num_locally_computed_gpu_blocks

            if group_config.sliding_window_size_in_blocks is not None:
                assert (
                    num_pending_gpu_blocks
                    <= group_config.sliding_window_size_in_blocks
                    * RequestOffloadState.BLOCK_SIZE_FACTOR
                )

            start_block_idx = (
                num_locally_computed_gpu_blocks // RequestOffloadState.BLOCK_SIZE_FACTOR
            )
            num_blocks = cdiv(num_cached_tokens, offloaded_block_size)
            assert len(offload_keys) >= num_blocks
            keys_to_load.extend(offload_keys[start_block_idx:num_blocks])

            dst_block_ids.extend(
                block.block_id
                for block in group_blocks[
                    num_locally_computed_gpu_blocks:num_gpu_blocks
                ]
            )
            group_sizes.append(num_pending_gpu_blocks)
            block_indices.append(num_locally_computed_gpu_blocks)

            group_state.next_stored_block_idx = num_blocks

        src_spec = self.manager.prepare_load(keys_to_load)
        dst_spec = GPULoadStoreSpec(
            dst_block_ids, group_sizes=group_sizes, block_indices=block_indices
        )

        self._reqs_to_load[request.request_id] = (src_spec, dst_spec)
        req_blocks_being_loaded = self._reqs_being_loaded[request.request_id]
        req_blocks_being_loaded.update(keys_to_load)

        if self._blocks_being_loaded is not None:
            self._blocks_being_loaded.update(req_blocks_being_loaded)

    def _get_reqs_to_store(
        self, scheduler_output: SchedulerOutput
    ) -> dict[ReqId, TransferSpec]:
        block_size_factor = RequestOffloadState.BLOCK_SIZE_FACTOR
        reqs_to_store: dict[ReqId, TransferSpec] = {}
        # iterate over both new and cached requests
        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            req_status = self._req_status[req_id]
            req_status.update_offload_keys()
            req = req_status.req

            if preempted:
                for group_state in req_status.group_states:
                    group_state.block_ids.clear()

            if new_block_id_groups:
                req_status.update_block_id_groups(new_block_id_groups)

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_tokens_after_batch = req.num_computed_tokens + num_scheduled_tokens
            # with async scheduling, some tokens may be missing
            num_offloadable_tokens = min(num_tokens_after_batch, req.num_tokens)

            new_offload_keys: list[OffloadKey] = []
            for group_config, group_state in zip(
                RequestOffloadState.KV_GROUP_CONFIGS, req_status.group_states
            ):
                num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
                start_block_idx = group_state.next_stored_block_idx
                if num_blocks <= start_block_idx:
                    continue
                offload_keys = group_state.offload_keys[start_block_idx:num_blocks]
                offload_block_ids = group_state.block_ids[start_block_idx:num_blocks]
                assert len(offload_keys) == len(offload_block_ids)

                for offload_key, block_id in zip(offload_keys, offload_block_ids):
                    if block_id != 0:
                        new_offload_keys.append(offload_key)

            if not new_offload_keys:
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue

            store_output = self.manager.prepare_store(new_offload_keys)
            if store_output is None:
                logger.warning("Request %s: cannot store blocks", req_id)
                continue

            if not store_output.keys_to_store:
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue

            self._touch(req_status)

            keys_to_store = set(store_output.keys_to_store)

            group_sizes: list[int] = []
            src_block_ids: list[int] = []
            for group_config, group_state in zip(
                RequestOffloadState.KV_GROUP_CONFIGS, req_status.group_states
            ):
                is_sliding_window = (
                    group_config.sliding_window_size_in_blocks is not None
                )
                num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
                start_block_idx = group_state.next_stored_block_idx
                block_ids = group_state.block_ids
                num_group_blocks = 0
                for idx, offload_key in enumerate(
                    group_state.offload_keys[start_block_idx:num_blocks]
                ):
                    if offload_key not in keys_to_store:
                        continue

                    offloaded_block_idx = start_block_idx + idx
                    gpu_block_idx = offloaded_block_idx * block_size_factor
                    num_group_blocks += block_size_factor
                    for i in range(block_size_factor):
                        block_id = block_ids[gpu_block_idx + i]
                        assert block_id != 0
                        src_block_ids.append(block_id)
                        if is_sliding_window:
                            # TODO (orozery):
                            #  protect sliding window blocks from GPU eviction
                            pass
                group_sizes.append(num_group_blocks)
                group_state.next_stored_block_idx = num_blocks

            src_spec = GPULoadStoreSpec(src_block_ids, group_sizes=group_sizes)
            dst_spec = store_output.store_spec

            reqs_to_store[req_id] = (src_spec, dst_spec)
            self._reqs_being_stored[req_id] |= keys_to_store

            logger.debug(
                "Request %s offloading %s blocks upto %d tokens",
                req_id,
                len(keys_to_store),
                num_offloadable_tokens,
            )

        return reqs_to_store

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = OffloadingConnectorMetadata(
            reqs_to_load=self._reqs_to_load,
            reqs_to_store=self._get_reqs_to_store(scheduler_output),
            reqs_to_flush=scheduler_output.preempted_req_ids,
        )
        self._reqs_to_load = {}

        # NOTE (orozery): we should move this logic to update_connector_output
        # once KVConnectorOutput allows us to report completed transfers
        for req_id in scheduler_output.preempted_req_ids or ():
            keys = self._reqs_being_stored.get(req_id)
            if keys:
                self.manager.complete_store(keys)
                keys.clear()

        return meta

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        for req_id in connector_output.finished_sending or []:
            keys = self._reqs_being_stored.pop(req_id, None)
            if keys:
                self.manager.complete_store(keys)

        for req_id in connector_output.finished_recving or []:
            keys = self._reqs_being_loaded.pop(req_id, None)
            if keys:
                if self._blocks_being_loaded:
                    self._blocks_being_loaded.difference_update(keys)
                self.manager.complete_load(keys)

    def request_finished(
        self,
        request: Request,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        req_id = request.request_id

        # TODO(orozery): possibly kickoff offload for last block
        # which may have been deferred due to async scheduling
        self._req_status.pop(req_id, None)

        request_being_stored = req_id in self._reqs_being_stored
        return request_being_stored, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Take the KV cache events from the connector.

        Returns:
            A list of KV cache events.
        """
        for event in self.manager.take_events():
            block_hashes = [get_offload_block_hash(key) for key in event.keys]
            if event.removed:
                yield BlockRemoved(block_hashes=block_hashes, medium=event.medium)
            else:
                yield BlockStored(
                    block_hashes=block_hashes,
                    parent_block_hash=None,
                    token_ids=[],
                    lora_id=None,
                    block_size=0,
                    medium=event.medium,
                    lora_name=None,
                )
