# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import torch

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import (KVConnectorBase_V1,
                                                          KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.offloading.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.offloading.factory import OffloadingSpecFactory
from vllm.v1.offloading.mediums import GPULoadStoreSpec
from vllm.v1.offloading.spec import OffloadingSpec
from vllm.v1.offloading.worker.worker import (OffloadingQueueManager,
                                              TransferSpec)
from vllm.v1.request import Request

ReqId = str

logger = init_logger(__name__)


@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    reqs_to_load: dict[ReqId, TransferSpec]
    reqs_to_store: dict[ReqId, TransferSpec]


class OffloadingConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        super().__init__(vllm_config, role)

        spec = OffloadingSpecFactory.create_spec(vllm_config)

        self.connector_scheduler: Optional[OffloadingConnectorScheduler] = None
        self.connector_worker: Optional[OffloadingConnectorWorker] = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = OffloadingConnectorScheduler(spec)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = OffloadingConnectorWorker(spec)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata,
                          OffloadingConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        pass

    def wait_for_save(self):
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata,
                          OffloadingConnectorMetadata)
        self.connector_worker.start_store_kv(self._connector_metadata)

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(finished_req_ids)

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens)

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, spec: OffloadingSpec):
        self.gpu_block_size = spec.gpu_block_size
        self.offloaded_block_size = spec.offloaded_block_size
        self.block_size_factor = (self.offloaded_block_size //
                                  self.gpu_block_size)
        self.manager: OffloadingManager = spec.create_manager()

        self._requests: dict[ReqId, Request] = {}
        self._reqs_to_load: dict[ReqId, TransferSpec] = {}

        # request ID -> set(block hashes being stored/load)
        self._reqs_being_stored: defaultdict[ReqId, set[int]] = (defaultdict(
            set[int]))
        self._reqs_being_loaded: defaultdict[ReqId, set[int]] = (defaultdict(
            set[int]))

    def get_num_new_matched_tokens(
            self, request: Request,
            num_computed_tokens: int) -> tuple[int, bool]:
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
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).
        """
        if (request.num_tokens - num_computed_tokens
                < self.offloaded_block_size):
            return 0, False

        start_block_idx = num_computed_tokens // self.offloaded_block_size
        end_block_idx = request.num_tokens // self.offloaded_block_size
        if start_block_idx == end_block_idx:
            return 0, False

        full_block_tokens = self.offloaded_block_size * end_block_idx
        if full_block_tokens - num_computed_tokens < self.offloaded_block_size:
            return 0, False

        assert (len(request.block_hashes)
                >= end_block_idx * self.block_size_factor)
        block_hashes = [
            blk_hash.hash_value for blk_hash in request.
            block_hashes[start_block_idx *
                         self.block_size_factor::self.block_size_factor]
        ]
        hits = self.manager.lookup(block_hashes)
        if hits == 0:
            return 0, False

        num_hit_tokens = (self.offloaded_block_size *
                          (start_block_idx + hits) - num_computed_tokens)
        logger.debug(f"Request {request.request_id} hit "
                     f"{num_hit_tokens} offloaded tokens after "
                     f"{num_computed_tokens} GPU hit tokens")
        if num_hit_tokens < self.offloaded_block_size:
            return 0, False

        return num_hit_tokens, True

    def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks,
                                 num_external_tokens: int):
        self._requests[request.request_id] = request

        if num_external_tokens == 0:
            return

        block_groups = blocks.get_block_ids()
        assert len(block_groups) == 1, "Only one group is supported"

        num_computed_gpu_blocks = sum(block is not None
                                      for block in block_groups[0])
        num_computed_tokens = num_computed_gpu_blocks * self.gpu_block_size
        assert (num_computed_tokens +
                num_external_tokens) % self.offloaded_block_size == 0

        num_pending_gpu_blocks = len(block_groups[0]) - num_computed_gpu_blocks
        assert (num_external_tokens == num_pending_gpu_blocks *
                self.gpu_block_size)

        start_block_idx = num_computed_tokens // self.offloaded_block_size
        end_block_idx = request.num_tokens // self.offloaded_block_size

        assert (len(request.block_hashes)
                >= end_block_idx * self.block_size_factor)
        block_hashes = [
            blk_hash.hash_value for blk_hash in request.
            block_hashes[start_block_idx *
                         self.block_size_factor::self.block_size_factor]
        ]

        src_specs = self.manager.prepare_load(block_hashes)
        dst_specs = [
            GPULoadStoreSpec(gpu_block_id)
            for gpu_block_id in block_groups[0][num_computed_gpu_blocks:]
        ]

        self._reqs_to_load[request.request_id] = (src_specs, dst_specs)
        self._reqs_being_loaded[request.request_id] |= set(block_hashes)

    def _get_reqs_to_store(self, scheduler_output: SchedulerOutput):
        reqs_to_store: dict[ReqId, TransferSpec] = {}
        # iterate over both new and cached requests
        for req_id, block_ids in itertools.chain(
            ((req_data.req_id, req_data.block_ids)
             for req_data in scheduler_output.scheduled_new_reqs),
                zip(scheduler_output.scheduled_cached_reqs.req_ids,
                    scheduler_output.scheduled_cached_reqs.new_block_ids)):
            req = self._requests[req_id]
            new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            total_tokens = req.num_computed_tokens + new_tokens
            num_blocks = total_tokens // self.offloaded_block_size

            if num_blocks == 0:
                continue

            num_gpu_blocks = num_blocks * self.block_size_factor
            block_hashes = [
                blk_hash.hash_value for blk_hash in
                req.block_hashes[:num_gpu_blocks:self.block_size_factor]
            ]
            assert len(block_hashes) == num_blocks

            store_output = self.manager.prepare_store(block_hashes)
            if store_output is None:
                logger.warning(f"Cannot store {num_blocks} blocks")
                break

            block_hashes_to_store = set(store_output.block_hashes_to_store)
            if not block_hashes_to_store:
                continue

            dst_specs = store_output.store_specs
            src_specs: list[LoadStoreSpec] = []
            for gpu_block_idx, blk_hash in enumerate(req.block_hashes):
                if blk_hash.hash_value not in block_hashes_to_store:
                    continue
                for i in range(self.block_size_factor):
                    src_specs.append(
                        GPULoadStoreSpec(block_ids[gpu_block_idx + i]))

            reqs_to_store[req_id] = (src_specs, dst_specs)
            self._reqs_being_stored[req_id] |= block_hashes_to_store

            logger.debug(f"Request {req_id} offloading "
                         f"{len(block_hashes_to_store)} blocks")

        return reqs_to_store

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = OffloadingConnectorMetadata(
            reqs_to_load=self._reqs_to_load,
            reqs_to_store=self._get_reqs_to_store(scheduler_output))
        self._reqs_to_load.clear()
        return meta

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """

        # TODO (orozery): Move this call to when the request
        #  is actually done storing
        block_hashes = self._reqs_being_stored.pop(request.request_id)
        if block_hashes:
            self.manager.complete_store(list(block_hashes))

        # TODO (orozery): Move this call to when the request
        #  is actually done loading
        block_hashes = self._reqs_being_loaded.pop(request.request_id)
        if block_hashes:
            self.manager.complete_load(list(block_hashes))

        self._requests.pop(request.request_id, None)

        return True, None


class OffloadingConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, spec: OffloadingSpec):
        self.spec = spec
        self.manager = OffloadingQueueManager()

        self._job_counter = 0

        # req_id -> (job_id, is_store)
        self._jobs: dict[int, tuple[ReqId, bool]] = {}
        # req_id -> set(active job IDs)
        self._load_jobs: defaultdict[ReqId, set[int]] = defaultdict(set[int])
        self._store_jobs: defaultdict[ReqId, set[int]] = defaultdict(set[int])

        self._finished_reqs_waiting_for_store: set[ReqId] = set()

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter = job_id + 1
        return job_id

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        for src_cls, dst_cls, transfer_fn in self.spec.get_transfer_functions(
                kv_caches):
            self.manager.register_worker(src_cls, dst_cls, transfer_fn)

    def start_load_kv(self, metadata: OffloadingConnectorMetadata):
        for req_id, transfer_spec in metadata.reqs_to_load.items():
            job_id = self._generate_job_id()
            self._jobs[job_id] = (req_id, False)
            self._load_jobs[req_id].add(job_id)
            self.manager.transfer_async(job_id, transfer_spec)

    def start_store_kv(self, metadata: OffloadingConnectorMetadata):
        for req_id, transfer_spec in metadata.reqs_to_store.items():
            job_id = self._generate_job_id()
            self._jobs[job_id] = (req_id, True)
            self._store_jobs[req_id].add(job_id)
            self.manager.transfer_async(job_id, transfer_spec)

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.
        Returns a list of request IDs that finished loading or storing.

        Returns:
            ids of requests that have finished asynchronous transfer
            tuple of (sending/saving ids, recving/loading ids).
        """
        finished_sending = set()
        finished_recving = set()
        for job_id, is_success in self.manager.get_finished():
            assert is_success
            req_id, is_store = self._jobs.pop(job_id)
            if is_store:
                req_jobs = self._store_jobs[req_id]
                req_jobs.remove(job_id)
                if not req_jobs:
                    del self._store_jobs[req_id]
                    if req_id in self._finished_reqs_waiting_for_store:
                        self._finished_reqs_waiting_for_store.remove(req_id)
                        finished_sending.add(req_id)
            else:
                req_jobs = self._load_jobs[req_id]
                req_jobs.remove(job_id)
                if not req_jobs:
                    del self._load_jobs[req_id]
                    finished_recving.add(req_id)

        for req_id in finished_req_ids:
            if self._store_jobs[req_id]:
                self._finished_reqs_waiting_for_store.add(req_id)

        return finished_sending, finished_recving
