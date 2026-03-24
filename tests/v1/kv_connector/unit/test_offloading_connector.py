# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from vllm import SamplingParams
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_events import BlockRemoved, BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
    RequestOffloadState,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.forward_context import ForwardContext
from vllm.utils.hashing import sha256
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    get_offload_block_hash,
    make_offload_key,
)
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from .utils import (
    EOS_TOKEN_ID,
    create_model_runner_output,
    create_vllm_config,
)


def to_keys(int_ids: list[int]) -> list[OffloadKey]:
    return [make_offload_key(str(i).encode(), 0) for i in int_ids]


class MockLoadStoreSpec(LoadStoreSpec):
    def __init__(self, offload_keys: Iterable[OffloadKey]):
        self.offload_keys: list[OffloadKey] = list(offload_keys)

    @staticmethod
    def medium() -> str:
        return "Mock"

    def __repr__(self) -> str:
        return repr(self.offload_keys)


class MockOffloadingHandler(OffloadingHandler):
    def __init__(self):
        self.transfer_specs: dict[int, TransferSpec] = {}
        self.completed_transfers: list[TransferResult] = []
        self.waiting_jobs: set[int] = set()
        self.completed_jobs: list[int] = []
        self.flushed_jobs: set[int] = set()

    def get_finished(self) -> list[TransferResult]:
        finished = self.completed_transfers
        self.completed_transfers = []
        return finished

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        self.transfer_specs[job_id] = spec
        self.waiting_jobs.add(job_id)
        return True

    def complete_jobs(self, job_ids: set[int]) -> None:
        for job_id in job_ids:
            if job_id in self.waiting_jobs:
                self.waiting_jobs.remove(job_id)
                self.completed_jobs.append(job_id)
                result = TransferResult(
                    job_id=job_id,
                    success=True,
                    transfer_size=None,
                    transfer_time=None,
                    transfer_type=None,
                )
                self.completed_transfers.append(result)

    def wait(self, job_ids: set[int]) -> None:
        self.flushed_jobs |= job_ids
        self.complete_jobs(job_ids)


class MockOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        self.manager = MagicMock(spec=OffloadingManager)
        self.manager.lookup.return_value = 0
        self.manager.prepare_load = lambda keys: MockLoadStoreSpec(keys)
        self.handler = MockOffloadingHandler()

    def get_manager(self) -> OffloadingManager:
        return self.manager

    def get_handlers(
        self, _, __
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        yield GPULoadStoreSpec, MockLoadStoreSpec, self.handler
        yield MockLoadStoreSpec, GPULoadStoreSpec, self.handler

    def complete_transfers(self):
        self.handler.complete_jobs(self.handler.waiting_jobs.copy())

    def get_completed_transfers(self) -> list[TransferSpec]:
        specs = [
            self.handler.transfer_specs[job_id]
            for job_id in self.handler.completed_jobs
        ]
        self.handler.completed_jobs.clear()
        return specs

    def get_flushed_transfers(self):
        specs = [
            self.handler.transfer_specs[job_id] for job_id in self.handler.flushed_jobs
        ]
        self.handler.flushed_jobs.clear()
        return specs


@dataclass(frozen=True)
class GPUBlock:
    group_idx: int
    request_block_offset: int


@dataclass
class TransferSummary:
    gpu_blocks: list[GPUBlock]
    offload_addresses: list[Any]


class RequestRunner:
    def __init__(
        self,
        offloaded_block_size: int,
        gpu_block_size: int,
        num_gpu_blocks: int,
        async_scheduling: bool = True,
    ):
        self.offloaded_block_size: int = offloaded_block_size
        self.gpu_block_size: int = gpu_block_size
        self.num_gpu_blocks: int = num_gpu_blocks
        self.async_scheduling: bool = async_scheduling

        self.req_id: int = -1

        vllm_config = create_vllm_config(
            block_size=gpu_block_size, max_num_batched_tokens=1000
        )
        vllm_config.scheduler_config.async_scheduling = async_scheduling
        vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "spec_name": "MockOffloadingSpec",
                "spec_module_path": "tests.v1.kv_connector.unit.test_offloading_connector",  # noqa: E501
                "block_size": offloaded_block_size,
            },
        )

        block_size = vllm_config.cache_config.block_size
        kv_cache_config = KVCacheConfig(
            num_blocks=num_gpu_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                )
            ],
        )
        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        self.num_kv_groups = len(kv_cache_config.kv_cache_groups)

        scheduler_cls = AsyncScheduler if async_scheduling else Scheduler
        self.scheduler = scheduler_cls(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            log_stats=True,
            structured_output_manager=StructuredOutputManager(vllm_config),
            block_size=block_size,
        )

        self.worker_connector = OffloadingConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )

        # register worker kv_caches to enable OffloadingWorker creations
        self.worker_connector.register_cross_layers_kv_cache(
            kv_cache=torch.empty(0),
            attn_backend=FlashAttentionBackend,
        )

        # extract connector of scheduler
        scheduler_connector = self.scheduler.connector
        assert scheduler_connector is not None
        assert isinstance(scheduler_connector, OffloadingConnector)
        self.scheduler_connector: OffloadingConnector = scheduler_connector

        # extract mocked OffloadingManager of scheduler connector
        self.connector_scheduler = scheduler_connector.connector_scheduler
        assert self.connector_scheduler is not None
        manager = self.connector_scheduler.manager
        assert isinstance(manager, MagicMock)
        self.manager: MagicMock = manager

        assert len(RequestOffloadState.KV_GROUP_CONFIGS) == 1
        assert RequestOffloadState.KV_GROUP_CONFIGS[0].gpu_block_size == gpu_block_size
        assert (
            RequestOffloadState.KV_GROUP_CONFIGS[0].offloaded_block_size
            == offloaded_block_size
        )

        # extract OffloadingSpec of worker_connector
        connector_worker = self.worker_connector.connector_worker
        assert connector_worker is not None
        offloading_spec = connector_worker.spec
        assert isinstance(offloading_spec, MockOffloadingSpec)
        self.offloading_spec: MockOffloadingSpec = offloading_spec

        # mapping (offloading address) -> GPUBlock
        self.offloaded: dict[Any, GPUBlock] = {}

        self.completed_loads: list[TransferSummary] = []
        self.completed_stores: list[TransferSummary] = []
        self.flushed_gpu_blocks: set[GPUBlock] = set()

        # block_id -> GPUBlock
        self.gpu_blocks: dict[int, GPUBlock] = {}

        init_none_hash(sha256)
        self._block_hasher = get_request_block_hasher(gpu_block_size, sha256)

        self._dummy_ctx: ForwardContext = ForwardContext(
            no_compile_layers={},
            attn_metadata={},
            slot_mapping={},
        )

    def new_request(self, token_ids: list[int]):
        self.req_id += 1

        sampling_params = SamplingParams(max_tokens=1000)
        sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)

        req = Request(
            request_id=str(self.req_id),
            prompt_token_ids=token_ids,
            sampling_params=sampling_params,
            pooling_params=None,
            block_hasher=self._block_hasher,
        )

        self.scheduler.add_request(req)

    def _parse_transfers(self):
        for transfer_spec in self.offloading_spec.get_flushed_transfers():
            src_spec, dst_spec = transfer_spec
            assert isinstance(src_spec, GPULoadStoreSpec)

            for block_id in src_spec.block_ids:
                self.flushed_gpu_blocks.add(self.gpu_blocks[block_id.item()])

        block_size_factor = self.offloaded_block_size // self.gpu_block_size

        for transfer_spec in self.offloading_spec.get_completed_transfers():
            src_spec, dst_spec = transfer_spec

            if isinstance(src_spec, GPULoadStoreSpec):
                store = True
                gpu_spec = src_spec
                offload_spec = dst_spec
            else:
                store = False
                gpu_spec = dst_spec
                offload_spec = src_spec

            assert isinstance(offload_spec, MockLoadStoreSpec)
            assert isinstance(gpu_spec, GPULoadStoreSpec)
            assert len(gpu_spec.group_sizes) == self.num_kv_groups

            gpu_blocks: list[GPUBlock] = []
            for block_id in gpu_spec.block_ids:
                gpu_blocks.append(self.gpu_blocks[block_id.item()])

            # list of (offload_key, sub_block_offset)
            offload_addresses: list[Any] = []
            for offload_key in offload_spec.offload_keys:
                for sub_block_idx in range(block_size_factor):
                    offload_addresses.append((offload_key, sub_block_idx))

            if store:
                assert len(gpu_blocks) == len(offload_addresses)
                assert gpu_spec.block_indices is None

                self.completed_stores.append(
                    TransferSummary(gpu_blocks, offload_addresses)
                )
            else:
                assert gpu_spec.block_indices is not None
                assert len(gpu_spec.block_indices) == self.num_kv_groups

                gpu_block_offset = 0
                offload_address_offset = 0
                for group_size, logical_offset in zip(
                    gpu_spec.group_sizes, gpu_spec.block_indices
                ):
                    gpu_block_end_offset = gpu_block_offset + group_size
                    assert gpu_block_end_offset <= len(gpu_blocks)

                    offload_addresses_to_skip = -logical_offset % block_size_factor
                    offload_addresses_end_offset = (
                        offload_address_offset + offload_addresses_to_skip + group_size
                    )
                    assert offload_addresses_end_offset <= len(offload_addresses)

                    offload_addresses = (
                        offload_addresses[:offload_address_offset]
                        + offload_addresses[
                            offload_address_offset + offload_addresses_to_skip :
                        ]
                    )

                    gpu_block_offset += group_size
                    offload_address_offset += group_size

                assert gpu_block_offset == len(gpu_blocks)
                assert offload_address_offset == len(offload_addresses)

                self.completed_loads.append(
                    TransferSummary(gpu_blocks, offload_addresses)
                )

    def _update_gpu_blocks(self):
        for group_idx, manager in enumerate(
            self.scheduler.kv_cache_manager.coordinator.single_type_managers
        ):
            for blocks in manager.req_to_blocks.values():
                for block_idx, block in enumerate(blocks):
                    self.gpu_blocks[block.block_id] = GPUBlock(group_idx, block_idx)

    def _run(self, decoded_tokens: list[int], complete_transfers: bool):
        """
        Runs multiple engine (scheduler + worker) steps.
        Assumes a single request is running.

        Args:
            decoded_tokens: the tokens to yield at each step.
            complete_transfers: complete transfers immediately
        """

        tokens_iter = iter(decoded_tokens)
        token_id = next(tokens_iter, None)
        prev_scheduler_output = None
        prev_model_runner_output = None
        while True:
            assert self.scheduler.requests

            scheduler_output = self.scheduler.schedule()
            self._update_gpu_blocks()

            kv_connector_metadata = scheduler_output.kv_connector_metadata
            assert kv_connector_metadata is not None
            assert isinstance(kv_connector_metadata, OffloadingConnectorMetadata)

            self.worker_connector.handle_preemptions(kv_connector_metadata)

            self.worker_connector.bind_connector_metadata(kv_connector_metadata)
            self.worker_connector.start_load_kv(self._dummy_ctx)

            if scheduler_output.total_num_scheduled_tokens > 0:
                self.worker_connector.wait_for_save()

            if complete_transfers:
                self.offloading_spec.complete_transfers()

            finished_sending, finished_recving = self.worker_connector.get_finished(
                scheduler_output.finished_req_ids
            )

            self.worker_connector.clear_connector_metadata()

            model_runner_output = create_model_runner_output(
                reqs=self.scheduler.running,
                finished_sending=finished_sending,
                finished_recving=finished_recving,
                token_id=token_id or 0,
            )

            prev_token_id = token_id
            if self.scheduler.running:
                token_id = next(tokens_iter, None)

            if self.async_scheduling:
                # in async scheduling we update the output of the previous step
                if prev_model_runner_output is not None:
                    self.scheduler.update_from_output(
                        prev_scheduler_output, prev_model_runner_output
                    )
                prev_scheduler_output = scheduler_output
                prev_model_runner_output = model_runner_output
            else:
                self.scheduler.update_from_output(scheduler_output, model_runner_output)

            if (
                prev_token_id == EOS_TOKEN_ID
                and prev_token_id != token_id
                and self.scheduler.requests
            ):
                # continue for one more step to allow offloading to kick off
                continue

            if token_id is None:
                if self.async_scheduling:
                    # sample last token
                    self.scheduler.update_from_output(
                        prev_scheduler_output, prev_model_runner_output
                    )
                break

        self._parse_transfers()

        # run one more step to update finished stored
        if EOS_TOKEN_ID in decoded_tokens:
            assert not self.scheduler.running

            while self.scheduler.requests:
                scheduler_output = self.scheduler.schedule()

                finished_sending, finished_recving = self.worker_connector.get_finished(
                    scheduler_output.finished_req_ids
                )

                assert not finished_recving

                model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
                model_runner_output.kv_connector_output = KVConnectorOutput(
                    finished_sending=finished_sending
                )

                self.scheduler.update_from_output(scheduler_output, model_runner_output)

    def _to_gpu_blocks(
        self, blocks: tuple[int | tuple[int, int], ...]
    ) -> list[GPUBlock]:
        gpu_blocks: list[GPUBlock] = []
        for block in blocks:
            if isinstance(block, int):
                for group_idx in range(self.num_kv_groups):
                    gpu_blocks.append(
                        GPUBlock(group_idx=group_idx, request_block_offset=block)
                    )
            else:
                group_idx, offset = block
                gpu_blocks.append(
                    GPUBlock(group_idx=group_idx, request_block_offset=offset)
                )
        return gpu_blocks

    def run(
        self,
        decoded_tokens: list[int],
        complete_transfers: bool = True,
        expected_stored: tuple[int | tuple[int, int], ...] = (),
        expected_loaded: tuple[int | tuple[int, int], ...] = (),
        expected_flushed: tuple[int | tuple[int, int], ...] = (),
    ):
        """
        Runs multiple engine (scheduler + worker) steps.
        Assumes a single request is running.

        Args:
            decoded_tokens: the tokens to yield at each step.
            complete_transfers: complete transfers immediately
            expected_stored: GPU blocks
                that are expected to be written during the run.
            expected_loaded: GPU blocks
                that are expected to be loaded during the run.
            expected_flushed: GPU blocks
                that are expected to be flushed during the run.

            A GPU block is either a (group_idx: int, request_block_offset: int)
            or just request_block_offset: int.
            The latter case is a convenience for representing all groups.
        """

        expected_stored_gpu_blocks = self._to_gpu_blocks(expected_stored)
        expected_loaded_gpu_blocks = self._to_gpu_blocks(expected_loaded)
        expected_flushed_gpu_blocks = self._to_gpu_blocks(expected_flushed)

        self.manager.reset_mock()
        self._run(decoded_tokens, complete_transfers)

        loaded_gpu_blocks: set[GPUBlock] = set()
        for transfer in self.completed_loads:
            for gpu_block, offloaded_address in zip(
                transfer.gpu_blocks, transfer.offload_addresses
            ):
                loaded_gpu_blocks.add(gpu_block)
                assert gpu_block == self.offloaded[offloaded_address]

        assert set(expected_loaded_gpu_blocks) == loaded_gpu_blocks
        self.completed_loads.clear()

        stored_gpu_blocks: set[GPUBlock] = set()
        for transfer in self.completed_stores:
            for gpu_block, offloaded_address in zip(
                transfer.gpu_blocks, transfer.offload_addresses
            ):
                stored_gpu_blocks.add(gpu_block)
                self.offloaded[offloaded_address] = gpu_block

        assert set(expected_stored_gpu_blocks) == stored_gpu_blocks
        self.completed_stores.clear()

        assert set(expected_flushed_gpu_blocks) == self.flushed_gpu_blocks
        self.flushed_gpu_blocks.clear()


@pytest.fixture
def request_runner():
    runners = []

    def runner_factory(
        offloaded_block_size, gpu_block_size, num_gpu_blocks, async_scheduling
    ):
        runner = RequestRunner(
            offloaded_block_size=offloaded_block_size,
            gpu_block_size=gpu_block_size,
            num_gpu_blocks=num_gpu_blocks,
            async_scheduling=async_scheduling,
        )
        runners.append(runner)
        return runner

    yield runner_factory  # pass factory to the test


def generate_store_output(keys: Iterable[OffloadKey]):
    keys = list(keys)
    return PrepareStoreOutput(
        keys_to_store=list(keys),
        store_spec=MockLoadStoreSpec(keys),
        evicted_keys=[],
    )


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_offloading_connector(request_runner, async_scheduling: bool):
    offloaded_block_size = 12
    gpu_block_size = 4
    num_gpu_blocks = 100
    block_size_factor = offloaded_block_size // gpu_block_size

    runner = request_runner(
        offloaded_block_size=offloaded_block_size,
        gpu_block_size=gpu_block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
    )

    # 3 blocks, store just the middle block (skip first and last)
    # blocks = [0, 1, 2], [3, 4, 5], [6, 7, 8]
    runner.new_request(token_ids=[0] * offloaded_block_size * 3)
    runner.manager.prepare_store.side_effect = lambda keys: generate_store_output(
        list(keys)[1:2]
    )
    runner.run(decoded_tokens=[0])

    # add block missing 1 token -> no offload
    runner.run(
        decoded_tokens=[0] * (offloaded_block_size - 1),
        expected_stored=(3, 4, 5),
    )
    runner.manager.touch.assert_not_called()

    # +1 token -> single block, fail prepare_store
    runner.manager.prepare_store.side_effect = lambda keys: None
    runner.run(decoded_tokens=[0])
    runner.manager.prepare_store.assert_called()

    # 1 more block (+ token for async scheduling)
    # now set keys_to_store = []
    runner.manager.prepare_store.side_effect = lambda keys: generate_store_output([])
    runner.run(decoded_tokens=[0] * (offloaded_block_size + 1))

    # 1 more block (+ token for kicking off offloading)
    # now check touch was called with all 6 blocks
    runner.manager.prepare_store.side_effect = lambda keys: generate_store_output(keys)
    runner.run(
        decoded_tokens=[0] * (offloaded_block_size + 1),
        expected_stored=(15, 16, 17),
    )
    runner.manager.touch.assert_called()
    block_hashes1 = list(runner.manager.touch.call_args.args[0])
    assert len(block_hashes1) == 6

    # terminate request
    runner.run(decoded_tokens=[EOS_TOKEN_ID])

    # create a new request differing only on the last token
    runner.new_request(token_ids=[0] * (offloaded_block_size * 6 - 1) + [1])
    runner.run(decoded_tokens=[0])
    runner.manager.touch.assert_called()
    block_hashes2 = list(runner.manager.touch.call_args.args[0])
    assert len(block_hashes2) == 6

    # verify hashes are the same, except for the last block
    assert block_hashes1[:5] == block_hashes2[:5]
    assert block_hashes1[5] != block_hashes2[5]

    # terminate request
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=tuple(range(6 * block_size_factor)),
    )

    # full_block_tokens - num_computed_tokens < offloaded_block_size
    runner.new_request(
        token_ids=[0] * gpu_block_size + [1] * (offloaded_block_size - gpu_block_size)
    )
    runner.manager.prepare_store.side_effect = lambda keys: generate_store_output([])
    runner.run(decoded_tokens=[EOS_TOKEN_ID])
    runner.manager.lookup.assert_not_called()

    # single block lookup with no hits
    runner.new_request(token_ids=[1] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = lambda keys: generate_store_output([])
    runner.run(decoded_tokens=[EOS_TOKEN_ID])
    runner.manager.lookup.assert_called_once()

    # single block lookup with a hit
    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = lambda keys: generate_store_output([])
    runner.connector_scheduler._maximal_prefix_lookup = lambda key: 1
    runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_loaded=(0, 1, 2))

    # single block lookup with a hit in a middle block
    runner.new_request(
        token_ids=[0] * offloaded_block_size * 2 + [1] * offloaded_block_size
    )
    runner.manager.prepare_store.side_effect = lambda keys: generate_store_output([])
    runner.connector_scheduler._maximal_prefix_lookup = lambda key: 1
    runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_loaded=(3, 4, 5))

    # test take_events
    def to_keys(int_ids: list[int]) -> list[OffloadKey]:
        return [make_offload_key(str(i).encode(), 0) for i in int_ids]

    def to_hashes(int_ids: list[int]) -> list[bytes]:
        return [str(i).encode() for i in int_ids]

    def take_events() -> Iterable[OffloadingEvent]:
        yield OffloadingEvent(
            keys=to_keys([1, 2, 3]), block_size=16, medium="A", removed=False
        )
        yield OffloadingEvent(
            keys=to_keys([4, 5, 6]), block_size=32, medium="B", removed=True
        )

    runner.manager.take_events.side_effect = take_events
    events = list(runner.scheduler_connector.take_events())
    assert len(events) == 2
    event = events[0]
    assert isinstance(event, BlockStored)
    assert event.block_hashes == to_hashes([1, 2, 3])
    assert event.block_size == 16
    assert event.medium == "A"
    assert event.token_ids == []
    assert event.parent_block_hash is None
    assert event.lora_id is None
    assert event.lora_name is None
    event = events[1]
    assert isinstance(event, BlockRemoved)
    assert event.block_hashes == to_hashes([4, 5, 6])
    assert event.medium == "B"


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_request_preemption(request_runner, async_scheduling: bool):
    offloaded_block_size = 12
    gpu_block_size = 4
    num_gpu_blocks = 100

    runner = request_runner(
        offloaded_block_size=offloaded_block_size,
        gpu_block_size=gpu_block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
    )

    free_block_queue = runner.scheduler.kv_cache_manager.block_pool.free_block_queue
    num_free_blocks_empty = free_block_queue.num_free_blocks

    # 2 blocks, store all, without flushing
    # blocks = [0, 1, 2], [3, 4, 5]
    runner.new_request(token_ids=[0] * offloaded_block_size * 2)
    runner.manager.prepare_store.side_effect = lambda keys: generate_store_output(keys)
    runner.run(
        decoded_tokens=[0],
        complete_transfers=False,
    )

    # decode 2 more blocks - 1 gpu block, storing [6, 7, 8] (no flush)
    runner.manager.prepare_store.side_effect = lambda keys: generate_store_output(keys)
    runner.run(
        decoded_tokens=[0] * (2 * offloaded_block_size - gpu_block_size),
        complete_transfers=False,
    )

    # simulate KV cache running out of space
    free_block_queue.num_free_blocks = 0

    # request should be preempted now
    runner.run(
        decoded_tokens=[],
        complete_transfers=False,
        expected_flushed=(0, 1, 2, 3, 4, 5, 6, 7, 8),
        expected_stored=(0, 1, 2, 3, 4, 5, 6, 7, 8),
    )

    # restore KV cache space and reset GPU prefix cache
    free_block_queue.num_free_blocks = num_free_blocks_empty
    runner.scheduler.reset_prefix_cache()

    # request should now return from preemption
    # re-load [0, ..., 8] from the CPU and store [9, 10, 11]
    runner.manager.lookup.return_value = 3
    runner.manager.prepare_store.side_effect = lambda keys: generate_store_output(keys)
    runner.run(
        decoded_tokens=[0] * gpu_block_size,
        expected_loaded=(0, 1, 2, 3, 4, 5, 6, 7, 8),
    )

    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=(9, 10, 11),
    )


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_concurrent_lookups_of_the_same_prefix(request_runner, async_scheduling: bool):
    offloaded_block_size = 12
    gpu_block_size = 4
    num_gpu_blocks = 100

    runner = request_runner(
        offloaded_block_size=offloaded_block_size,
        gpu_block_size=gpu_block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
    )

    # store 1 blocks
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = lambda keys: generate_store_output(keys)
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=(0, 1, 2),
    )

    # start a request to load the first block, but don't complete
    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.connector_scheduler._maximal_prefix_lookup = lambda bh: 1
    runner.run(
        decoded_tokens=[],
        complete_transfers=False,
    )

    # request triggered a load
    transfer_jobs = list(runner.offloading_spec.handler.transfer_specs)
    assert transfer_jobs

    # start a new request to load the same first block
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.connector_scheduler._maximal_prefix_lookup = lambda bh: 1
    runner.run(
        decoded_tokens=[],
        complete_transfers=False,
    )

    # request did not trigger a load
    assert transfer_jobs == list(runner.offloading_spec.handler.transfer_specs)

    # complete transfers
    runner.manager.prepare_store.side_effect = lambda keys: generate_store_output([])
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_loaded=(0, 1, 2),
    )

    # second request will use the GPU prefix cache
    assert transfer_jobs == list(runner.offloading_spec.handler.transfer_specs)


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_abort_loading_requests(request_runner, async_scheduling: bool):
    offloaded_block_size = 12
    gpu_block_size = 4
    num_gpu_blocks = 100

    runner = request_runner(
        offloaded_block_size=offloaded_block_size,
        gpu_block_size=gpu_block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
    )

    # store 1 blocks
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = lambda keys: generate_store_output(keys)
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=(0, 1, 2),
    )

    # start a request to load the first block, but don't complete
    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.connector_scheduler._maximal_prefix_lookup = lambda key: 1
    runner.run(
        decoded_tokens=[],
        complete_transfers=False,
    )

    # request triggered a load
    transfer_jobs = list(runner.offloading_spec.handler.transfer_specs)
    assert transfer_jobs

    # abort request
    req_id = str(runner.req_id)
    runner.scheduler.finish_requests((req_id,), RequestStatus.FINISHED_ABORTED)

    # verify request is not deleted
    assert req_id in runner.scheduler.requests

    # complete loading request
    runner.run(
        decoded_tokens=[],
        expected_loaded=(0, 1, 2),
    )

    # assert request is deleted
    assert req_id not in runner.scheduler.requests


class TestOffloadingConnectorStats:
    """Tests for OffloadingConnector stats reconstruction and operations."""

    def test_build_kv_connector_stats_with_none(self):
        """Test that build_kv_connector_stats returns empty stats when given None."""
        stats = OffloadingConnector.build_kv_connector_stats(data=None)

        assert stats is not None
        assert isinstance(stats, OffloadingConnectorStats)
        assert len(stats.data) == 0
        assert stats.is_empty()

    def test_build_kv_connector_stats_with_empty_dict(self):
        """Test that build_kv_connector_stats returns empty stats with empty dict."""
        stats = OffloadingConnector.build_kv_connector_stats(data={})

        assert stats is not None
        assert isinstance(stats, OffloadingConnectorStats)
        assert len(stats.data) == 0
        assert stats.is_empty()

    def test_build_kv_connector_stats_reconstructs_offload_stats(self):
        """Test that OffloadingConnector stats are properly reconstructed with
        correct data."""
        serialized_data = {
            "CPU_to_GPU": [
                {"op_size": 16, "op_time": 1.0},
                {"op_size": 8, "op_time": 0.5},
            ],
            "GPU_to_CPU": [
                {"op_size": 1, "op_time": 0.1},
                {"op_size": 2, "op_time": 0.2},
            ],
        }

        stats = OffloadingConnector.build_kv_connector_stats(data=serialized_data)

        offload_connector_stats = stats
        assert isinstance(offload_connector_stats, OffloadingConnectorStats)
        assert offload_connector_stats.data["CPU_to_GPU"] == [
            {"op_size": 16, "op_time": 1.0},
            {"op_size": 8, "op_time": 0.5},
        ]
        assert offload_connector_stats.data["GPU_to_CPU"] == [
            {"op_size": 1, "op_time": 0.1},
            {"op_size": 2, "op_time": 0.2},
        ]

    def test_aggregate_same_connector(self):
        """Test aggregating stats from the same connector type."""
        stats1 = OffloadingConnectorStats(
            data={
                "CPU_to_GPU": [
                    {"op_size": 16, "op_time": 1.0},
                    {"op_size": 8, "op_time": 0.5},
                ],
                "GPU_to_CPU": [
                    {"op_size": 1, "op_time": 0.1},
                    {"op_size": 2, "op_time": 0.2},
                ],
            }
        )

        stats2 = OffloadingConnectorStats(
            data={
                "CPU_to_GPU": [
                    {"op_size": 3, "op_time": 0.2},
                    {"op_size": 7, "op_time": 0.9},
                ],
                "GPU_to_CPU": [{"op_size": 16, "op_time": 2}],
            }
        )

        result = stats1.aggregate(stats2)

        assert result is stats1  # Should return self
        offload_connector_stats = result
        assert offload_connector_stats.data["CPU_to_GPU"] == [
            {"op_size": 16, "op_time": 1.0},
            {"op_size": 8, "op_time": 0.5},
            {"op_size": 3, "op_time": 0.2},
            {"op_size": 7, "op_time": 0.9},
        ]
        assert offload_connector_stats.data["GPU_to_CPU"] == [
            {"op_size": 1, "op_time": 0.1},
            {"op_size": 2, "op_time": 0.2},
            {"op_size": 16, "op_time": 2},
        ]

    def test_reduce(self):
        """Test that reduce() correctly reduces all nested connector stats."""
        stats = OffloadingConnectorStats(
            data={
                "CPU_to_GPU": [
                    {"op_size": 16, "op_time": 1.0},
                    {"op_size": 8, "op_time": 0.5},
                    {"op_size": 3, "op_time": 0.2},
                    {"op_size": 7, "op_time": 0.9},
                ],
                "GPU_to_CPU": [
                    {"op_size": 1, "op_time": 0.1},
                    {"op_size": 2, "op_time": 0.2},
                    {"op_size": 16, "op_time": 2},
                ],
            }
        )

        reduced = stats.reduce()

        assert isinstance(reduced, dict)
        # Check that the stats were reduced (should have aggregated values)
        assert "CPU_to_GPU_total_bytes" in reduced
        assert "CPU_to_GPU_total_time" in reduced
        assert "GPU_to_CPU_total_bytes" in reduced
        assert "GPU_to_CPU_total_time" in reduced
        assert reduced["CPU_to_GPU_total_bytes"] == 34
        assert reduced["CPU_to_GPU_total_time"] == 2.6
        assert reduced["GPU_to_CPU_total_time"] == 2.3
        assert reduced["GPU_to_CPU_total_bytes"] == 19

    def test_reset(self):
        """Test that reset() resets all nested connector stats."""
        offload_connector_stats = OffloadingConnectorStats(
            data={
                "CPU_to_GPU": [
                    {"op_size": 3, "op_time": 0.2},
                    {"op_size": 7, "op_time": 0.9},
                ],
                "GPU_to_CPU": [{"op_size": 16, "op_time": 2}],
            }
        )

        assert not offload_connector_stats.is_empty()

        offload_connector_stats.reset()

        # After reset, stats should be empty
        assert offload_connector_stats.is_empty()
        assert len(offload_connector_stats.data) == 0


# ---------------------------------------------------------------------------
# Unit tests for _maximal_prefix_lookup / _sliding_window_lookup
# ---------------------------------------------------------------------------


def _make_scheduler_with_lookup(
    lookup_results: dict[int, bool | None],
) -> OffloadingConnectorScheduler:
    """Create an OffloadingConnectorScheduler with a mocked manager.lookup."""
    manager = MagicMock(spec=OffloadingManager)
    manager.lookup.side_effect = lambda key: lookup_results.get(
        int(get_offload_block_hash(key).decode()), False
    )

    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler.manager = manager
    return scheduler


class TestMaximalPrefixLookup:
    def test_all_hit(self):
        sched = _make_scheduler_with_lookup({1: True, 2: True})
        assert sched._maximal_prefix_lookup(to_keys([1, 2])) == 2

    def test_all_miss(self):
        sched = _make_scheduler_with_lookup({})
        assert sched._maximal_prefix_lookup(to_keys([1, 2])) == 0

    def test_partial_prefix(self):
        sched = _make_scheduler_with_lookup({1: True, 2: True})
        assert sched._maximal_prefix_lookup(to_keys([1, 2, 3])) == 2

    def test_miss_then_hit(self):
        sched = _make_scheduler_with_lookup({2: True})
        assert sched._maximal_prefix_lookup(to_keys([1, 2])) == 0

    def test_single_hit(self):
        sched = _make_scheduler_with_lookup({1: True})
        assert sched._maximal_prefix_lookup(to_keys([1])) == 1

    def test_empty(self):
        sched = _make_scheduler_with_lookup({})
        assert sched._maximal_prefix_lookup([]) == 0

    def test_none_defers(self):
        sched = _make_scheduler_with_lookup({1: None, 2: True})
        assert sched._maximal_prefix_lookup(to_keys([1, 2])) is None

    def test_none_after_hit_defers(self):
        sched = _make_scheduler_with_lookup({1: True, 2: None})
        assert sched._maximal_prefix_lookup(to_keys([1, 2])) is None

    def test_none_stops_at_miss(self):
        """None is treated as hit for iteration, but miss stops the scan."""
        sched = _make_scheduler_with_lookup({1: None, 2: False, 3: True})
        assert sched._maximal_prefix_lookup(to_keys([1, 2, 3])) is None
        # lookup should have been called for blocks 1 and 2 (stops at miss)
        assert sched.manager.lookup.call_count == 2


class TestSlidingWindowLookup:
    def test_all_hit_exact_window(self):
        sched = _make_scheduler_with_lookup({1: True, 2: True})
        assert sched._sliding_window_lookup(to_keys([1, 2]), 2) == 2

    def test_all_miss(self):
        sched = _make_scheduler_with_lookup({})
        assert sched._sliding_window_lookup(to_keys([1, 2, 3]), 1) == 0

    def test_window_at_end(self):
        sched = _make_scheduler_with_lookup({2: True, 3: True})
        assert sched._sliding_window_lookup(to_keys([1, 2, 3]), 2) == 3

    def test_window_in_middle(self):
        sched = _make_scheduler_with_lookup({2: True, 3: True})
        assert sched._sliding_window_lookup(to_keys([1, 2, 3, 4]), 2) == 3

    def test_no_full_window_falls_back_to_prefix(self):
        sched = _make_scheduler_with_lookup({1: True, 2: True})
        assert sched._sliding_window_lookup(to_keys([1, 2, 3]), 3) == 2

    def test_single_block_window(self):
        sched = _make_scheduler_with_lookup({2: True, 3: True})
        assert sched._sliding_window_lookup(to_keys([1, 2, 3]), 1) == 3

    def test_gap_resets_consecutive(self):
        sched = _make_scheduler_with_lookup({2: True, 3: True, 4: True})
        # [1, 2, 3, 0, 4] — gap at 0 resets, window of 2 found at [2,3]
        assert sched._sliding_window_lookup(to_keys([1, 2, 3, 0, 4]), 2) == 3

    def test_window_prefers_rightmost(self):
        sched = _make_scheduler_with_lookup({1: True, 2: True, 4: True, 5: True})
        # two valid windows: [1,2] at positions 0-1 and [4,5] at positions 3-4
        # scans right-to-left, finds [4,5] first
        assert sched._sliding_window_lookup(to_keys([1, 2, 3, 4, 5]), 2) == 5

    def test_prefix_fallback_with_gap(self):
        sched = _make_scheduler_with_lookup({2: True, 3: True, 4: True, 5: True})
        # window of 4 not found contiguously (gap at 1)
        assert sched._sliding_window_lookup(to_keys([2, 1, 3, 4, 5]), 4) == 1

    def test_empty(self):
        sched = _make_scheduler_with_lookup({})
        assert sched._sliding_window_lookup([], 1) == 0

    def test_none_defers(self):
        sched = _make_scheduler_with_lookup({1: True, 2: None})
        assert sched._sliding_window_lookup(to_keys([1, 2]), 2) is None

    def test_none_with_full_window_still_defers(self):
        """Even if a real window is found after a None, result is deferred."""
        # Scan right-to-left: 4(True), 3(None) resets, 2(True), 1(True) = window
        # but block 3 was None so defer_lookup is set
        sched = _make_scheduler_with_lookup({1: True, 2: True, 3: None, 4: True})
        assert sched._sliding_window_lookup(to_keys([1, 2, 3, 4]), 2) is None
