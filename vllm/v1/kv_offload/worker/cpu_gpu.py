# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import deque

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.attention import AttentionBackend
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.kv_offload.mediums import BlockIDsLoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

logger = init_logger(__name__)


def expand_block_ids(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    skip_count: int = 0,
):
    """
    Convert a list of block IDs to a list of matching block ids,
    assuming each block is composed of actual block_size_factor blocks.
    Outputs to output tensor.
    The first skip_count blocks will be skipped.
    Note that skip_count must be less than block_size_factor.

    For example, if block_ids = [0, 1, 3] and block_size_factor =  4,
    then it yields [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]
    since 0 maps to [0, 1, 2, 3]
    1 maps to [4, 5, 6, 7]
    and 3 maps to [12, 13, 14, 15]
    """
    assert skip_count < block_size_factor

    first_range = np.arange(skip_count, block_size_factor)
    full_range = np.arange(0, block_size_factor)

    output_idx = 0
    for i, block_id in enumerate(block_ids):
        base_block_id = block_id * block_size_factor
        indices = first_range if i == 0 else full_range
        output_end_idx = output_idx + len(indices)
        output[output_idx:output_end_idx] = base_block_id + indices
        output_idx = output_end_idx


class SingleDirectionOffloadingHandler(OffloadingHandler):
    """
    SingleDirectionOffloadingHandler handles transfers for a single direction,
    either CPU->GPU or GPU->CPU.
    It submits transfers to a single CUDA stream, and track their completions
    using CUDA events.
    The transfers are gradually submitted to the CUDA stream so that
    the number of pending operations (cudaMemcpyAsync) on a stream will not
    exceed a limit. This is to ensure the queue of the CUDA stream does not
    get full, which may cause the transfer submission to block until
    the stream queue has enough capacity.
    """

    # maximal number of allowed pending ops per CUDA stream
    STREAM_PENDING_OPS_COUNT_LIMIT = 1000

    def __init__(
            self,
            src_tensors: list[torch.Tensor],
            dst_tensors: list[torch.Tensor],
            kv_dim_before_num_blocks: list[bool],
            src_block_size_factor: int,
            dst_block_size_factor: int,
    ):
        """
        Initialize a SingleDirectionOffloadingHandler.

        Args:
            src_tensors: list of KV cache tensors to copy from.
            dst_tensors: list of KV cache tensors to copy to.
                Order should match src_tensors.
            kv_dim_before_num_blocks: list of bools, indicating
                whether the respective KV cache tensor has a KV
                dimension before its num_blocks dimension.
                e.g. (2, num_blocks, ...)
            src_block_size_factor: The number of kernel blocks
                per KV block in a source tensor.
            dst_block_size_factor: The number of kernel blocks
                per KV block in a destination tensor.
        """
        self.src_tensors: list[torch.Tensor] = src_tensors
        self.dst_tensors: list[torch.Tensor] = dst_tensors
        self.kv_dim_before_num_blocks: list[bool] = kv_dim_before_num_blocks
        self.src_block_size_factor: int = src_block_size_factor
        self.dst_block_size_factor: int = dst_block_size_factor

        self._stream = torch.cuda.Stream()
        # job_id -> TransferSpec
        self._transfer_spec: dict[int, TransferSpec] = {}
        # queue of (job ID, op_count) waiting to be submitted to the CUDA stream
        self._waiting_transfers: deque[tuple[int, int]] = deque()
        # queue of (job ID, op_count, CUDA event) transfers submitted to the CUDA stream
        self._running_transfers: deque[tuple[int, int, torch.Event]] = deque()
        # list of job_ids that have finished but not yet reported
        self._finished_transfers: list[int] = []
        # number of ops that were submitted to the CUDA stream
        # and have not yet completed
        self._pending_ops_count = 0
        # list of cuda events available for re-use
        self._events_pool: list[torch.Event] = []
        # for each layer, count 1 op if (num_blocks, 2, ...) and 2 ops if (2, num_blocks, ...)
        self._num_ops_per_block = (len(self.kv_dim_before_num_blocks) +
                                   self.kv_dim_before_num_blocks.count(True))

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec):
        _, dst_spec = transfer_spec
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)
        num_blocks = dst_spec.block_ids.size * self.dst_block_size_factor
        op_count = num_blocks * self._num_ops_per_block
        self._transfer_spec[job_id] = transfer_spec
        self._waiting_transfers.append((job_id, op_count))
        self._check_stream_completions_and_submit_transfers()

    def get_finished(self) -> list[TransferResult]:
        self._check_stream_completions_and_submit_transfers()
        results: list[TransferResult] = []
        for job_id in self._finished_transfers:
            results.append((job_id, True))
        self._finished_transfers.clear()
        return results

    def _check_stream_completions_and_submit_transfers(self):
        # check for completed transfers
        while self._running_transfers:
            job_id, op_count, event = self._running_transfers[0]
            if not event.query():
                break

            self._pending_ops_count -= op_count
            self._finished_transfers.append(job_id)
            self._running_transfers.popleft()

        # try to submit new transfers to stream
        while self._waiting_transfers:
            job_id, op_count = self._waiting_transfers[0]
            new_pending_ops_count = self._pending_ops_count + op_count
            if (self._pending_ops_count == 0 or
                    (new_pending_ops_count >
                     SingleDirectionOffloadingHandler.STREAM_PENDING_OPS_COUNT_LIMIT)):
                break

            transfer_spec = self._transfer_spec.pop(job_id)
            event = self._submit_single_transfer_to_stream(transfer_spec)
            self._pending_ops_count = new_pending_ops_count
            self._waiting_transfers.popleft()
            self._running_transfers.append((job_id, op_count, event))

    def _submit_single_transfer_to_stream(self, transfer_spec: TransferSpec) -> torch.Event:
        src_spec, dst_spec = transfer_spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        src_sub_block_count = src_blocks.size * self.src_block_size_factor
        dst_sub_block_count = dst_blocks.size * self.dst_block_size_factor
        src_sub_blocks_to_skip = -dst_blocks.size % self.src_block_size_factor

        assert dst_sub_block_count == src_sub_block_count - src_sub_blocks_to_skip

        src_to_dst = np.empty((dst_sub_block_count, 2), dtype=np.int64)
        expand_block_ids(
            src_blocks,
            self.src_block_size_factor,
            src_to_dst[:, 0],
            skip_count=src_sub_blocks_to_skip,
        )
        expand_block_ids(dst_blocks, self.dst_block_size_factor, src_to_dst[:, 1])
        src_to_dst_tensor = torch.from_numpy(src_to_dst)

        event = self._events_pool.pop() if self._events_pool else torch.Event()
        with torch.cuda.stream(self._stream):
            for src_tensor, dst_tensor, kv_dim in zip(
                    self.src_tensors, self.dst_tensors, self.kv_dim_before_num_blocks
            ):
                if kv_dim:
                    src_key_cache = src_tensor[0]
                    dst_key_cache = dst_tensor[0]
                    ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst_tensor)
                    src_value_cache = src_tensor[1]
                    dst_value_cache = dst_tensor[1]
                    ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst_tensor)
                else:
                    ops.swap_blocks(src_tensor, dst_tensor, src_to_dst_tensor)
            event.record(self._stream)

        return event


class CpuGpuOffloadingHandlers:
    def __init__(
        self,
        gpu_block_size: int,
        cpu_block_size: int,
        num_cpu_blocks: int,
        gpu_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ):
        assert gpu_caches
        assert cpu_block_size % gpu_block_size == 0
        block_size_factor = cpu_block_size // gpu_block_size

        pin_memory = is_pin_memory_available()

        # allocate cpu tensors
        logger.info("Allocating %d CPU tensors...", len(gpu_caches))
        gpu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        kv_dim_before_num_blocks: list[bool] = []
        kernel_block_size: int | None = None
        for layer_name, gpu_tensor in gpu_caches.items():
            gpu_tensors.append(gpu_tensor)

            gpu_shape = gpu_tensor.shape
            attn_backend = attn_backends[layer_name]
            test_shape = attn_backend.get_kv_cache_shape(
                num_blocks=1234, block_size=16, num_kv_heads=8, head_size=256
            )

            has_layers_dim = False
            if len(gpu_shape) != len(test_shape):
                # cross-layers tensor
                # shape is (num_blocks, ...)
                assert len(gpu_shape) == len(test_shape) + 1
                num_blocks_idx = 0
                has_layers_dim = True
                kv_dim_before_num_blocks.append(False)

                # prepend a dummy num_layers=80 to test_shape
                test_shape = (80,) + test_shape
            elif test_shape[0] == 1234:
                # shape is (num_blocks, ...)
                num_blocks_idx = 0
                kv_dim_before_num_blocks.append(False)
            else:
                # shape should be (2, num_blocks, ...)
                assert test_shape[0] == 2
                assert test_shape[1] == 1234
                assert gpu_shape[0] == 2

                num_blocks_idx = 1
                kv_dim_before_num_blocks.append(True)

            try:
                kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                    include_num_layers_dimension=has_layers_dim
                )
                assert len(kv_cache_stride_order) == len(gpu_shape)
            except (AttributeError, NotImplementedError):
                kv_cache_stride_order = tuple(range(len(gpu_shape)))

            # permute test_shape according to stride_order
            test_shape = tuple(test_shape[i] for i in kv_cache_stride_order)

            # find block_size (16) dimension index
            block_size_idx = test_shape.index(16)
            if kernel_block_size is not None:
                assert kernel_block_size == gpu_shape[block_size_idx]
            else:
                kernel_block_size = gpu_shape[block_size_idx]
                assert gpu_block_size % kernel_block_size == 0

            cpu_shape = list(gpu_shape)
            cpu_shape[num_blocks_idx] = num_cpu_blocks * block_size_factor

            logger.debug("Allocating CPU tensor of shape %r", cpu_shape)
            cpu_tensors.append(
                torch.zeros(
                    cpu_shape,
                    dtype=gpu_tensor.dtype,
                    device="cpu",
                    pin_memory=pin_memory,
                )
            )

        assert kernel_block_size is not None
        gpu_block_size_factor = gpu_block_size // kernel_block_size
        cpu_block_size_factor = cpu_block_size // kernel_block_size

        # TODO (orozery): adapt swap_blocks to support gpu_block_size_factor
        assert gpu_block_size_factor == 1

        self.gpu_to_cpu_handler = SingleDirectionOffloadingHandler(
            src_tensors=gpu_tensors,
            dst_tensors=cpu_tensors,
            kv_dim_before_num_blocks=kv_dim_before_num_blocks,
            src_block_size_factor=gpu_block_size_factor,
            dst_block_size_factor=cpu_block_size_factor
        )

        self.cpu_to_gpu_handler = SingleDirectionOffloadingHandler(
            src_tensors=cpu_tensors,
            dst_tensors=gpu_tensors,
            kv_dim_before_num_blocks=kv_dim_before_num_blocks,
            src_block_size_factor=cpu_block_size_factor,
            dst_block_size_factor=gpu_block_size_factor
        )
