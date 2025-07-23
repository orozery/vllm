# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional

from vllm.v1.offloading.abstract import LoadStoreSpec, PrepareStoreOutput
from vllm.v1.offloading.cpu_manager import CPUBackend
from vllm.v1.offloading.lru_manager import LRUOffloadingManager
from vllm.v1.offloading.mediums import CPULoadStoreSpec


@dataclass
class ExpectedPrepareStoreOutput:
    block_hashes_to_store: list[int]
    store_block_ids: list[int]
    block_hashes_evicted: list[int]


def verify_store_output(
        prepare_store_output: Optional[PrepareStoreOutput],
        expected_prepare_store_output: ExpectedPrepareStoreOutput):
    assert prepare_store_output is not None
    assert (prepare_store_output.block_hashes_to_store ==
            expected_prepare_store_output.block_hashes_to_store)
    assert (prepare_store_output.block_hashes_evicted ==
            expected_prepare_store_output.block_hashes_evicted)
    assert (len(prepare_store_output.store_specs) == len(
        expected_prepare_store_output.store_block_ids))
    for store_spec, expected_store_block_id in zip(
            prepare_store_output.store_specs,
            expected_prepare_store_output.store_block_ids):
        assert isinstance(store_spec, CPULoadStoreSpec)
        assert store_spec.block_id == expected_store_block_id


def verify_load_output(prepare_load_output: list[LoadStoreSpec],
                       expected_prepare_load_output: list[int]):
    for load_spec, expected_block_id in zip(prepare_load_output,
                                            expected_prepare_load_output):
        assert isinstance(load_spec, CPULoadStoreSpec)
        assert load_spec.block_id == expected_block_id


def test_cpu_manager():
    """
    Tests LRUOffloadingManager with a CPUBackend.
    """
    # initialize a CPU backend with a capacity of 4 blocks
    cpu_backend = CPUBackend(num_blocks=4)
    cpu_manager = LRUOffloadingManager(cpu_backend)

    # prepare store [1, 2]
    prepare_store_output = cpu_manager.prepare_store([1, 2])
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            block_hashes_to_store=[1, 2],
            store_block_ids=[0, 1],
            block_hashes_evicted=[],
        ))

    # lookup [1, 2] -> not ready
    assert cpu_manager.lookup([1, 2]) == 0

    # complete store [1, 2]
    cpu_manager.complete_store([1, 2])

    # lookup [1, 2]
    assert cpu_manager.lookup([1]) == 1
    assert cpu_manager.lookup([1, 2]) == 2
    assert cpu_manager.lookup([1, 2, 3]) == 2

    # prepare store [2, 3, 4, 5] -> evicts [1]
    prepare_store_output = cpu_manager.prepare_store([2, 3, 4, 5])
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            block_hashes_to_store=[3, 4, 5],
            store_block_ids=[2, 3, 0],
            block_hashes_evicted=[1],
        ))

    # prepare store with no space
    assert cpu_manager.prepare_store([1, 6]) is None

    # complete store [2, 3, 4, 5]
    cpu_manager.complete_store([2, 3, 4, 5])

    # prepare load [2, 3]
    prepare_load_output = cpu_manager.prepare_load([2, 3])
    verify_load_output(prepare_load_output, [1, 2])

    # prepare store with no space ([2, 3] is being loaded)
    assert cpu_manager.prepare_store([6, 7, 8]) is None

    # complete load [2, 3]
    cpu_manager.complete_load([2, 3])

    # prepare store [6, 7, 8] -> evicts [2, 3, 4] (oldest)
    prepare_store_output = cpu_manager.prepare_store([6, 7, 8])
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            block_hashes_to_store=[6, 7, 8],
            store_block_ids=[3, 2, 1],
            block_hashes_evicted=[2, 3, 4],
        ))

    # complete store [6, 7, 8]
    cpu_manager.complete_store([6, 7, 8])

    # touch [5, 6, 7] (move to end of LRU order)
    cpu_manager.touch([5, 6, 7])

    # prepare store [7, 9] -> evicts [8] (oldest following previous touch)
    prepare_store_output = cpu_manager.prepare_store([9])
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            block_hashes_to_store=[9],
            store_block_ids=[1],
            block_hashes_evicted=[8],
        ))

    # complete store [7, 9] with failure
    cpu_manager.complete_store([7, 9], is_success=False)

    # assert [7] is still stored, but [9] is not
    assert cpu_manager.lookup([7]) == 1
    assert cpu_manager.lookup([9]) == 0
