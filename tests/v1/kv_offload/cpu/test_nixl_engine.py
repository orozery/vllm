# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fake-agent tests for the NIXL copy engine prototype.

The engine takes an injected NIXL agent, so these tests exercise the stream
lifecycle, direction handling and descriptor construction without a NIC.
"""

import time

import numpy as np
import pytest

from vllm.v1.kv_offload.cpu.engines.base import CopyBatch
from vllm.v1.kv_offload.cpu.engines.factory import CopyEngineFactory
from vllm.v1.kv_offload.cpu.engines.nixl import NixlCopyEngine, NixlEngineConfig


class FakeAgent:
    """NIXL agent stand-in that copies bytes between registered buffers."""

    def __init__(self) -> None:
        self.xfers: list[dict] = []
        self.released: list[dict] = []
        self.polls_until_done = 1
        self.fail_check = False
        self.fail_setup = False
        self.raise_check = False
        self.transfer_state = "PROC"
        self.buffers: dict[int, np.ndarray] = {}
        self.mem_types: list[str] = []

    def register(self, array: np.ndarray) -> None:
        self.buffers[array.ctypes.data] = array

    def get_xfer_descs(self, descs, mem_type):
        self.mem_types.append(mem_type)
        return list(descs)

    def initialize_xfer(self, op, local, remote, agent):
        if self.fail_setup:
            raise RuntimeError("setup boom")
        handle = {
            "op": op,
            "local": local,
            "remote": remote,
            "polls": self.polls_until_done,
        }
        self.xfers.append(handle)
        return handle

    def transfer(self, handle):
        for (local_addr, local_size, _), (remote_addr, remote_size, _) in zip(
            handle["local"], handle["remote"]
        ):
            local = self.buffers[local_addr][:local_size]
            remote = self.buffers[remote_addr][:remote_size]
            if handle["op"] == "WRITE":
                remote[:] = local
            else:
                local[:] = remote
        return self.transfer_state

    def check_xfer_state(self, handle):
        if self.raise_check:
            raise RuntimeError("state boom")
        if self.fail_check:
            return "ERR"
        if handle["polls"] > 0:
            handle["polls"] -= 1
            return "PROC"
        return "DONE"

    def release_xfer_handle(self, handle):
        self.released.append(handle)


def _engine(
    agent: FakeAgent, *, device_to_cpu: bool, timeout_s: float = 1.0
) -> NixlCopyEngine:
    return NixlCopyEngine(
        NixlEngineConfig(
            agent=agent,
            remote_agent="peer",
            wait_for_compute=lambda: None,
            timeout_s=timeout_s,
        ),
        device_to_cpu=device_to_cpu,
    )


def _batch(src: int, dst: int, size: int) -> CopyBatch:
    return CopyBatch(
        src_ptrs=np.array([src], dtype=np.uint64),
        dst_ptrs=np.array([dst], dtype=np.uint64),
        sizes=np.array([size], dtype=np.int64),
    )


def test_write_moves_device_to_host() -> None:
    agent = FakeAgent()
    device = np.arange(64, dtype=np.uint8)
    host = np.zeros(64, dtype=np.uint8)
    agent.register(device)
    agent.register(host)

    stream = _engine(agent, device_to_cpu=True).new_stream()
    stream.submit(_batch(device.ctypes.data, host.ctypes.data, 64), 1)

    assert not stream.done()
    stream.synchronize()
    np.testing.assert_array_equal(host, device)
    assert stream.done()
    assert stream.elapsed_seconds() >= 0

    assert agent.xfers[0]["op"] == "WRITE"
    assert agent.mem_types == ["VRAM", "DRAM"]
    assert agent.released == agent.xfers


def test_read_moves_host_to_device() -> None:
    agent = FakeAgent()
    device = np.zeros(64, dtype=np.uint8)
    host = np.arange(64, dtype=np.uint8)
    agent.register(device)
    agent.register(host)

    stream = _engine(agent, device_to_cpu=False).new_stream()
    stream.submit(_batch(host.ctypes.data, device.ctypes.data, 64), 1)
    stream.synchronize()

    np.testing.assert_array_equal(device, host)
    assert agent.xfers[0]["op"] == "READ"


def test_stream_can_be_reused() -> None:
    agent = FakeAgent()
    first = np.arange(16, dtype=np.uint8)
    second = np.arange(16, 32, dtype=np.uint8)
    host = np.zeros(16, dtype=np.uint8)
    for array in (first, second, host):
        agent.register(array)

    engine = _engine(agent, device_to_cpu=True)
    stream = engine.new_stream()
    stream.submit(_batch(first.ctypes.data, host.ctypes.data, 16), 1)
    stream.synchronize()
    np.testing.assert_array_equal(host, first)

    stream.submit(_batch(second.ctypes.data, host.ctypes.data, 16), 1)
    stream.synchronize()
    np.testing.assert_array_equal(host, second)
    assert len(agent.released) == 2


def test_error_state_raises_and_releases() -> None:
    agent = FakeAgent()
    agent.fail_check = True
    device = np.zeros(8, dtype=np.uint8)
    host = np.zeros(8, dtype=np.uint8)
    agent.register(device)
    agent.register(host)

    stream = _engine(agent, device_to_cpu=True).new_stream()
    stream.submit(_batch(device.ctypes.data, host.ctypes.data, 8), 1)

    with pytest.raises(RuntimeError, match="NIXL transfer failed"):
        stream.done()
    assert len(agent.released) == 1


def test_submit_error_is_finished() -> None:
    agent = FakeAgent()
    agent.transfer_state = "ERR"
    device = np.zeros(8, dtype=np.uint8)
    host = np.zeros(8, dtype=np.uint8)
    agent.register(device)
    agent.register(host)

    stream = _engine(agent, device_to_cpu=True).new_stream()
    with pytest.raises(RuntimeError, match="posted with ERR"):
        stream.submit(_batch(device.ctypes.data, host.ctypes.data, 8), 1)
    stream.abort()

    with pytest.raises(RuntimeError, match="NIXL transfer failed"):
        stream.done()
    assert len(agent.released) == 1


def test_wait_for_orders_after_previous() -> None:
    agent = FakeAgent()
    agent.polls_until_done = 2
    device = np.zeros(8, dtype=np.uint8)
    host = np.zeros(8, dtype=np.uint8)
    agent.register(device)
    agent.register(host)

    engine = _engine(agent, device_to_cpu=True)
    previous = engine.new_stream()
    previous.submit(_batch(device.ctypes.data, host.ctypes.data, 8), 1)
    following = engine.new_stream()

    following.wait_for(previous)
    assert previous.done()


def test_stuck_transfer_is_fatal() -> None:
    agent = FakeAgent()
    agent.polls_until_done = 1 << 30
    device = np.zeros(8, dtype=np.uint8)
    host = np.zeros(8, dtype=np.uint8)
    agent.register(device)
    agent.register(host)

    stream = _engine(agent, device_to_cpu=True, timeout_s=0.01).new_stream()
    stream.submit(_batch(device.ctypes.data, host.ctypes.data, 8), 1)
    time.sleep(0.05)

    with pytest.raises(TimeoutError):
        stream.done()


def test_setup_error_is_recoverable() -> None:
    agent = FakeAgent()
    agent.fail_setup = True
    device = np.zeros(8, dtype=np.uint8)
    host = np.zeros(8, dtype=np.uint8)
    agent.register(device)
    agent.register(host)

    stream = _engine(agent, device_to_cpu=True).new_stream()
    with pytest.raises(RuntimeError, match="setup failed"):
        stream.submit(_batch(device.ctypes.data, host.ctypes.data, 8), 1)
    with pytest.raises(RuntimeError, match="NIXL transfer failed"):
        stream.done()


def test_state_error_is_fatal() -> None:
    agent = FakeAgent()
    agent.raise_check = True
    device = np.zeros(8, dtype=np.uint8)
    host = np.zeros(8, dtype=np.uint8)
    agent.register(device)
    agent.register(host)

    stream = _engine(agent, device_to_cpu=True).new_stream()
    stream.submit(_batch(device.ctypes.data, host.ctypes.data, 8), 1)
    with pytest.raises(TimeoutError, match="state is unknown"):
        stream.done()


def test_factory_registers_nixl() -> None:
    assert CopyEngineFactory.get_engine("nixl") is NixlCopyEngine
