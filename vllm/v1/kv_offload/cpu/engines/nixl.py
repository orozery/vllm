# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL copy mechanism for the CPU offload worker.

This is a prototype whose purpose is to validate the engine abstraction: it
implements `CopyEngine`/`CopyStream` over NIXL without importing torch, and
takes an injected NIXL agent. The follow-up NIXL PR will supply the real agent
setup, GPU memory registration and stream-ordered submission.

The agent must already have the device ranges and the peer's host region
registered. Transfers are posted as one NIXL request per stream; a posted
request that never reaches a terminal state is treated as fatal (the worker's
poll propagates `TimeoutError`).

Prototype caveat: `wait_for` and `wait_for_compute` block the caller. Do not
select this engine until the follow-up PR replaces them with stream-ordered
submission; `submit_load` must not block.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from vllm.v1.kv_offload.cpu.engines.base import (
    CopyBatch,
    CopyEngine,
    CopyStream,
)

# Stdlib logging keeps this module importable without vllm/torch, so the
# fake-agent tests can run it outside the package.
logger = logging.getLogger(__name__)

_TERMINAL_STATES = ("DONE", "ERR")
_POLL_INTERVAL_S = 1e-4


@dataclass(frozen=True, slots=True)
class NixlEngineConfig:
    """A NIXL agent and the peer that owns the host region.

    `wait_for_compute` blocks until pending device work is visible before a
    transfer is posted. The prototype requires it because the worker's store
    path depends on that fence; the follow-up PR replaces it with
    stream-ordered submission.
    """

    agent: Any
    remote_agent: str
    wait_for_compute: Callable[[], None]
    device_index: int = 0
    local_mem_type: str = "VRAM"
    remote_mem_type: str = "DRAM"
    timeout_s: float = 30.0


class NixlCopyStream(CopyStream):
    """One NIXL request, polled until it reaches DONE or ERR."""

    def __init__(self, engine: NixlCopyEngine):
        self._engine = engine
        self._reset()

    def _reset(self) -> None:
        self._handle: Any = None
        self._state: str | None = None
        self._start = 0.0
        self._end = 0.0
        self._released = False

    def wait_for_compute(self) -> None:
        self._engine.config.wait_for_compute()

    def wait_for(self, previous: CopyStream) -> None:
        # NIXL posts requests asynchronously; serialize explicitly. A failed
        # previous request still orders the next one, so its error is ignored.
        with contextlib.suppress(RuntimeError):
            previous.synchronize()

    def submit(self, batch: CopyBatch, num_ops: int) -> None:
        self._reset()
        self._start = time.monotonic()
        if num_ops == 0:
            self._finish("DONE")
            return

        src = batch.src_ptrs[:num_ops]
        dst = batch.dst_ptrs[:num_ops]
        sizes = batch.sizes[:num_ops]
        if self._engine.device_to_cpu:
            local, remote, op = src, dst, "WRITE"
        else:
            local, remote, op = dst, src, "READ"

        dev = self._engine.config.device_index
        agent = self._engine.config.agent
        local_mem_type = self._engine.config.local_mem_type
        remote_mem_type = self._engine.config.remote_mem_type
        try:
            with self._engine.lock:
                local_descs = agent.get_xfer_descs(
                    [(int(a), int(s), dev) for a, s in zip(local, sizes)],
                    local_mem_type,
                )
                remote_descs = agent.get_xfer_descs(
                    [(int(a), int(s), 0) for a, s in zip(remote, sizes)],
                    remote_mem_type,
                )
                handle = agent.initialize_xfer(
                    op,
                    local_descs,
                    remote_descs,
                    self._engine.config.remote_agent,
                )
        except Exception as e:
            self._finish("ERR")
            raise RuntimeError(f"NIXL transfer setup failed: {e}") from e
        self._handle = handle

        # A raise here may mean the request was posted. Keep the handle and
        # let the worker treat the unknown state as fatal.
        with self._engine.lock:
            state = agent.transfer(handle)
        if state == "ERR":
            self._finish("ERR")
            raise RuntimeError(f"NIXL {op} posted with ERR")

    def abort(self) -> None:
        """Finish a stream whose submit raised, once the request is terminal."""
        if self._handle is None:
            self._finish("ERR")
            return
        state = self._poll()
        if state in _TERMINAL_STATES:
            self._finish(state)

    def done(self) -> bool:
        if self._state is not None:
            if self._state == "ERR":
                raise RuntimeError("NIXL transfer failed")
            return True
        state = self._poll()
        if state == "ERR":
            self._finish("ERR")
            raise RuntimeError("NIXL transfer failed")
        if state == "DONE":
            self._finish("DONE")
        return self._state is not None

    def synchronize(self) -> None:
        if self._wait() == "ERR":
            raise RuntimeError("NIXL transfer failed")

    def elapsed_seconds(self) -> float:
        if self._state is None:
            self._wait()
        return self._end - self._start

    def _poll(self) -> str:
        if self._state is not None:
            return self._state
        if self._handle is None:
            return "ERR"
        if time.monotonic() - self._start > self._engine.config.timeout_s:
            # A request that may still land must stop the engine, not be
            # reported as a failure the scheduler could reuse blocks after.
            raise TimeoutError("NIXL transfer did not finish in time")
        try:
            with self._engine.lock:
                return self._engine.config.agent.check_xfer_state(self._handle)
        except Exception as e:
            # The request may still land; state it cannot be trusted, so this
            # is fatal rather than a failed job.
            raise TimeoutError(f"NIXL transfer state is unknown: {e}") from e

    def _wait(self) -> str:
        while True:
            state = self._poll()
            if state in _TERMINAL_STATES:
                self._finish(state)
                return state
            time.sleep(_POLL_INTERVAL_S)

    def _finish(self, state: str) -> None:
        if self._state is not None:
            return
        self._state = state
        self._end = time.monotonic()
        self._release()

    def _release(self) -> None:
        if self._released or self._handle is None:
            return
        try:
            with self._engine.lock:
                self._engine.config.agent.release_xfer_handle(self._handle)
        except Exception as e:
            logger.warning("NIXL release_xfer_handle failed: %s", e)
        self._released = True
        self._handle = None


class NixlCopyEngine(CopyEngine):
    """NIXL implementation of `CopyEngine`."""

    def __init__(self, config: NixlEngineConfig, *, device_to_cpu: bool):
        self.config = config
        self.device_to_cpu = device_to_cpu
        # NIXL agents run without thread synchronization by default; the
        # prototype serializes every agent call.
        self.lock = threading.Lock()

    def new_stream(self) -> CopyStream:
        return NixlCopyStream(self)

    def new_batch(self, num_ops: int) -> CopyBatch:
        return CopyBatch(
            src_ptrs=np.empty(num_ops, dtype=np.uint64),
            dst_ptrs=np.empty(num_ops, dtype=np.uint64),
            sizes=np.empty(num_ops, dtype=np.int64),
        )
