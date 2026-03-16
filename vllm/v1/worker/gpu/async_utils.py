# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from dataclasses import dataclass

import numpy as np
import torch

from vllm.v1.outputs import AsyncModelRunnerOutput, LogprobsTensors, ModelRunnerOutput
from vllm.v1.worker.gpu.sample.output import SamplerOutput


@dataclass
class D2HCopyBuffers:
    """Pre-allocated pinned+UVA buffers for async D2H copies.

    When UVA copies are enabled, these buffers allow Triton SM-based
    copies instead of DMA engine copies, freeing the copy engine for
    KV cache offloading.
    """

    # [max_num_reqs, max_gen_tokens] for sampled_token_ids
    sampled_token_ids_pinned: torch.Tensor
    sampled_token_ids_uva: torch.Tensor
    # [max_num_reqs] for num_sampled_tokens
    num_sampled_tokens_pinned: torch.Tensor
    num_sampled_tokens_uva: torch.Tensor
    # [max_num_reqs] for num_nans (optional, int32)
    num_nans_pinned: torch.Tensor
    num_nans_uva: torch.Tensor

    @staticmethod
    def create(
        max_num_reqs: int,
        max_gen_tokens: int = 1,
    ) -> "D2HCopyBuffers":
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        sampled_pinned = torch.empty(
            (max_num_reqs, max_gen_tokens),
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        num_sampled_pinned = torch.empty(
            max_num_reqs, dtype=torch.int32, device="cpu", pin_memory=True
        )
        num_nans_pinned = torch.empty(
            max_num_reqs, dtype=torch.int32, device="cpu", pin_memory=True
        )
        return D2HCopyBuffers(
            sampled_token_ids_pinned=sampled_pinned,
            sampled_token_ids_uva=get_accelerator_view_from_cpu_tensor(
                sampled_pinned
            ),
            num_sampled_tokens_pinned=num_sampled_pinned,
            num_sampled_tokens_uva=get_accelerator_view_from_cpu_tensor(
                num_sampled_pinned
            ),
            num_nans_pinned=num_nans_pinned,
            num_nans_uva=get_accelerator_view_from_cpu_tensor(num_nans_pinned),
        )


class AsyncOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
        copy_event: torch.cuda.Event,
        d2h_buffers: D2HCopyBuffers | None = None,
    ):
        # NOTE(woosuk): We must retain references to the GPU tensors,
        # as the copy operations are performed on a different CUDA stream than
        # the one where the tensors were created.
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.copy_event = copy_event

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)

            self.sampled_token_ids = async_copy_to_np(
                sampler_output.sampled_token_ids,
                pinned_buf=d2h_buffers.sampled_token_ids_pinned
                if d2h_buffers
                else None,
                uva_buf=d2h_buffers.sampled_token_ids_uva
                if d2h_buffers
                else None,
            )
            self.logprobs_tensors: LogprobsTensors | None = None
            if sampler_output.logprobs_tensors is not None:
                self.logprobs_tensors = (
                    sampler_output.logprobs_tensors.to_cpu_nonblocking()
                )
            self.num_nans: np.ndarray | None = None
            if sampler_output.num_nans is not None:
                self.num_nans = async_copy_to_np(
                    sampler_output.num_nans,
                    pinned_buf=d2h_buffers.num_nans_pinned
                    if d2h_buffers
                    else None,
                    uva_buf=d2h_buffers.num_nans_uva if d2h_buffers else None,
                )
            self.num_sampled_tokens_np = async_copy_to_np(
                num_sampled_tokens,
                pinned_buf=d2h_buffers.num_sampled_tokens_pinned
                if d2h_buffers
                else None,
                uva_buf=d2h_buffers.num_sampled_tokens_uva
                if d2h_buffers
                else None,
            )
            self.prompt_logprobs_dict = {
                k: v.to_cpu_nonblocking() if v is not None else None
                for k, v in self.model_runner_output.prompt_logprobs_dict.items()
            }
            self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        self.copy_event.synchronize()

        # NOTE(woosuk): The following code is to ensure compatibility with
        # the existing model runner.
        # Going forward, we should keep the data structures as NumPy arrays
        # rather than Python lists.
        sampled_token_ids: list[list[int]] = self.sampled_token_ids.tolist()
        num_sampled_tokens: list[int] = self.num_sampled_tokens_np.tolist()
        for token_ids, num_tokens in zip(sampled_token_ids, num_sampled_tokens):
            del token_ids[num_tokens:]
        self.model_runner_output.sampled_token_ids = sampled_token_ids

        if self.num_nans is not None:
            self.model_runner_output.num_nans_in_logits = dict(
                zip(self.model_runner_output.req_ids, self.num_nans.tolist())
            )

        if self.logprobs_tensors is not None:
            self.model_runner_output.logprobs = self.logprobs_tensors.tolists()
        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict
        return self.model_runner_output


class AsyncPoolingOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        pooler_output: torch.Tensor,
        is_valid: torch.Tensor | None,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
        copy_event: torch.cuda.Event,
    ):
        self.model_runner_output = model_runner_output
        self.pooler_output = pooler_output
        self.is_valid = is_valid
        self.copy_event = copy_event

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)
            self.pooler_output_cpu = self.pooler_output.to("cpu", non_blocking=True)
            if self.is_valid is not None:
                self.is_valid_cpu = self.is_valid.to("cpu", non_blocking=True)
            else:
                self.is_valid_cpu = None
            self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        pooler_output = list(self.pooler_output_cpu.unbind(dim=0))
        self.copy_event.synchronize()
        if self.is_valid_cpu is not None:
            is_valid_cpu = self.is_valid_cpu.tolist()
            for i, is_valid in enumerate(is_valid_cpu):
                if not is_valid:
                    pooler_output[i] = None
        self.model_runner_output.pooler_output = pooler_output
        return self.model_runner_output


def async_copy_to_np(
    x: torch.Tensor,
    pinned_buf: torch.Tensor | None = None,
    uva_buf: torch.Tensor | None = None,
) -> np.ndarray:
    """Copy a GPU tensor to CPU as a numpy array.

    When uva_buf is provided (a UVA view of pinned_buf), uses a Triton
    SM-based copy instead of the DMA copy engine.
    """
    if uva_buf is not None and pinned_buf is not None:
        from vllm.v1.worker.gpu.buffer_utils import uva_copy

        # Handle both 1D [N] and 2D [N, M] tensors.
        # We slice the buffer to match x's shape along dim 0.
        n = x.shape[0]
        src = x
        dst = uva_buf[:n] if uva_buf.dim() > 0 else uva_buf
        if x.dim() == 2 and uva_buf.dim() == 2:
            dst = uva_buf[:n, : x.shape[1]]
        elif x.dim() == 1:
            dst = uva_buf[:n]
        uva_copy(src, dst)
        pinned_slice = pinned_buf[:n]
        if x.dim() == 2 and pinned_buf.dim() == 2:
            pinned_slice = pinned_buf[:n, : x.shape[1]]
        return pinned_slice.numpy()
    return x.to("cpu", non_blocking=True).numpy()


@contextlib.contextmanager
def stream(to_stream: torch.cuda.Stream, from_stream: torch.cuda.Stream):
    """Lightweight version of torch.cuda.stream() context manager which
    avoids current_stream and device lookups.
    """
    try:
        torch.cuda.set_stream(to_stream)
        yield
    finally:
        torch.cuda.set_stream(from_stream)
