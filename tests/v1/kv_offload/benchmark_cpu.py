# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
import time

from vllm.attention import get_attn_backend
from vllm.multimodal.inputs import torch
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.worker.cpu_gpu import CpuGpuOffloadingHandler


def main():
    head_size = 64
    num_heads = 8
    num_layers = 16
    gpu_block_size = 16
    cpu_block_size = 16
    blocks_to_copy = 100
    num_blocks = 1000
    gpu_shape = [2, num_blocks, gpu_block_size, num_heads, head_size]
    dtype = torch.float16

    attn_backend = get_attn_backend(
        head_size,
        dtype,
        None,
        gpu_block_size,
        use_mla=False)

    gpu_kv_caches = {}
    attn_backends = {}
    for idx in range(num_layers):
        layer_name = str(idx)
        attn_backends[layer_name] = attn_backend
        gpu_kv_caches[layer_name] = torch.zeros(gpu_shape,
                                                dtype=dtype,
                                                device="cuda:0")

    handler = CpuGpuOffloadingHandler(gpu_block_size, cpu_block_size,
                                      num_blocks, gpu_kv_caches, attn_backends)

    src_idx = list(range(num_blocks))
    dst_idx = list(range(num_blocks))
    dst2_idx = list(range(num_blocks))
    gpu_blocks_to_copy = blocks_to_copy * (cpu_block_size // gpu_block_size)

    total_time = 0.0
    warmup_count = 1
    iter_count = 100
    for i in range(warmup_count + iter_count):
        random.shuffle(src_idx)
        random.shuffle(dst_idx)
        random.shuffle(dst2_idx)

        gpu_to_cpu_spec = (GPULoadStoreSpec(src_idx[:gpu_blocks_to_copy]),
                           CPULoadStoreSpec(dst_idx[:blocks_to_copy]))
        cpu_to_gpu_spec = (CPULoadStoreSpec(dst_idx[:blocks_to_copy]),
                           GPULoadStoreSpec(dst2_idx[:gpu_blocks_to_copy]))

        t = time.time()
        handler.transfer_async(0, gpu_to_cpu_spec)
        while not handler.get_finished():
            time.sleep(0.0001)
        if i > warmup_count:
            total_time += time.time() - t

    bytes_size =  (gpu_kv_caches["0"].element_size() *
                   head_size *
                   num_heads *
                   gpu_block_size *
                   gpu_blocks_to_copy  *
                   num_layers *
                   iter_count * 2)
    print("%.2f GB/s" % (bytes_size / total_time / (1024 * 1024 * 1024)))


if __name__ == "__main__":
    main()
