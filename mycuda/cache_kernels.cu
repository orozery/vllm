#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cassert>


template <typename T, typename TENSOR_TYPE>
T* get_kernel_ptr(TENSOR_TYPE& tensor) {
  // Get the kernel-accessible pointer of the given type T
  // Returns NULL if the tensor is on CPU and non-pinned
  torch::Device device = tensor.device();
  if (device.is_cuda()) {
    return static_cast<T*>(tensor.data_ptr());
  } else if (device.is_cpu()) {
    T* ptr;
    auto st = cudaHostGetDevicePointer(
        (void**)&ptr, static_cast<void*>(tensor.data_ptr()), 0);
    TORCH_CHECK(st == cudaSuccess,
                "Host tensor not registered/pinned (or bad ptr)");
    return ptr;
  } else {
    TORCH_CHECK(false, "Invalid device. Device must be cuda or pinned cpu.");
  }
}

__global__ void swap_blocks_kernel(
    int64_t** __restrict__ src_ptrs,
    int64_t** __restrict__ dst_ptrs,
    const int64_t* __restrict__ block_mapping,
    const int qwords_per_block,
    const int k_or_v_stride_qword_size) {
  const int block_id = blockIdx.x;
  const int layer_id = blockIdx.y;
  const int k_or_v = blockIdx.z;
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  const int64_t src_block_number = block_mapping[2 * block_id];
  const int64_t dst_block_number = block_mapping[2 * block_id + 1];
  const int k_or_v_offset = k_or_v == 0 ? 0 : k_or_v_stride_qword_size;
  int64_t* src_ptr = src_ptrs[layer_id] + k_or_v_offset + src_block_number * qwords_per_block;
  int64_t* dst_ptr = dst_ptrs[layer_id] + k_or_v_offset + dst_block_number * qwords_per_block;

  for (int i = tid; i < qwords_per_block; i += num_threads) {
    dst_ptr[i] = src_ptr[i];
  }
}

void swap_blocks_multi_layer(torch::Tensor& src, torch::Tensor& dst,
                 const torch::Tensor& block_mapping,
                 const int block_size_bytes, const int total_layer_bytes) {
  torch::Device src_device = src.device();
  torch::Device dst_device = dst.device();
  const at::cuda::OptionalCUDAGuard device_guard(
      src_device.is_cuda() ? src_device : dst_device);

  int64_t** src_ptrs = get_kernel_ptr<int64_t*, torch::Tensor>(src);
  int64_t** dst_ptrs = get_kernel_ptr<int64_t*, torch::Tensor>(dst);
  const int64_t* block_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(block_mapping.cuda());

  int num_layers = src.size(0);
  int num_blocks = block_mapping.size(0);
  int qwords_per_block = block_size_bytes / 8;

  dim3 grid(num_blocks, num_layers, 2);
  dim3 block(std::min(qwords_per_block, 128));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  swap_blocks_kernel
        <<<grid, block, 0, stream>>>(src_ptrs, dst_ptrs,
                                     block_mapping_ptr,
                                     qwords_per_block,
                                     total_layer_bytes / 2 / 8);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
