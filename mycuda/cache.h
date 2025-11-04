#pragma once

#include <torch/all.h>

void swap_blocks_multi_layer(torch::Tensor& src,
                             torch::Tensor& dst,
                             const torch::Tensor& block_mapping,
                             const int block_size_bytes,
                             const int total_layer_bytes);
