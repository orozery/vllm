#pragma once

#include <torch/all.h>

#include <map>
#include <vector>

void swap_blocks(torch::Tensor& src, torch::Tensor& dst,
                 const torch::Tensor& block_mapping);

void swap_blocks_multi_layer(const std::vector<torch::Tensor>& src_kv_caches,
                             const std::vector<torch::Tensor>& dst_kv_caches,
                             const torch::Tensor& block_mapping);
