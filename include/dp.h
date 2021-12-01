#pragma once

#include <vector>
#include <cstdint>

namespace cuda_playground {

namespace dp {

template <typename index_t>
std::vector<index_t> get_optimal_group_delimeters(const index_t* sizes_ptr, index_t batch_size, index_t num_groups);

template <typename index_t>
std::vector<index_t> get_optimal_group_delimeters_2(const index_t* sizes_ptr, index_t batch_size, index_t num_groups);

std::vector<int32_t> get_optimal_group_delimeters_wrapper(const std::vector<int32_t>& sizes, int32_t num_groups);

} // namespace dp

} // namespace cuda_playground
