#pragma once

#include <vector>

namespace cuda_playground {

namespace dp {

template <typename index_t>
std::vector<index_t> get_optimal_group_delimeters(index_t* sizes_ptr, index_t batch_size, index_t num_groups);

} // namespace dp

} // namespace cuda_playground
