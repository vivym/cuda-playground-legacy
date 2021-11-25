#include "dp.h"
#include <iostream>
#include <algorithm>
#include <chrono>

namespace cuda_playground {

namespace dp {

constexpr int kInf = 100000000;

template <typename index_t>
std::vector<index_t> get_optimal_group_delimeters(index_t* sizes_ptr, index_t batch_size, index_t num_groups) {
  auto start = std::chrono::steady_clock::now();
  std::vector<index_t> f(batch_size * num_groups);
  std::vector<index_t> d(batch_size * num_groups);
  std::vector<index_t> S(batch_size);
  std::vector<index_t> A(num_groups, 0);

  S[0] = sizes_ptr[0];
  for (index_t i = 1; i < batch_size; ++i) {
    S[i] = S[i - 1] + sizes_ptr[i];
  }

  f[0] = 0;
  d[0] = -1;
  for (index_t k = 1; k < num_groups; k++) {
    f[k] = kInf;
    d[k] = -1;
  }
  index_t total = 0;
  for (index_t i = 1; i < batch_size; i++) {
    f[i * num_groups + 0] = sizes_ptr[i] * (i + 1) - S[i];
    d[i * num_groups + 0] = -1;
    for (index_t k = 1; k < num_groups; k ++) {
      index_t min_f = kInf;
      total += i - A[k];
      for (index_t j = A[k]; j < i; j++) {
        auto value = f[j * num_groups + k - 1] + sizes_ptr[i] * (i - j) - (S[i] - S[j]);
        if (value < min_f) {
          min_f = value;
          d[i * num_groups + k] = j;
          A[k] = j;
        }
      }
      f[i * num_groups + k] = min_f;
    }
  }

  std::vector<index_t> delimeters;
  delimeters.push_back(sizes_ptr[batch_size - 1]);
  index_t pos = d[batch_size * num_groups - 1];
  index_t k = num_groups - 1;
  while (pos != -1 && k >= 0) {
    delimeters.push_back(sizes_ptr[pos]);
    pos = d[pos * num_groups + (--k)];
  }
  std::reverse(delimeters.begin(), delimeters.end());

  auto end = std::chrono::steady_clock::now();

  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "dp: " << elapsed.count() << " " << total << std::endl;

  std::cout << "delimeters:";
  for (index_t i = 0; i < delimeters.size(); i++) {
    std::cout << " " << delimeters[i];
  }
  std::cout << std::endl;

  return delimeters;
}

template
std::vector<int32_t> get_optimal_group_delimeters(int32_t* sizes_ptr, int32_t batch_size, int32_t num_groups);

} // namespace dp

} // namespace cuda_playground
