#include "dp.h"
#include <iostream>
#include <algorithm>
#include <chrono>

namespace cuda_playground {

namespace dp {

constexpr int kInf = 100000000;

template <typename index_t>
std::vector<index_t> get_optimal_group_delimeters(const index_t* sizes_ptr, index_t batch_size, index_t num_groups) {
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
std::vector<int32_t> get_optimal_group_delimeters(const int32_t* sizes_ptr, int32_t batch_size, int32_t num_groups);

namespace {
  template <typename index_t>
  struct Point {
    index_t x;
    index_t y;

    Point(index_t x, index_t y) : x(x), y(y) {}
  };

  template <typename index_t>
  class MonotonicQueue {
  public:
    void push(Point<index_t> point) {

    }

    const Point<index_t>& back() {
      return points_.back();
    }

  private:
    std::vector<Point<index_t>> points_;
  };
}

template <typename index_t>
std::vector<index_t> get_optimal_group_delimeters_2(const index_t* sizes_ptr, index_t batch_size, index_t num_groups) {
  auto start = std::chrono::steady_clock::now();
  std::vector<index_t> f(batch_size * num_groups);
  std::vector<index_t> d(batch_size * num_groups);
  std::vector<index_t> S(batch_size);
  std::vector<std::vector<Point<index_t>>> queues(num_groups);
  std::vector<index_t> curs(num_groups);

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

  auto check = [&](index_t i, index_t j, index_t k, index_t& min_f) {
    auto value = f[j * num_groups + k - 1] + sizes_ptr[i] * (i - j) - (S[i] - S[j]);
    if (value < min_f) {
      min_f = value;
      d[i * num_groups + k] = j;
    }
  };

  auto enqueue = [&](index_t k, Point<index_t> p) {
    // if (p.y >= kInf) {
    //   return;
    // }

    auto& queue = queues[k];
    
    if (queue.size() < 2) {
      queue.push_back(p);
    } else {
      while (queue.size() >= 2) {
        auto p2 = queue.back();
        auto p1 = queue[queue.size() - 2];
        auto slope1 = (p2.y - p1.y) / (p2.x - p1.x);
        auto slope2 = (p.y - p2.y) / (p.x - p2.x);
        if (slope1 > slope2) {
          queue.pop_back();
        } else {
          break;
        }
      }
      queue.push_back(p);
    }

    // std::cout << "queue " << k << ":";
    // for (auto p : queue) {
    //   std::cout << " (" << p.x << ", " << p.y << ")";
    // }
    // std::cout << std::endl;

    // std::cout << "cur " << k << ": " << curs[k] << std::endl;
  };

  for (index_t i = 1; i < batch_size; i++) {
    // f[i][0]
    f[i * num_groups + 0] = sizes_ptr[i] * (i + 1) - S[i];
    d[i * num_groups + 0] = -1;
    enqueue(0, Point(i, f[i * num_groups + 0] + S[i]));

    for (index_t k = 1; k < num_groups; k ++) {
      index_t min_f = kInf;

      auto& queue = queues[k - 1];
      auto& cur = curs[k - 1];
      if (queue.size() < 3) {
        for (index_t j = 0; j < i; j ++) {
          check(i, j, k, min_f);
        }
      } else {
        auto slope = sizes_ptr[i];
        while (true) {
          auto p1 = queue[cur - 1];
          auto p2 = queue[cur];
          auto p3 = (cur + 1 < queue.size()) ? queue[cur + 1] : Point(batch_size, kInf);
          // std::cout << "hhhh: " << cur << "  slope: " << slope << " " << (p2.y - p1.y) / (p2.x - p1.x) << " " << (p3.y - p2.y) / (p3.x - p2.x) << std::endl;
          // std::cout << p1.x << " " << p1.y << " " << p2.x << " " << p2.y << " " << p3.x << " " << p3.y << std::endl;
          if ((p2.y - p1.y) / (p2.x - p1.x) <= slope && (p3.y - p2.y) / (p3.x - p2.x) >= slope) {
            if (p1.x >= 0 && p1.x < batch_size) {
              check(i, p1.x, k, min_f);
            }
            if (p2.x >= 0 && p2.x < batch_size) {
              check(i, p2.x, k, min_f);
            }
            if (p3.x >= 0 && p3.x < batch_size) {
              check(i, p3.x, k, min_f);
            }
            break;
          }
          cur++;
        }
      }

      f[i * num_groups + k] = min_f;

      enqueue(k, Point(i, f[i * num_groups + k] + S[i]));
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
  std::cout << "dp: " << elapsed.count() << std::endl;

  std::cout << "delimeters:";
  for (index_t i = 0; i < delimeters.size(); i++) {
    std::cout << " " << delimeters[i];
  }
  std::cout << std::endl;

  return delimeters;
}

template
std::vector<int32_t> get_optimal_group_delimeters_2(const int32_t* sizes_ptr, int32_t batch_size, int32_t num_groups);

std::vector<int32_t> get_optimal_group_delimeters_wrapper(const std::vector<int32_t>& sizes, int32_t num_groups) {
  return get_optimal_group_delimeters_2<int32_t>(sizes.data(), sizes.size(), num_groups);
}

} // namespace dp

} // namespace cuda_playground
