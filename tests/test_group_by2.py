import random
import time

import numpy as np
import torch

from cuda_playground.ops.vb_matrices import VBMatrices
from cuda_playground.ops.vb_mm import VBMMAlgo, vbmm
from cuda_playground.ops.dp import get_optimal_group_delimeters_wrapper


def get_offsets(tensor):
    offsets = torch.zeros_like(tensor)
    offsets[1:] = tensor[:-1].cumsum(dim=0)
    return offsets


def main():
    random.seed(0)

    batch_size = 10000
    A_, B_ = [], []
    m = []
    for _ in range(batch_size):
        # m.append(random.randint(32, 256))
        m.append(random.randint(1, 9))

    # m.sort()
    for i in range(batch_size):
        A_.append(torch.randn(m[i], 64, dtype=torch.float32).contiguous().cuda())
    
    print(sum(m))

    num_groups = 3
    A = VBMatrices(A_)
    grouped_A, masks = A.group_by(num_groups)
    tmp = masks

    A_m = A.m[:-1].cpu()
    inverse_sorted_indices, padded_m = tmp[:10000], tmp[10000:20000]
    inverse_sorted_indices, padded_m = inverse_sorted_indices.cpu(), padded_m.cpu()
    m_offsets = get_offsets(A_m)
    padded_m_offsets = get_offsets(padded_m)

    # thrust::for_each(
    #     policy,
    #     thrust::make_counting_iterator<index_t>(0),
    #     thrust::make_counting_iterator<index_t>(batch_size),
    #     [=] __device__ (index_t i) {
    #       auto m_offset = unsorted_m_offsets_ptr[i];
    #       auto m = unsorted_m_ptr[i];
    #       auto idx = inverse_sorted_indices_ptr[i];
    #       auto padded_m_offset = padded_m_offsets_ptr[idx];
    #       auto padded_m = padded_m_ptr[idx];
    #       if (m > padded_m) {
    #         printf("fuck! %d %d\n", static_cast<index_t>(m), static_cast<index_t>(padded_m));
    #       }
    #       thrust::sequence(
    #           policy,
    #           indices_ptr + m_offset, indices_ptr + m_offset + m,
    #           static_cast<index_t>(padded_m_offset));
    #     });
    A_m_sorted, indices = A_m.sort()
    print(A_m)
    print(inverse_sorted_indices, inverse_sorted_indices.unique().shape)
    print(indices, indices.unique().shape)

    tmp = torch.gather(torch.arange(0, batch_size, dtype=torch.int32), 0, indices)
    print("tmp == inverse_sorted_indices", (tmp == inverse_sorted_indices).all())
    print(tmp)

    for i in range(batch_size):
        m = A_m[i]
        idx = inverse_sorted_indices[i]
        if m != A_m_sorted[idx]:
            print("fuck", i, m, idx, A_m_sorted[idx])
            return


    for i in range(batch_size):
        m_offset = m_offsets[i]
        m = A_m[i]
        idx = inverse_sorted_indices[i]
        padded_m_offset = padded_m_offsets[idx]
        padded_m_i = padded_m[idx]
        if m > padded_m_i:
            print("fuck!! %d %d" % (m, padded_m_i))

    # C = VBMatrices()
    # vbmm(grouped_A, grouped_A, C, 1.0, 0, False, True, VBMMAlgo.Vanilla)


if __name__ == "__main__":
    main()
