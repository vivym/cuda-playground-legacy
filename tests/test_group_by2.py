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

    m.sort()
    for i in range(batch_size):
        A_.append(torch.randn(m[i], 2, dtype=torch.float32).contiguous().cuda())
    
    print(sum(m))

    num_groups = 2
    A = VBMatrices(A_)
    grouped_A, masks = A.group_by(num_groups)

    print(grouped_A.data.shape)
    print(masks.shape)
    print(grouped_A.data[masks].shape)
    data = grouped_A.data[masks]

    print((data == A.data).all())

    # C = VBMatrices()
    # vbmm(grouped_A, grouped_A, C, 1.0, 0, False, True, VBMMAlgo.Vanilla)


if __name__ == "__main__":
    main()
