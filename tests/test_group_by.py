import random
import time

import numpy as np
import torch

from cuda_playground.ops.vb_matrices import VBMatrices
from cuda_playground.ops.vb_mm import VBMMAlgo, vbmm


def main():
    random.seed(0)

    batch_size = 4096
    A_, B_ = [], []
    for _ in range(batch_size):
        m, n, k = random.randint(32, 256), random.randint(32, 256), 64
        # m, n, k = 256, 256, 64
        A_.append(torch.randn(m, k).contiguous().cuda())
        B_.append(torch.randn(k, n).contiguous().cuda())

    A = VBMatrices(A_)
    for _ in range(10):
        indices = A.group_by()

    times = []
    for _ in range(50):
        start_time = time.time()
        indices = A.group_by()
        times.append(time.time() - start_time)

    print(np.mean(times) * 1000)

    print(indices.shape)
    print(A.data.shape)


if __name__ == "__main__":
    main()
