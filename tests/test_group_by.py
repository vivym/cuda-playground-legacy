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
        A_.append(torch.randn(m, k, dtype=torch.float32).contiguous().cuda())
        B_.append(torch.randn(k, n, dtype=torch.float32).contiguous().cuda())

    A = VBMatrices(A_)
    for _ in range(10):
        grouped_A, masks = A.group_by()

    times = []
    for _ in range(50):
        start_time = time.time()
        grouped_A, masks = A.group_by()
        times.append(time.time() - start_time)

    print(np.mean(times) * 1000)

    print(grouped_A.data.shape)
    print(masks.shape)

    C = VBMatrices()
    for _ in range(10):
        vbmm(grouped_A, grouped_A, C, 1.0, 0, False, True, VBMMAlgo.Vanilla)
    
    times = []
    for _ in range(50):
        start_time = time.time()
        vbmm(grouped_A, grouped_A, C, 1.0, 0, False, True, VBMMAlgo.Vanilla)
        times.append(time.time() - start_time)

    print(np.mean(times) * 1000)

    C = VBMatrices()
    for _ in range(10):
        vbmm(A, A, C, 1.0, 0, False, True, VBMMAlgo.Vanilla)
    
    times = []
    for _ in range(50):
        start_time = time.time()
        vbmm(A, A, C, 1.0, 0, False, True, VBMMAlgo.Vanilla)
        times.append(time.time() - start_time)

    print(np.mean(times) * 1000)

    A = torch.randn(batch_size, 256, 64, dtype=torch.float32).cuda()

    for _ in range(10):
        C = A @ A.transpose(1, 2)
    
    times = []
    for _ in range(50):
        start_time = time.time()
        C = A @ A.transpose(1, 2)
        times.append(time.time() - start_time)

    print(np.mean(times) * 1000)


if __name__ == "__main__":
    main()
