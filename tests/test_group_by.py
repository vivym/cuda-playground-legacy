import random
import time

import numpy as np
import torch

from cuda_playground.ops.vb_matrices import VBMatrices
from cuda_playground.ops.vb_mm import VBMMAlgo, vbmm
from cuda_playground.ops.dp import get_optimal_group_delimeters_wrapper


def main_():
    random.seed(0)

    batch_size = 4096
    A_, B_ = [], []
    m = []
    for _ in range(batch_size):
        m.append(random.randint(32, 256))
    # m.sort()
    for i in range(batch_size):
        # m, n, k = random.randint(32, 256), random.randint(32, 256), 64
        # m, n, k = 256, 256, 64
        # A_.append(torch.randn(m, k, dtype=torch.float32).contiguous().cuda())
        # B_.append(torch.randn(k, n, dtype=torch.float32).contiguous().cuda())
        A_.append(torch.randn(m[i], 64, dtype=torch.float32).contiguous().cuda())

    num_groups = 6
    A = VBMatrices(A_)
    for _ in range(10):
        grouped_A, masks = A.group_by(num_groups)

    times = []
    torch.cuda.synchronize()
    for _ in range(50):
        start_time = time.time()
        grouped_A, masks = A.group_by(num_groups)
        torch.cuda.synchronize()
        times.append(time.time() - start_time)

    print("group_by", np.mean(times) * 1000)

    print(grouped_A.data.shape)
    print(masks.shape)

    C = VBMatrices()
    for _ in range(10):
        vbmm(grouped_A, grouped_A, C, 1.0, 0, False, True, VBMMAlgo.Vanilla)
    
    times = []
    torch.cuda.synchronize()
    for _ in range(50):
        start_time = time.time()
        vbmm(grouped_A, grouped_A, C, 1.0, 0, False, True, VBMMAlgo.Vanilla)
        torch.cuda.synchronize()
        times.append(time.time() - start_time)

    print("grouped vanilla", np.mean(times) * 1000)

    C = VBMatrices()
    for _ in range(10):
        vbmm(grouped_A, grouped_A, C, 1.0, 0, False, True, VBMMAlgo.Stream)
    
    times = []
    torch.cuda.synchronize()
    for _ in range(50):
        start_time = time.time()
        vbmm(grouped_A, grouped_A, C, 1.0, 0, False, True, VBMMAlgo.Stream)
        torch.cuda.synchronize()
        times.append(time.time() - start_time)

    print("grouped stream", np.mean(times) * 1000)

    C = VBMatrices()
    for _ in range(10):
        vbmm(A, A, C, 1.0, 0, False, True, VBMMAlgo.Vanilla)
    
    times = []
    torch.cuda.synchronize()
    for _ in range(50):
        start_time = time.time()
        vbmm(A, A, C, 1.0, 0, False, True, VBMMAlgo.Vanilla)
        torch.cuda.synchronize()
        times.append(time.time() - start_time)

    print("vanilla", np.mean(times) * 1000)

    C = VBMatrices()
    for _ in range(10):
        vbmm(A, A, C, 1.0, 0, False, True, VBMMAlgo.MAGMA)
    
    times = []
    torch.cuda.synchronize()
    for _ in range(50):
        start_time = time.time()
        vbmm(A, A, C, 1.0, 0, False, True, VBMMAlgo.MAGMA)
        torch.cuda.synchronize()
        times.append(time.time() - start_time)

    print("magma", np.mean(times) * 1000)

    A = torch.randn(batch_size, 256, 64, dtype=torch.float32).cuda()

    torch.cuda.synchronize()
    for _ in range(10):
        C = A @ A.transpose(1, 2)
    
    torch.cuda.synchronize()
    times = []
    for _ in range(50):
        start_time = time.time()
        C = A @ A.transpose(1, 2)
        torch.cuda.synchronize()
        times.append(time.time() - start_time)

    print("full padding:", np.mean(times) * 1000)


def main():
    random.seed(0)

    batch_size = 4096
    m = []
    for _ in range(batch_size):
        m.append(random.randint(32, 256))

    m.sort()
    # print(m)

    for _ in range(10):
        get_optimal_group_delimeters_wrapper(m, 6)

    times = []
    for _ in range(50):
        start_time = time.time()
        get_optimal_group_delimeters_wrapper(m, 6)
        times.append(time.time() - start_time)

    print("time:", np.mean(times) * 1000)

    delimeters = get_optimal_group_delimeters_wrapper(m, 5)
    print(delimeters)


if __name__ == "__main__":
    main_()
