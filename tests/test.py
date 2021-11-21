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

    A, B, C = VBMatrices(A_), VBMatrices(B_), VBMatrices()
    algo = VBMMAlgo.MAGMA
    for _ in range(10):
        vbmm(A, B, C, 1.0, 0, False, False, VBMMAlgo.Vanilla)
        vbmm(A, B, C, 1.0, 0, False, False, VBMMAlgo.Stream)
        vbmm(A, B, C, 1.0, 0, False, False, VBMMAlgo.MAGMA)
    
    ################################################################################
    print("Vanilla")
    times = []
    for _ in range(50):
        start_time = time.time()
        vbmm(A, B, C, 1.0, 0, False, False, VBMMAlgo.Vanilla)
        times.append(time.time() - start_time)
    
    print("time:", np.mean(times) * 1000)

    C_data = C.data

    offset = 0
    for a, b in zip(A_, B_):
        c_std = a @ b
        m, n = c_std.shape
        c = C_data[offset:offset + m * n].reshape(m, n)
        if not torch.allclose(c, c_std, atol=1e-4, rtol=1e-4):
            print("failed", (c - c_std).abs())
        offset += m * n

    print("done")

    ################################################################################
    print("Stream")
    times = []
    for _ in range(50):
        start_time = time.time()
        vbmm(A, B, C, 1.0, 0, False, False, VBMMAlgo.Stream)
        times.append(time.time() - start_time)
    
    print("time:", np.mean(times) * 1000)

    C_data = C.data

    offset = 0
    for a, b in zip(A_, B_):
        c_std = a @ b
        m, n = c_std.shape
        c = C_data[offset:offset + m * n].reshape(m, n)
        if not torch.allclose(c, c_std, atol=1e-4, rtol=1e-4):
            print("failed", (c - c_std).abs())
        offset += m * n

    print("done")

    ################################################################################
    print("MAGMA")
    times = []
    for _ in range(50):
        start_time = time.time()
        vbmm(A, B, C, 1.0, 0, False, False, VBMMAlgo.MAGMA)
        times.append(time.time() - start_time)
    
    print("time:", np.mean(times) * 1000)

    C_data = C.data

    offset = 0
    for a, b in zip(A_, B_):
        c_std = a @ b
        m, n = c_std.shape
        c = C_data[offset:offset + m * n].reshape(m, n)
        if not torch.allclose(c, c_std, atol=1e-4, rtol=1e-4):
            print("failed", (c - c_std).abs())
        offset += m * n

    print("done")

    ################################################################################
    print("Fully Packed")

    A = torch.randn(batch_size, 256, 64, device="cuda")
    B = torch.randn(batch_size, 64, 256, device="cuda")
    for _ in range(10):
        C = A @ B
    
    times = []
    for _ in range(50):
        start_time = time.time()
        offset = 0
        for _ in range(8):
            C = A[offset:offset+512] @ B[offset:offset+512]
            offset += 512
        times.append(time.time() - start_time)
    
    print(C.sum(), C.shape)
    
    print("time:", np.mean(times) * 1000)


if __name__ == "__main__":
    main()
