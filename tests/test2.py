import random
import time

import numpy as np
import torch

from cuda_playground.ops.vb_matrices import VBMatrices
from cuda_playground.ops.vb_mm import VBMMAlgo, vbmm


def main():
    batch_size = 8
    A_, B_ = [], []
    for _ in range(batch_size):
        # m, n, k = random.randint(1, 2048), random.randint(1, 2048), random.randint(1, 2048)
        m, n, k = 1024, 1024, 64
        A_.append(torch.randn(m, k).contiguous().cuda())
        B_.append(torch.randn(k, n).contiguous().cuda())

    A, B, C = VBMatrices(A_), VBMatrices(B_), VBMatrices()
    vbmm(A, B, C, 1.0, 0, False, False, VBMMAlgo.Stream)
    torch.cuda.cudart().cudaProfilerStart()
    vbmm(A, B, C, 1.0, 0, False, False, VBMMAlgo.Stream)
    torch.cuda.cudart().cudaProfilerStop()

    print("done")


if __name__ == "__main__":
    main()
