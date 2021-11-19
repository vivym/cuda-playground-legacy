import random
import torch

from cuda_playground.ops.vb_matrices import VBMatrices
from cuda_playground.ops.vb_mm import VBMMAlgo, vbmm


def main():
    batch_size = 10
    A, B = [], []
    for _ in range(batch_size):
        m, n, k = random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)
        A.append(torch.randn(m, k).contiguous())
        B.append(torch.randn(k, n).contiguous())

    A, B, C = VBMatrices(A), VBMatrices(B), VBMatrices()
    vbmm(A, B, C, 1.0, 0, False, False, VBMMAlgo.Vanilla)


if __name__ == "__main__":
    main()
