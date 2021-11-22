import random

import numpy as np
from numpy.core.fromnumeric import sort


def main():
    inf = 10000000
    batch_size = 1000
    K = 6
    A = [random.randint(1, 256) for _ in range(batch_size)]
    A.sort()
    S = [A[0]]
    for i in range(1, batch_size):
        S.append(S[-1] + A[i])
    
    # print(A)
    # print(S)

    f = [
        [0] + [inf for _ in range(K - 1)]
    ]
    d = [[-1 for _ in range(K)]]
    # print(0, f[0])
    for i in range(1, batch_size):
        tmp = []
        d.append([-1 for _ in range(K)])
        for k in range(K):
            min_f = inf
            if k == 0:
                min_f = A[i] * (i + 1) - S[i]
            else:
                for j in range(i):
                    value = f[j][k - 1] + A[i] * (i - j) - (S[i] - S[j])
                    if value < min_f:
                        min_f = value
                        d[i][k] = j
            tmp.append(min_f)
        # print(i, tmp)
        f.append(tmp)

    print(f[-1])
    for f_i in f[-1]:
        print(f_i / S[-1] * 100)

    print(d)


if __name__ == "__main__":
    main()
