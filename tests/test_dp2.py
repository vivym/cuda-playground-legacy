import random

from cuda_playground.ops.dp import get_optimal_group_delimeters_wrapper


def main():
    random.seed(0)

    batch_size = 4096
    m = []
    for _ in range(batch_size):
        m.append(random.randint(32, 256))

    # delimeters = get_optimal_group_delimeters_wrapper(m, 3)
    # print(delimeters)


if __name__ == "__main__":
    main()
