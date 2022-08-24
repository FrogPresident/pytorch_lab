import time

import torch


def main():
    # define p0 and p1
    p0 = torch.zeros(1, requires_grad=True)
    p1 = torch.zeros(1, requires_grad=True)
    # define lr
    lr = 0.001
    # define point into x and y
    point = [(2, 3),
             (6, 4),
             (23, 11)]
    # define x and y
    x = torch.tensor([x for x, _ in point], dtype=torch.float)
    y = torch.tensor([y for _, y in point], dtype=torch.float)
    # loop in GD
    for i in range(1000):
        # use model to define parameter
        parameter = model(x, p0, p1)
        # use mse to get loss
        loss = mse(parameter, y)
        print(f'loss={loss.item()}')
        # set p0 and p1 grad to None
        p0.grad = None
        p1.grad = None
        # set loss backward
        loss.backward()
        print(f'gradient of p0 is {p0.grad}, of p1 is {p1.grad}')
        # adjust lr and minus self
        with torch.no_grad():
            p0 -= p0.grad * lr
            p1 -= p1.grad * lr
        print(f"the adjusted parameters: (p0: {p0.item(): .6f}, p1: {p1.item(): .6f})")

        time.sleep(0.001)


def model(x, p0, p1) -> torch.Tensor:
    return p1 + p0 * x


def mse(predict, target) -> torch.Tensor:
    return torch.sum((predict - target) ** 2)


if __name__ == '__main__':
    main()
