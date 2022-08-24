import time

from matplotlib import pyplot as plt

import torch
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


def main():
    p0 = torch.zeros(1, requires_grad=True)
    p1 = torch.zeros(1, requires_grad=True)

    lr = 0.0001
    points = [
        (2, 3),
        (4, 6),
        (7, 9)
    ]

    X = torch.tensor([x for x, _ in points], dtype=torch.float)
    Y = torch.tensor([y for _, y in points], dtype=torch.float)

    fig: plt.Figure = plt.figure()
    ax: Axes = fig.add_subplot(1, 1, 1)
    ax.plot(X, Y, "ob")
    line: Line2D = ax.plot(*get_line_space(p0, p1, X))[0]

    fig.show()

    for i in range(1000):
        predictions = model(X, p0, p1)
        loss = mse(predictions, Y)

        print(f'loss={loss.item()}')
        p0.grad = None
        p1.grad = None
        loss.backward()

        print(f'gradient of p0 is {p0.grad}, of p1 is {p1.grad}')
        with torch.no_grad():
            p0 -= lr * p0.grad
            p1 -= lr * p1.grad
        print(f"the adjusted parameters: (p0: {p0.item(): .6f}, p1: {p1.item(): .6f})")
        line.set_data(*get_line_space(p0, p1, X))
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.001)


def model(x, p0, p1) -> torch.Tensor:
    return p0 + p1 * x


def mse(pred, target) -> torch.Tensor:
    return torch.sum((pred - target) ** 2)


@torch.no_grad()
def get_line_space(p0, p1, X: torch.Tensor):
    space = torch.linspace(X.min(), X.max(), 1000)
    return space, model(space, p0, p1).numpy()


if __name__ == '__main__':
    main()
