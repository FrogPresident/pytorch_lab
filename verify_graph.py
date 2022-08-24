import torch


def main():
    a = torch.tensor(2, requires_grad=True, dtype=torch.float)
    b = torch.tensor(3, requires_grad=True, dtype=torch.float)
    c = torch.tensor(4, requires_grad=True, dtype=torch.float)

    w = a + b
    f = w ** 2 + w * c

    f.backward()
    print(f"{a.grad=} {b.grad=} {c.grad=} ")


if __name__ == '__main__':
    main()
