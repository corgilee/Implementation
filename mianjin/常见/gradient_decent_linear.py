def gradient_descent_linear(xs, ys, lr=0.01, iters=1000):
    """
    Simple gradient descent for y = w * x + b without NumPy.

    Args:
        xs, ys: lists of equal length containing inputs and targets.
        lr: learning rate.
        iters: number of iterations.

    Returns:
        Tuple (w, b) approximating the best-fit line.
    """

    w = 0.0
    b = 0.0
    m = len(xs)

    for _ in range(iters):
        preds = [w * x + b for x in xs]

        dw_num = 0.0
        db_num = 0.0
        for pred, x, y in zip(preds, xs, ys):
            dw_num += (pred - y) * x
            db_num += (pred - y)
        dw = dw_num / m
        db = db_num / m

        w -= lr * dw
        b -= lr * db

    return w, b


if __name__ == "__main__":
    # toy example
    xs = [1, 2, 3, 4]
    ys = [3, 5, 7, 9]  # roughly y = 2x + 1

    w, b = gradient_descent_linear(xs, ys, lr=0.01, iters=5000)
    print(f"w={w:.3f}, b={b:.3f}")
