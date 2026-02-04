'''
用Python实现 Minimize a blackbox convex function F(x) in interval [a,b]，F(x) is convex in [a, b]

'''

'''
This function implements golden-section search, a derivative-free optimization method for finding the minimum of a one-dimensional convex (unimodal) blackbox function within a closed interval [l,r]. 

Since the function has only one basin, we can iteratively shrink the search interval by comparing function values at two interior points chosen using the golden ratio. 
This special ratio allows the algorithm to reuse one previous function evaluation each iteration, making it efficient when function evaluations are expensive. 
The process continues until the interval is sufficiently small, and the midpoint of the final interval is returned as the approximate minimizer.

'''


from typing import Callable

def golden_section_search(f: Callable[[float], float],
                          l: float,
                          r: float,
                          tol: float = 1e-8,
                          max_iter: int = 10_000) -> float:
    """
    Minimize a 1D blackbox convex (unimodal) function f on [l, r].
    Returns approximate minimizer x*.
    """

    phi = (1 + 5 ** 0.5) / 2      # golden ratio ≈ 1.618
    inv_phi = 1 / phi             # ≈ 0.618

    # Initial interior points
    x1 = r - inv_phi * (r - l)
    x2 = l + inv_phi * (r - l)
    f1 = f(x1)
    f2 = f(x2)

    for _ in range(max_iter):
        if abs(r - l) <= tol:
            break

        if f1 <= f2:
            # Minimum is in [l, x2]
            r = x2
            x2, f2 = x1, f1                # reuse
            x1 = r - inv_phi * (r - l)
            f1 = f(x1)
        else:
            # Minimum is in [x1, r]
            l = x1
            x1, f1 = x2, f2                # reuse
            x2 = l + inv_phi * (r - l)
            f2 = f(x2)

    return (l + r) / 2