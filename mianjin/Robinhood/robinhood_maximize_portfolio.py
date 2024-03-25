TAB = '\t'
def optimize_portfolio_without_fractionals(amount, b_vals, s_vals, counts):
    return dfs(0, amount, {}, b_vals, s_vals, counts)

def dfs(i, amount, memo, b_vals, s_vals, counts, depth=0):
    if i == len(b_vals): return amount
    print(f"{TAB * depth} dfs(i={i}, amount={amount}, i_val={b_vals[i]}, f_val={s_vals[i]})")

    if (i, amount) in memo:
        print(f"{TAB * depth} dfs(i={i} return from memo")
        return memo[(i, amount)]
    if b_vals[i] >= s_vals[i]:
        print(f"{TAB * depth} returning")
        return 0

    vals = []
    for count in range(counts[i] + 1):
        if amount < count * b_vals[i]: continue
        print(f"{TAB * depth} count: {count}, amount: {amount}")
        rev_from_rest = dfs(i + 1, amount - count * b_vals[i], memo, b_vals, s_vals, counts, depth + 1)
        vals.append(
            (rev_from_rest + count * s_vals[i], -b_vals[i])
        )
        max_val = max(vals)[0]
    print(f"{TAB * depth} revs: {vals}, max_rev: {max_val}")
    memo[(i, amount)] = max_val
    print(f"{TAB * depth} return: {max_val}")
    return max_val


def optimize_portfolio_with_fractionals(amount, b_vals, s_vals, counts):
    n = len(b_vals)
    sorted_arrs = [(s_val/b_val, b_val, s_val, c) for b_val, s_val, c in zip(b_vals, s_vals, counts)]
    sorted_arrs.sort(key=lambda x: x[0], reverse=True)
    b_vals = [sorted_arrs[i][1] for i in range(n)]
    s_vals = [sorted_arrs[i][2] for i in range(n)]
    counts = [sorted_arrs[i][3] for i in range(n)]
    revenue, i = 0, 0

    while amount > 0 and i < n:
        num_shares = min(amount/b_vals[i], counts[i])
        revenue += num_shares * s_vals[i]
        amount -= num_shares * b_vals[i]
        i += 1

    return revenue

assert optimize_portfolio_with_fractionals(30, [15, 20], [30, 45], [3, 3]) == 67.5
assert optimize_portfolio_without_fractionals(30, [15, 20], [30, 45], [3, 3]) == 60

assert optimize_portfolio_with_fractionals(140, [15, 40, 25, 30], [45, 50, 35, 25], [3, 3, 3, 4]) == 265
assert optimize_portfolio_without_fractionals(140, [15, 40, 25, 30], [45, 50, 35, 25], [3, 3, 3, 4]) == 255

assert optimize_portfolio_without_fractionals(35, [15, 20], [30, 45], [3, 3]) == 75
assert optimize_portfolio_without_fractionals(45, [15, 20], [30, 45], [3, 3]) == 95
assert optimize_portfolio_without_fractionals(10, [1, 2, 3, 4], [4, 3, 2, 1], [4, 3, 2, 1]) == 25
assert optimize_portfolio_without_fractionals(10, [1, 2, 3, 4], [4, 3, 2, 1], [1, 2, 3, 4]) == 10