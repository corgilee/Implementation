'''
You have some securities available to buy that each has a price Pi.
Your friend predicts for each security the stock price will be Si at some future date.
But based on volatility of each share, you only want to buy up to Ai shares of each security i.
Given M dollars to spend, calculate the maximum value you could potentially
achieve based on the predicted prices Si (and including any cash you have remaining).

Pi = Current Price
Si = Expected Future Price
Ai = Maximum units you are willing to purchase
M = Dollars available to invest
Example 1:
Input:
M = $140 available

P1=15, S1=45, A1=3 (AAPL)
P2=40, S2=50, A2=3 (BYND)
P3=25, S3=35, A3=3 (SNAP)
P4=30, S4=25, A4=4 (TSLA)

Output: $265 (no cash remaining) (I'm not sure if this is with fractional or without) This was not specified in the question.

But we'll have two answers based on our implementation:

With fractional buying (Unbounded knapsack)
Without fractional buying (1/0 Knapsack)

Example 2:
Input:
M = $30
P1=15, S1=30, A1=3 (AAPL)
P2=20, S2=45, A2=3 (TSLA)

Output:
When buying fractionals,
Buy 1.5 shares of TSLA ($67.5 value)

When buying whole shares,
Buy 2 shares of AAPL ($60 value)
'''

def optimize_portfolio_without_fractionals(amount, b_vals, s_vals, counts):
    memo = {}
    return dfs(0, amount, memo, b_vals, s_vals, counts)

def dfs(i, amount, memo, b_vals, s_vals, counts):
    if i == len(b_vals):
        return amount  # leftover cash counts as value

    key = (i, amount)
    if key in memo:
        return memo[key]

    best = 0  # will be overwritten by at least count=0 case
    # Option: buy 0..Ai shares of i
    for k in range(counts[i] + 1):
        cost = k * b_vals[i]
        if cost > amount:
            break
        val = dfs(i + 1, amount - cost, memo, b_vals, s_vals, counts) + k * s_vals[i]
        if val > best:
            best = val

    memo[key] = best
    return best


def optimize_portfolio_with_fractionals(amount, b_vals, s_vals, counts):
    n = len(b_vals)
    assets = []
    for b, s, c in zip(b_vals, s_vals, counts):
        # skip non-profitable assets (they reduce value if cash is valued)
        if s <= b:
            continue
        assets.append(((s - b) / b, b, s, c))  # profit density

    assets.sort(key=lambda x: x[0], reverse=True)

    total_value = amount  # leftover cash baseline value
    remaining = amount

    for _, b, s, c in assets:
        if remaining <= 0:
            break
        take = min(remaining / b, c)
        total_value += take * (s - b)  # net improvement over holding cash
        remaining -= take * b

    return total_value


assert optimize_portfolio_with_fractionals(30, [15, 20], [30, 45], [3, 3]) == 67.5
assert optimize_portfolio_without_fractionals(
    30, [15, 20], [30, 45], [3, 3]) == 60

assert optimize_portfolio_with_fractionals(
    5,
    [10, 20],
    [30, 50],
    [3, 3]
) == 5


assert optimize_portfolio_without_fractionals(
    45,
    [15, 20],
    [30, 45],
    [3, 3]
) == 95