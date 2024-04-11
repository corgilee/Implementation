
#https://leetcode.com/discuss/interview-question/625536/robhinhood-phone-interview-portfolio-value-optimization

#problem 1

def max_port(stocks, funds):
    stock_num = len(stocks)
    
    # Initialize the DP table
    dp = [[0] * (funds + 1) for _ in range(stock_num + 1)]
    
    # Process each stock
    for i in range(1, stock_num + 1):
        for j in range(funds + 1):
            P = stocks[i - 1][0]  # Price per unit
            S = stocks[i - 1][1]  # Selling price
            C = stocks[i - 1][2]  # Maximum units you can buy
            
            # If no stock is bought
            dp[i][j] = dp[i - 1][j]
            
            # Try buying up to C units of the stock
            for k in range(1, C + 1):
                if j >= k * P:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - k * P] + k * (S - P))
            #print(f"i {i} j {j} dp[i][j] {dp[i][j]}")
    
    return dp[stock_num][funds]

if __name__ == "__main__":
    stocks = [
        [15, 30, 3],  # stock 1 details: Price=15, Sell Price=30, Max Units=3
        [25, 40, 3]   # stock 2 details: Price=25, Sell Price=40, Max Units=3
    ]
    
    print("Starting computation...")
    result = max_port(stocks, 30)
    print(f"Maximum portfolio value: {result}")
    print("Computation ended.")


#problem 2

def optimize_portfolio_with_fractionals(amount, b_vals, s_vals, counts):
    n = len(b_vals)
    '''
    Basically, you want to make every dollar you spent has the maximum return. 
    You can make a data structure to do that. After the sort, get as many as shares you can including fractions
    '''
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