
import math
'''
Q2

给出一列的records[_, sticker, symbol, operation, quantity, price]，返回user 的portfolio with Margincall

这里margin call 的意思是 ， 如果cash<0, 那么就依次卖出买入价格最高的股票，直到卖到cash 为正，或者不能卖为止

如果有股票被margin call卖掉的话，相应的记录要被record下来
、
'''


class solution:
    def __init__(self):
        self.initial_cash_amount=1000

    
    def get_user_portfolio_with_margin_call(self, records):
        
        portfolio=[]

        self.symbol_map = {}
        self.symbol_price={}
        self.cash=self.initial_cash_amount
        if len(records)==0:
            return portfolio

        for record in records:
            symbol, operation, qt, price=record[1],record[2],record[3],record[4]
            self.trade_stock(symbol,operation, qt, price)

            self.symbol_price[symbol]=int(price)
            self.margin_call() #每一次都要检查一下是否触发margin call
            #print(self.cash)


        for symbol, qt in sorted(self.symbol_map.items()):
            portfolio.append([symbol,qt])

        portfolio.insert(0,["cash",self.cash])

        return portfolio


    def margin_call(self):
        if self.cash>0:
            return
        else:
            while self.cash<0:
                stock_with_highest_price,highest_price=max(self.symbol_price.items(),key=lambda x:x[1])

                qt_to_sell=min(self.symbol_map[stock_with_highest_price],math.ceil(-self.cash/highest_price))

                #要实施margin call了，所以要触发 trade_stock
                self.trade_stock(stock_with_highest_price,'S',qt_to_sell,highest_price)
                if self.symbol_map[stock_with_highest_price]==0:
                    del self.symbol_map[stock_with_highest_price]
                    del self.symbol_price[stock_with_highest_price]
                #break
        return


    def trade_stock(self, symbol,operation, qt, price):
        if operation=="B":
            #print(self.symbol_map)
            self.symbol_map[symbol]=self.symbol_map.get(symbol,0)+int(qt)

            self.cash-=int(qt)*int(price)

        elif operation=='S':
            self.symbol_map[symbol]=self.symbol_map.get(symbol,0)-int(qt)
            self.cash+=int(qt)*int(price)


margin_call=solution()

records2 = [
    ["1", "AAPL", "B", "10", "100"],
    ["2", "AAPL", "S", "2", "80"],
    ["3", "GOOG", "B", "15", "20"],
]
portfolio2 = margin_call.get_user_portfolio_with_margin_call(records2)
print("Q2 test case 1:")
print(portfolio2)
# [CASH, 20]
# [AAPL, 6]
# [GOOG, 15]


# Test case 2
records3 = [
    ["1", "AAPL", "B", "5", "100"],
    ["2", "ABPL", "B", "5", "100"],
    ["3", "AAPL", "S", "2", "80"],
    ["4", "ABPL", "S", "2", "80"],
    ["5", "GOOG", "B", "15", "30"],
]


portfolio3 = margin_call.get_user_portfolio_with_margin_call(records3)
print("Q2 test case 2:")
print(portfolio3)

# [CASH, 30]
# [AAPL, 1]
# [ABPL, 3]
# [GOOG, 15]

# Test case 3
records4 = [
    ["1", "AAPL", "B", "5", "100"],
    ["2", "ABPL", "B", "5", "100"],
    ["3", "AAPL", "S", "2", "80"],
    ["4", "ABPL", "S", "2", "120"],
    ["5", "GOOG", "B", "15", "30"],
]

portfolio4 = margin_call.get_user_portfolio_with_margin_call(records4)
print("Q2 test case 3:")
print(portfolio4)


# [CASH, 70]
# [AAPL, 3]
# [ABPL, 2]
# [GOOG, 15]

# Test case 4
records5 = [
    ["1", "AAPL", "B", "5", "100"],
    ["2", "ABPL", "B", "5", "100"],
    ["3", "AAPL", "S", "2", "80"],
    ["4", "ABPL", "S", "2", "120"],
    ["5", "GOOG", "B", "10", "80"],
]

portfolio5 = margin_call.get_user_portfolio_with_margin_call(records5)
print("Q2 test case 4:")
print(portfolio5)

# [CASH, 40]
# [AAPL, 2]
# [GOOG, 10]

