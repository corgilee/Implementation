'''
Q1

给出一列的records[_, sticker, symbol, operation, quantity, price]，返回user 的portfolio
、
'''

class solution:
    def __init__(self):
        self.cash=1000
        self.symbol_map = {}
    
    def get_user_portfolio(self, records):
        
        portfolio=[]
        if len(records)==0:
            return portfolio

        for record in records:
            symbol, operation, qt, price=record[1],record[2],record[3],record[4]
            self.trade_stock(symbol,operation, qt, price)

        for symbol, qt in sorted(self.symbol_map.items()):
            portfolio.append([symbol,qt])

        portfolio.insert(0,["cash",self.cash])

        return portfolio

    def trade_stock(self, symbol,operation, qt, price):
        if operation=="B":
            print(self.symbol_map)
            self.symbol_map[symbol]=self.symbol_map.get(symbol,0)+int(qt)

            self.cash-=int(qt)*int(price)

        elif operation=='S':
            self.symbol_map[symbol]=self.symbol_map.get(symbol,0)-int(qt)
            self.cash+=int(qt)*int(price)

margin_call=solution()

records0 = [
    ["1", "AAPL", "B", "10", "10"],
    ["3", "GOOG", "B", "20", "5"],
    ["10", "AAPL", "S", "5", "15"],
]
portfolio0 = margin_call.get_user_portfolio(records0)
print("Q1 test case 0:")
print(portfolio0)

# [CASH, 875]
# [AAPL, 5]
# [GOOG, 20]


records1=[
    ["1", "AAPL", "B", "10", "10"],
    ["3", "GOOG", "B", "20", "5"],
    ["4", "  FB", "B", "5", "12"],
    ["3", "GOOG", "S", "3", "8"],
    ["3", "GOOG", "B", "5", "10"],
    ["10", "AAPL", "S", "5", "15"]
]

margin_call=solution() # 要重新initiate 一下
portfolio1 = margin_call.get_user_portfolio(records1)
print("Q1 test case 1:")
print(portfolio1)
# [CASH, 789]
# [  FB, 5]
# [AAPL, 5]
# [GOOG, 22]



