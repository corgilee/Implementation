
'''
模拟交易系统
有买单进入的话，看看池子里面有没有比比当前买单价格低的卖单，然后直到价格高的卖单全部卖掉
如果有卖单进入的话，看看池子里面有没有买单，而且价格要比当前卖单价格高，直到买单消耗完为止

典型的用heap 做的题目

'''

import heapq

# orders 是 list in list

# orders = [
#         ["150", "5", "buy"],
#         ["190", "1", "sell"],
#         ["200", "1", "sell"],
#         ["100", "9", "buy"],
#         ["140", "8", "sell"],
#         ["210", "4", "buy"]
#     ]

def order_book_2(orders):
    if len(orders)==0:
        return 0

    res=0
    # build a list for buy and sell
    buy=[] #maxheap with (-price,share)
    sell=[] # minheap with (price,share)

    for order in orders:
        c_type=order[2]
        #print(order)

        if c_type=="buy":
            c_price,c_share=int(order[0]),int(order[1])
            # check if there is any existing sell in the sell list:
            while sell and sell[0][0]<=c_price and c_share>0:
                sell_price, sell_share=heapq.heappop(sell) #这里是heap pop，不是一般的pop
                traded_shares=min(sell_share,c_share)
                res+=traded_shares
                sell_share-=traded_shares
                c_share-=traded_shares
                if sell_share>0:
                    heapq.heappush(sell, (sell_price,sell_share))
            if c_share>0:
                heapq.heappush(buy,(-c_price,c_share))

        else: ### c_type=="sell"
            c_price,c_share=int(order[0]),int(order[1])
            while buy and -buy[0][0]>=c_price and c_share>0:
                buy_price, buy_share=heapq.heappop(buy)
                traded_share=min(buy_share, c_share)
                #print(buy_share,c_share,traded_share)
                res+=traded_share
                buy_share-=traded_share
                c_share-=traded_share
                if buy_share>0:
                    heapq.heappush(buy,(buy_price, buy_share))
            if c_share>0:
                heapq.heappush(sell, (c_price,c_share))
        # print('buy ', buy)
        # print('sell ',sell)
        # print('-----------------')
    return res





# 原 java
# def order_book_1(orders):
#     if not orders or not orders[0]:
#         return 0
    
    
#     res = 0
#     # Use min-heap for sell orders directly, and invert prices for buy orders to simulate max-heap
#     buy = []
#     sell = []
#     for order in orders:
#         print(order)
#         price, share, order_type = int(order[0]), int(order[1]), order[2]
        
#         if order_type == "buy":
#             buynode = (-price, share)  # Invert price for buy orders
#             while sell and sell[0][0] <= price and share > 0:
#                 sell_price, sell_share = heapq.heappop(sell)
#                 trade_share = min(share, sell_share)
#                 res += trade_share
#                 share -= trade_share
#                 sell_share -= trade_share
#                 if sell_share > 0:
#                     heapq.heappush(sell, (sell_price, sell_share))
#             if share > 0:
#                 heapq.heappush(buy, (buynode[0], share))
#         else:
#             sellnode = (price, share)
#             while buy and -buy[0][0] >= price and share > 0:
#                 buy_price, buy_share = heapq.heappop(buy)
#                 trade_share = min(share, buy_share)
#                 res += trade_share
#                 share -= trade_share
#                 buy_share -= trade_share
#                 if buy_share > 0:
#                     heapq.heappush(buy, (buy_price, buy_share))
#             if share > 0:
#                 heapq.heappush(sell, (price, share))
#         print('buy',buy)
#         print('sell',sell)
#         print('-----------------')
#     return res

# Main function to execute the order book processing
if __name__ == "__main__":
    orders = [
        ["150", "5", "buy"],
        ["190", "1", "sell"],
        ["200", "1", "sell"],
        ["100", "9", "buy"],
        ["140", "8", "sell"],
        ["210", "4", "buy"]
    ]
    print("Total shares traded:", order_book_1(orders))
    print("Total shares traded:", order_book_2(orders))
