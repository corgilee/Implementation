'''
https://leetcode.com/discuss/post/1027102/robinhood-telephonic-round-by-anonymous_-zuxu/

Given a stream of incoming "buy" and "sell" orders (as lists of limit price, quantity, and side, like ["155", "3", "buy"]), determine the total quantity (or number of "shares") executed.

A "buy" order can be executed if there is a corresponding "sell" order with a price that is less than or equal to the price of the "buy" order.
Similarly, a "sell" order can be executed if there is a corresponding "buy" order with a price that is greater than or equal to the price of the "sell" order.
It is possible that an order does not execute immediately if it isn't paired to a counterparty. In that case, you should keep track of that order and execute it at a later time when a pairing order is found.
You should ensure that orders are filled immediately at the best possible price. That is, an order should be executed when it is processed, if possible. Further, "buy" orders should execute at the lowest possible price and "sell" orders at the highest possible price at the time the order is handled.

Note that orders can be partially executed.

--- Sample Input ---

orders = [
['150', '5', 'buy'], # Order A
['190', '1', 'sell'], # Order B
['200', '1', 'sell'], # Order C
['100', '9', 'buy'], # Order D
['140', '8', 'sell'], # Order E
['210', '4', 'buy'], # Order F
]

Sample Output
9
'''

import heapq

def total_executed_quantity(orders):
    """
    orders: list of [price(str), qty(str), side(str 'buy'/'sell')]
    returns: total executed quantity (int)
    """
    executed = 0

    # buy: max-heap by price -> store (-price, qty)
    buy_heap = []
    # sell: min-heap by price -> store (price, qty)
    sell_heap = []

    for price_str, qty_str, side in orders:
        price = int(price_str)
        qty = int(qty_str)

        if side == "buy":
            # Match against lowest sell prices first (best for buyer)
            while qty > 0 and sell_heap and sell_heap[0][0] <= price:
                sell_price, sell_qty = heapq.heappop(sell_heap)

                traded = min(qty, sell_qty)
                executed += traded
                qty -= traded
                sell_qty -= traded

                if sell_qty > 0:
                    heapq.heappush(sell_heap, (sell_price, sell_qty))

            if qty > 0:
                heapq.heappush(buy_heap, (-price, qty))

        else:  # side == "sell"
            # Match against highest buy prices first (best for seller)
            while qty > 0 and buy_heap and -buy_heap[0][0] >= price:
                buy_price_neg, buy_qty = heapq.heappop(buy_heap)
                buy_price = -buy_price_neg

                traded = min(qty, buy_qty)
                executed += traded
                qty -= traded
                buy_qty -= traded

                if buy_qty > 0:
                    heapq.heappush(buy_heap, (-buy_price, buy_qty))

            if qty > 0:
                heapq.heappush(sell_heap, (price, qty))

    return executed



