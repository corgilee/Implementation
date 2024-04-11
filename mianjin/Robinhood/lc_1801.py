class Solution:
    
    
    def getNumberOfBacklogOrders(self, orders: List[List[int]]) -> int:
        '''
        set up 2 heap
        sell_heap, #min_heap
        buy_heap, #max_heap

        # return total_amount of orders in the backlog, modulo 10**9+7
        '''
        amount=0
        sell_heap=[]
        buy_heap=[]
        backlog_sell_amount=0
        backlog_buy_amount=0
        for order in orders:
            #print(order)
            c_price,c_amount,o_type=order[0],order[1],order[2]
            # buy
            if o_type==0:
                #print('buy')
                while len(sell_heap)!=0 and c_amount!=0 and sell_heap[0][0]<=c_price:
                    top_price, top_amount=heapq.heappop(sell_heap)
                    c_trade=min(top_amount,c_amount)
                    c_amount-=c_trade
                    top_amount-=c_trade
                    backlog_sell_amount-=c_trade
                    if top_amount>0:
                        heapq.heappush(sell_heap,(top_price,top_amount))
                # buy is the maximum heap
                if c_amount>0:
                    heapq.heappush(buy_heap,(-c_price,c_amount))
                    backlog_buy_amount+=c_amount
            # sell
            elif o_type==1:
                #print('sell')
                while len(buy_heap)!=0 and c_amount!=0 and -buy_heap[0][0]>=c_price:
                    #print(buy_heap)
                    top_price, top_amount = heapq.heappop(buy_heap)
                    top_price=-top_price
                    c_trade=min(top_amount,c_amount)
                    c_amount-=c_trade
                    top_amount-=c_trade
                    backlog_buy_amount-=c_trade
                    if top_amount>0:
                        heapq.heappush(buy_heap,(-top_price,top_amount))
                if c_amount>0:
                    heapq.heappush(sell_heap,(c_price,c_amount))
                    backlog_sell_amount+=c_amount

            # print(buy_heap)
            # print(sell_heap)
            # print(amount)
        return (backlog_buy_amount+backlog_sell_amount)%(10**9+7)



        