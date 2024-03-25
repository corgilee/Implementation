
#https://leetcode.com/discuss/interview-question/882324/robinhood-phone-screen

from collections import defaultdict


def exact_match(house_trades,street_trades):
    house_dict=defaultdict(int)
    trade_dict=defaultdict(int)
    for trade in house_trades:
        house_dict[trade]+=1

    for trade in street_trades:
        trade_dict[trade]+=1
        if house_dict[trade]>=1:
            house_dict[trade]-=1
            trade_dict[trade]-=1

    res=[]
    new_house_list=[]
    new_trade_list=[]
    for key,val in house_dict.items():
        if val>0:
            cur=[key]*abs(val)
            res+=cur
            new_house_list+=cur

    for key,val in trade_dict.items():
        if val>0:
            cur=[key]*abs(val)
            res+=cur
            new_trade_list+=cur
  
    res.sort()
    
    return res,new_house_list,new_trade_list
    

# test case


house_trades=["AAPL,B,0100,ABC123",
 "AAPL,B,0100,ABC123",
 "AAPL,B,0100,ABC123",
"AAPL,S,0100,ABC123",
 "GOOG,S,0050,CDC333"]
street_trades=["  FB,B,0100,GBGGGG",
 "AAPL,B,0100,ABC123",
"AAPL,B,0100,ABC321",
"  FB,S,0100,GBGGGA",
"  FB,S,0200,GBGGGA",
"  FB,S,0300,GBGGGA",]


# print(res)
# print('new_house_list',new_house_list)
# print('new_trade_list',new_trade_list)

'''
Follow-up 1 (Test 6,7,8,9): A "fuzzy" match is a house_trade+street_trade pair with identical symbol, type, and quantity ignoring ID. 
Prioritize exact matches over fuzzy matches. 
Prioritize matching the earliest alphabetical house trade with the earliest alphabetical street trade in case of ties.
'''


def fuzzy_process(target_trades):
    fuzzy_house_list=[]
    for trade in target_trades:
        cur=trade.split(',')[:3]
        cur=','.join(cur)
        fuzzy_house_list.append(cur)
    #print(fuzzy_house_list)
    return fuzzy_house_list




# def offset process, 在同一组trade里面进行offset

def offset_process(target_trades):
    # after fuzzy matching, there are only 3 elements in the trade
    b_list=[]
    s_list=[]
    for trade in target_trades:
        cur=trade.split(',')
        name=','.join([cur[0],cur[2]])
        buy=cur[1]
        if buy=='B':
            b_list.append(name)
        else:
            s_list.append(name)
    # print('buy_list', b_list)
    # print('sell_list', s_list)

    new_trade_list,_,_= exact_match(b_list,s_list)

    #print(new_trade_list)

    return new_trade_list

offset_process(house_trades)


def main(house_trades,street_trades):
    res,new_house_list,new_trade_list=exact_match(house_trades,street_trades)
    print('After exact match, the result is ',res)

    #fuzzy process
    new_house_list=fuzzy_process(new_house_list)
    new_trade_list=fuzzy_process(new_trade_list)

    res,new_house_list,new_trade_list=exact_match(new_house_list,new_trade_list)
    print('After fuzzy match, the result is ',res)

    new_house_list=offset_process(new_house_list)
    print('After offset match, the result is ',new_house_list)

    new_trade_list=offset_process(new_trade_list)
    print('After offset match, the result is ',new_trade_list)   

    res,new_house_list,new_trade_list=exact_match(new_house_list,new_trade_list)

    print('After fuzzy & offset matching, the res is ',res)

main(house_trades,street_trades)