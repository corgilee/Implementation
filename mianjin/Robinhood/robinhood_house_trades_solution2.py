

#https://leetcode.com/discuss/interview-question/882324/robinhood-phone-screen

from collections import defaultdict
from collections import OrderedDict

def exact_filter(house,street):
    # 建立一个独立的dict，然后house 是+， street 是 -
    trade=defaultdict(int)
    for h in house:
        trade[h]+=1
    for h in street:
        trade[h]-=1

    house=[]
    street=[]
    for key, value in trade.items():
        if value>0:
            house+=value*[key]

        elif value<0:
            street+=-value*[key]
    #house.sort(lambda x:x.key)

    return house, street

def format_input(h):
    s=h.split(",")
    # return a tuple
    return (s[0],s[1],s[2],s[3])

def fuzzy_filter(house, street):

    trade=OrderedDict() # 这里的trade的value是一个list[0,0], p0储存在house 里面的count，p1 储存在street里面的count
    

    for h in house:
        key=format_input(h)[:3]
        if key not in trade:
            trade[key]=[0,0]
        trade[key][0]+=1
    for h in street:
        key=format_input(h)[:3]
        if key not in trade:
            trade[key]=[0,0]
        trade[key][1]+=1
    
    # to be offset
    # offset 即 p1很p2 的最小值，把offset 的值更新到trade的key的两个点上
    for key in trade.keys():
        offset=min(trade[key][0],trade[key][1])
        trade[key][0]=offset
        trade[key][1]=offset
    
    house_res=[]
    street_res=[]
    #print(trade)
    for h in house:
        key=format_input(h)[:3]
        if trade[key][0]>0:
            trade[key][0]-=1
        else:
            house_res.append(h)
    
    for h in street:
        key=format_input(h)[:3]
        if trade[key][1]>0:
            trade[key][1]-=1
        else:
            street_res.append(h)

    return house_res, street_res

    #key : [2,3]. we know the min of key is 2. So both sides we need to remove 2 occurences of the related key

def offset_trade_filter(house):
    trade={}
    for h in house:
        formated_h=format_input(h)
        key=(formated_h[0],formated_h[1])
        b_type=formated_h[1]
        if h not in trade:
            trade[key]=[0,0] # p0 for buy, p1 for sell
        if b_type=="B":
            trade[key][0]+=1
        else:
            trade[key][1]+=1
    for key in trade.keys():
        to_be_removed=min(trade[key][0],trade[key][1])
        trade[key][0]=to_be_removed
        trade[key][1]=to_be_removed
    
    house_res=[]
    for h in house:
        formated_h=format_input(h)
        key=(formated_h[0],formated_h[1])
        b_type=formated_h[1]
        if b_type=="B":
            if trade[key][0]>0:
                trade[key][0]-=1
            else:
                house_res.append(h)
        else:
             if trade[key][1]>0:
                trade[key][1]-=1
            else:
                house_res.append(h)    
    return house_res   



def offset_filter(house,street):
    house=offset_trade_filter(house)
    street=offset_trade_filter(street)
    return house, street

def filter_matches(house, street, fuzzy_match=False, offset_match=False):
    house.sort()
    street.sort()

    house, street =  exact_filter(house, street)
    if fuzzy_match==True:
        house, street = fuzzy_filter(house, street)
    if offset_match==True:
        house, street = offset_filter(house, street)

    return sorted(house + street)


test1 =  [[
 "AAPL,B,0100,ABC123",
 "AAPL,B,0100,ABC123",
 "GOOG,S,0050,CDC333"
 ],  [
 " FB,B,0100,GBGGGG",
 "AAPL,B,0100,ABC123"
 ]]

#print(filter_matches(test1[0], test1[1]))


test2 = [
    ["AAPL,S,0010,ZYX444", 
 "AAPL,S,0010,ZYX444", 
 "AAPL,B,0010,ABC123", 
 "GOOG,S,0050,GHG545"],
 ["GOOG,S,0050,GHG545", 
 "AAPL,S,0010,ZYX444", 
 "AAPL,B,0010,TTT222"]]

print(filter_matches(test2[0], test2[1],fuzzy_match=True))

test3 = [
["AAPL,B,0010,ABC123", 
 "AAPL,S,0015,ZYX444", 
 "AAPL,S,0015,ZYX444", 
 "GOOG,S,0050,GHG545"], 
["GOOG,S,0050,GHG545", 
 "AAPL,S,0015,ZYX444", 
 "AAPL,B,0500,TTT222"]]

print(filter_matches(test3[0], test3[1],fuzzy_match=True))