from collections import defaultdict

def create_map(trades):
    trade_map = defaultdict(list)
    for trade in trades:
        parts = trade.split(",")
        symbol, quantity = parts[0], parts[2]
        key = f"{symbol} {quantity}"
        trade_map[key].append(trade)
    return trade_map

def find_fuzzy_matches(list1, list2):
    i = len(list1) - 1
    while i >= 0:
        arr1 = list1[i].split(",")
        type1 = arr1[1]
        j = 0
        while j < len(list2):
            arr2 = list2[j].split(",")
            type2 = arr2[1]
            if type1 == type2:
                print("Fuzzy match found")
                list1.pop(i)
                list2.pop(j)
                break
            j += 1
        i -= 1

def find_offsetting_matches(trades):
    trades.sort()
    find = True
    while find:
        find = find_offsetting_matches_helper(trades)

def find_offsetting_matches_helper(trades):
    i = 0
    while i < len(trades):
        arr1 = trades[i].split(",")
        type1 = arr1[1]
        j = i + 1
        while j < len(trades):
            arr2 = trades[j].split(",")
            type2 = arr2[1]
            if type1 != type2:
                print("Offsetting match found")
                trades.pop(j)
                trades.pop(i)
                return True
            j += 1
        i += 1
    return False

def do_trade(house_trade, street_trade):
    output = []
    h_map = create_map(house_trade)
    #print(h_map)
    s_map = create_map(street_trade)
    #print(s_map)

    for key in s_map:
        if key in h_map:
            h_list = h_map[key]
            s_list = s_map[key]
            h_list.sort(reverse=True)
            s_list.sort()
            print('h_list',h_list)
            print('s_list',s_list)
            find_fuzzy_matches(h_list, s_list)

    for key in h_map:
        find_offsetting_matches(h_map[key])

    for key in s_map:
        find_offsetting_matches(s_map[key])

    for key in h_map:
        output.extend(h_map[key])

    for key in s_map:
        output.extend(s_map[key])

    output.sort()
    return output

# Example usage
if __name__ == "__main__":
    house_trade = ["AAPL,SELL,100", "GOOG,BUY,200", "AAPL,SELL,150"]
    street_trade = ["AAPL,SELL,100", "AAPL,BUY,150", "GOOG,BUY,200"]
    result = do_trade(house_trade, street_trade)
    for trade in result:
        print(trade)
