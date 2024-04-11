
'''
The Python version retains the functionality of parsing input data, aggregating it into "candlestick" objects based on time intervals
, and then processing these to account for missing time ranges:
'''

# 首先要定义candle 和 他的method
class Candle:
    def __init__(self):
        self.start = 0
        self.max = 0
        self.min = 0
        self.first = 0
        self.last = 0

# step 1: parse data
def parse_input(input_str):
    res = []
    if not input_str:
        return res
    lists = input_str.split(",")
    for s in lists:
        ss = s.split(":")
        tmp = [int(ss[0]), int(ss[1])] #注意这里 先是price，然后是时间
        res.append(tmp)
    return res

def merge(existing_candle, new_candle):
    existing_candle.max = max(existing_candle.max, new_candle.max)
    existing_candle.min = min(existing_candle.min, new_candle.min)
    existing_candle.last = new_candle.last

# 处理每一个time_and_prices
def get_candlesticks(timed_and_prices):
    map_ = {}
    mintime = float('inf')
    maxtime = float('-inf')
    for time_and_price in timed_and_prices:
        time, price = time_and_price[1], time_and_price[0]
        key = time // 10 * 10
        mintime = min(mintime, key)
        maxtime = max(maxtime, key)
        candle = Candle()
        candle.start = key
        candle.max = price
        candle.min = price
        candle.first = price
        candle.last = price
        if key not in map_:
            map_[key] = candle
        else:
            merge(map_[key], candle)

    # 分 step 开始处理 candle
    last_candle = Candle()
    result = []
    for key in range(mintime, maxtime + 10, 10):
        candle = map_.get(key, None)
        if candle is None:
            candle = Candle()
            candle.start = last_candle.start + 10
            candle.max = last_candle.last
            candle.min = last_candle.last
            candle.first = last_candle.last
            candle.last = last_candle.last
        result.append(candle)
        last_candle = candle
    return result

# Example usage
if __name__ == "__main__":
    input_str = "3:12,1:15,4:18,1:30,5:40,9:47,2:101,6:103,5:105,3:107,5:108,8:120,9:121,7:122,9:124,3:125,2:126,3:127,8:128,4:129"
    timed_and_prices = parse_input(input_str)
    result = get_candlesticks(timed_and_prices)
    for candle in result:
        print(candle.start, candle.first, candle.last, candle.max, candle.min)
