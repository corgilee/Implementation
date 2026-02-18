'''
https://leetcode.com/discuss/post/1681871/robinhood-vo-staff-by-mercuree-k9em/

Overview:

Our goal is to build a simplified version of a real Robinhood system that reads prices from a stream and aggregates those prices into historical datapoints aka candlestick charts. We’re looking for clean code, good naming, testing, etc.

Step 1: Parse Prices

Your input will be a comma-separated string of prices and timestamps in the format price:timestamp e.g.

1:0,3:10,2:12,4:19,5:35 is equivalent to

price: 1, timestamp: 0
price: 3, timestamp: 10
price: 2, timestamp: 12
price: 4, timestamp: 19
price: 5, timestamp: 35

You can assume the input is sorted by timestamp and values are non-negative integers.

Step 2: Aggregate Historical Data from Prices

We calculate historical data across fixed time intervals. In this case, we’re interested in intervals of 10, so the first interval will be [0, 10). For each interval, you’ll build a datapoint with the following values.

Start time
First price
Last price
Max price
Min price

Important: If an interval has no prices, use the previous datapoint’s last price for all prices. If there are no prices and no previous datapoints, skip the interval.

You should return a string formatted as {start,first,last,max,min}. For the prices shown above, the expected datapoints are

{0,1,1,1,1}{10,3,4,4,2}{20,4,4,4,4}{30,5,5,5,5}

'''


INTERVAL = 10

def parse_input(s):
    if not s:
        return []
    out = []
    for part in s.split(","):
        price_str, ts_str = part.split(":")
        out.append((int(ts_str), int(price_str)))  # (timestamp, price)
    return out


def format_candle(start, first, last, mx, mn):
    return f"{{{start},{first},{last},{mx},{mn}}}"


def get_candlesticks(input_str):
    points = parse_input(input_str)
    if not points:
        return ""

    # Build candles per interval with correct first/last order
    candles = {}  # start -> [first, last, max, min]
    min_start = None
    max_start = None

    for ts, price in points:
        start = (ts // INTERVAL) * INTERVAL
        if min_start is None or start < min_start:
            min_start = start
        if max_start is None or start > max_start:
            max_start = start

        if start not in candles:
            candles[start] = [price, price, price, price]  # first, last, max, min
        else:
            c = candles[start]
            c[1] = price                 # last
            c[2] = max(c[2], price)      # max
            c[3] = min(c[3], price)      # min

    # Walk intervals and fill empty ones using previous last price
    res = []
    prev_last = None

    for start in range(min_start, max_start + INTERVAL, INTERVAL):
        if start in candles:
            first, last, mx, mn = candles[start]
            res.append(format_candle(start, first, last, mx, mn))
            prev_last = last
        else:
            # empty interval: only fill if we have a previous datapoint
            if prev_last is None:
                continue
            res.append(format_candle(start, prev_last, prev_last, prev_last, prev_last))
            # prev_last stays the same

    return "".join(res)

###### test case #####################################################################

input_str = (
    "3:12,1:15,4:18,1:30,5:40,9:47,"
    "2:101,6:103,5:105,3:107,5:108,"
    "8:120,9:121,7:122,9:124,3:125,"
    "2:126,3:127,8:128,4:129"
)

result = get_candlesticks(input_str)

expected = (
    "{10,3,4,4,1}"
    "{20,4,4,4,4}"
    "{30,1,1,1,1}"
    "{40,5,9,9,5}"
    "{50,9,9,9,9}"
    "{60,9,9,9,9}"
    "{70,9,9,9,9}"
    "{80,9,9,9,9}"
    "{90,9,9,9,9}"
    "{100,2,5,6,2}"
    "{110,5,5,5,5}"
    "{120,8,4,9,2}"
)

assert result == expected, f"\nExpected:\n{expected}\nGot:\n{result}"

print("All tests passed ✅")
