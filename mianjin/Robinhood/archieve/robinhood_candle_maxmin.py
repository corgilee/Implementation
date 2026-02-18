
'''
The Python version retains the functionality of parsing input data, aggregating it into "candlestick" objects based on time intervals
, and then processing these to account for missing time ranges:
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



input_str = (
    "3:12,1:15,4:18,1:30,5:40,9:47,"
    "2:101,6:103,5:105,3:107,5:108,"
    "8:120,9:121,7:122,9:124,3:125,"
    "2:126,3:127,8:128,4:129"
)

result = get_candlesticks(input_str)

expected = (
    "{10,3,4,4,1}"
    "{20,1,1,1,1}"
    "{30,1,1,1,1}"
    "{40,5,9,9,5}"
    "{50,9,9,9,9}"
    "{60,9,9,9,9}"
    "{70,9,9,9,9}"
    "{80,9,9,9,9}"
    "{90,9,9,9,9}"
    "{100,2,3,6,2}"
    "{110,3,3,3,3}"
    "{120,8,4,9,2}"
)

assert result == expected, f"\nExpected:\n{expected}\nGot:\n{result}"

print("All tests passed ✅")
