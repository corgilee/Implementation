'''
https://leetcode.com/playground/k3HaoUmm
https://leetcode.com/discuss/post/882324/robinhood-phone-screen-by-anonymous_user-6cde/


// A trade is defined as a fixed-width string containing the 4 properties given below separated by commas:

// Symbol (4 alphabetical characters, left-padded by spaces)
// Type (either "B" or "S" for buy or sell)
// Quantity (4 digits, left-padded by zeros)
// ID (6 alphanumeric characters)
// e.g.
// "AAPL,B,0100,ABC123"

// which represents a trade of a buy of 100 shares of AAPL with ID "ABC123"

// Given two lists of trades - called "house" and "street" trades, write code to filter out groups of matches between trades and return a list of unmatched house and street trades sorted alphabetically. There are many ways to match trades, the first and most important way is an exact match (Tests 1-5):

// An exact match is a house_trade+street_trade pair with identical symbol, type, quantity, and ID
// Note: Trades are distinct but not unique

// For example, given the following input:

// house_trades:
[
"AAPL,B,0100,ABC123",
"AAPL,B,0100,ABC123",
"GOOG,S,0050,CDC333"
]

// street_trades:
[
" FB,B,0100,GBGGGG",
"AAPL,B,0100,ABC123"
]

We would expect the following output:

[
" FB,B,0100,GBGGGG",
"AAPL,B,0100,ABC123",
"GOOG,S,0050,CDC333"
]

// Because the first (or second) house trade and second street trade form an exact match, leaving behind three unmatched trades.

// Follow-up 1 (Test 6,7,8,9): A "fuzzy" match is a house_trade+street_trade pair with identical symbol, type, and quantity ignoring ID. Prioritize exact matches over fuzzy matches. Prioritize matching the earliest alphabetical house trade with the earliest alphabetical street trade in case of ties.

// Follow-up 2: (Test 10) An offsetting match is a house_trade+house_trade or street_trade+street_trade pair where the symbol and quantity of both trades are the same, but the type is different (one is a buy and one is a sell). Prioritize exact and fuzzy matches over offsetting matches. Prioritize matching the earliest alphabetical buy with the earliest alphabetical sell.

'''

### basice ###

from collections import Counter

def reconcile_trades(house_trades, street_trades):
    # Count occurrences of each trade string
    house_cnt = Counter(house_trades)
    street_cnt = Counter(street_trades)

    # Cancel out exact matches
    for trade in house_cnt.keys() & street_cnt.keys():
        matched = min(house_cnt[trade], street_cnt[trade])
        if matched > 0:
            house_cnt[trade] -= matched
            street_cnt[trade] -= matched

    # Collect remaining trades
    remaining = []

    for trade, count in house_cnt.items():
        remaining.extend([trade] * count)

    for trade, count in street_cnt.items():
        remaining.extend([trade] * count)

    # Sort alphabetically
    remaining.sort()
    return remaining


### follow up 1 ####
from collections import Counter, defaultdict

def reconcile_trades(house_trades, street_trades):
    # ---------- Step 1: Exact matches (symbol,type,qty,ID) ----------
    house_cnt = Counter(house_trades)
    street_cnt = Counter(street_trades)

    for trade in house_cnt.keys() & street_cnt.keys():
        m = min(house_cnt[trade], street_cnt[trade])
        if m > 0:
            house_cnt[trade] -= m
            street_cnt[trade] -= m

    # Build remaining lists after exact cancellation
    rem_house = []
    for t, c in house_cnt.items():
        if c > 0:
            rem_house.extend([t] * c)

    rem_street = []
    for t, c in street_cnt.items():
        if c > 0:
            rem_street.extend([t] * c)

    # ---------- Step 2: Fuzzy matches (symbol,type,qty) ignoring ID ----------
    # Tie-break: match earliest alphabetical house trade with earliest alphabetical street trade
    def fuzzy_key(trade_str):
        sym, typ, qty, _id = trade_str.split(",")
        return (sym, typ, qty)

    house_bucket = defaultdict(list)
    street_bucket = defaultdict(list)

    for t in rem_house:
        house_bucket[fuzzy_key(t)].append(t)
    for t in rem_street:
        street_bucket[fuzzy_key(t)].append(t)

    # Sort within each bucket so "earliest alphabetical" pairing is respected
    for k in house_bucket:
        house_bucket[k].sort()
    for k in street_bucket:
        street_bucket[k].sort()

    # Cancel fuzzy matches per key, pairing in sorted order
    for k in house_bucket.keys() & street_bucket.keys():
        hlist = house_bucket[k]
        slist = street_bucket[k]
        pairs = min(len(hlist), len(slist))
        if pairs > 0:
            del hlist[:pairs]
            del slist[:pairs]

    # Collect leftovers after fuzzy cancellation
    remaining = []
    for k, hlist in house_bucket.items():
        remaining.extend(hlist)
    for k, slist in street_bucket.items():
        remaining.extend(slist)

    remaining.sort()
    return remaining




###### follow up 2 ########
from collections import Counter, defaultdict

def reconcile_trades(house_trades, street_trades):
    # ---------- Step 1: Exact matches (symbol,type,qty,ID) ----------
    house_cnt = Counter(house_trades)
    street_cnt = Counter(street_trades)

    for trade in house_cnt.keys() & street_cnt.keys():
        m = min(house_cnt[trade], street_cnt[trade])
        if m > 0:
            house_cnt[trade] -= m
            street_cnt[trade] -= m

    rem_house = []
    for t, c in house_cnt.items():
        if c > 0:
            rem_house.extend([t] * c)

    rem_street = []
    for t, c in street_cnt.items():
        if c > 0:
            rem_street.extend([t] * c)

    # ---------- Step 2: Fuzzy matches (symbol,type,qty) ignoring ID ----------
    def fuzzy_key(trade_str):
        sym, typ, qty, _id = trade_str.split(",")
        return (sym, typ, qty)

    house_bucket = defaultdict(list)
    street_bucket = defaultdict(list)

    for t in rem_house:
        house_bucket[fuzzy_key(t)].append(t)
    for t in rem_street:
        street_bucket[fuzzy_key(t)].append(t)

    for k in house_bucket:
        house_bucket[k].sort()
    for k in street_bucket:
        street_bucket[k].sort()

    for k in house_bucket.keys() & street_bucket.keys():
        hlist = house_bucket[k]
        slist = street_bucket[k]
        pairs = min(len(hlist), len(slist))
        if pairs > 0:
            del hlist[:pairs]
            del slist[:pairs]

    rem_house2 = []
    for hlist in house_bucket.values():
        rem_house2.extend(hlist)

    rem_street2 = []
    for slist in street_bucket.values():
        rem_street2.extend(slist)

    # ---------- Step 3: Offsetting matches within same side (symbol,qty; B vs S) ----------
    def offset_key(trade_str):
        sym, typ, qty, _id = trade_str.split(",")
        return (sym, qty)

    def cancel_offset(trades):
        bucket = defaultdict(list)
        for t in trades:
            bucket[offset_key(t)].append(t)

        remaining = []
        for k, lst in bucket.items():
            lst.sort()  # ensures "earliest alphabetical" within this (symbol,qty) group

            buys = []
            sells = []
            for t in lst:
                _sym, typ, _qty, _id = t.split(",")
                if typ == "B":
                    buys.append(t)
                else:  # typ == "S"
                    sells.append(t)

            pairs = min(len(buys), len(sells))
            # match earliest buy with earliest sell (lists already sorted)
            remaining.extend(buys[pairs:])
            remaining.extend(sells[pairs:])

        return remaining

    rem_house3 = cancel_offset(rem_house2)
    rem_street3 = cancel_offset(rem_street2)

    # ---------- Final: combine + sort ----------
    remaining = rem_house3 + rem_street3
    remaining.sort()
    return remaining
