'''
Booking Airbnb When booking a trip on Airbnb, if we do not have a place available for all of the requested dates, we will suggest a Split Stay where you stay at two different Airbnbs, one for the first part of your trip and the other for the remainder. Given a list of Airbnbs with availability, make an api endpoint that will return all possible combinations of Split Stays across two listings for a given date range For this interview question, we will give each listing a name ("A", "B", etc) and our availability will just be a list of day numbers. For example: Airbnb A - [1,2,3,6,7,10,11] Airbnb B - [3,4,5,6,8,9,10,13] Airbnb C - [7,8,9,10,11] Given the date range [3-11], the expected result would be all sets of two Airbnbs that could form a split stay: Results：[ [B, C] ]

'''

def split_stays_two_listings(avail, start_day, end_day):
    # Build the list of days in the requested stay window, inclusive
    # Example: start=3, end=11 → [3,4,5,6,7,8,9,10,11]
    days = list(range(start_day, end_day + 1))
    m = len(days)  # number of days in the trip

    # Extract all listing names (like "A", "B", "C")
    names = list(avail.keys())

    # Convert each listing's availability list → set for O(1) day lookup
    # We will check many times "is day d available in this listing?"
    sets = {name: set(avail[name]) for name in names}

    # These dictionaries will store prefix/suffix coverage info per listing
    prefix_ok = {}  # prefix_ok[name][i] → can name cover days[0..i] continuously?
    suffix_ok = {}  # suffix_ok[name][i] → can name cover days[i..m-1] continuously?

    # ---- Step 1: Precompute prefix and suffix coverage for each listing ----
    for name in names:
        s = sets[name]  # set of available days for this listing

        # -------- PREFIX COVERAGE --------
        # pref[i] = True means listing can cover ALL days from start_day to days[i]
        # continuously without missing a day.
        pref = [False] * m
        ok = True  # "so far, coverage has been continuous"
        for i, d in enumerate(days):
            # If any day is missing, ok becomes False and stays False
            ok = ok and (d in s)
            pref[i] = ok
        prefix_ok[name] = pref

        # -------- SUFFIX COVERAGE --------
        # suf[i] = True means listing can cover ALL days from days[i] to end_day
        # continuously without missing a day.
        suf = [False] * m
        ok = True
        # Iterate backwards so we check suffixes
        for i in range(m - 1, -1, -1):
            d = days[i]
            ok = ok and (d in s)
            suf[i] = ok
        suffix_ok[name] = suf

    # ---- Step 2: Try all pairs of listings to see if they can form a split stay ----
    results = []

    for a in names:
        for b in names:
            # Must be two different listings
            if a == b:
                continue

            # Try every possible split point i:
            # A covers days[0..i], B covers days[i+1..m-1]
            # We need BOTH:
            #   prefix_ok[a][i]  → A covers first segment
            #   suffix_ok[b][i+1] → B covers second segment
            for i in range(m - 1):  # i+1 must exist, so stop at m-2
                if prefix_ok[a][i] and suffix_ok[b][i + 1]:
                    # Found a valid split point for (a, b)
                    results.append((a, b))
                    break  # no need to try more split points for this pair

    return results