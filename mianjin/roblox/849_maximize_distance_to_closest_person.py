'''
Strategy: Linear Scan

We’ll scan through the seats array and calculate distances in three key cases:
Leading zeros (e.g., [0, 0, 1, ...]): distance is the number of leading 0s.
Middle zeros between two 1s (e.g., [1, 0, 0, 1]): distance is floor(gap / 2).
Trailing zeros (e.g., [..., 1, 0, 0]): distance is the number of trailing 0s.

Track the maximum of all such distances

Time: O(n)
Space: O(1)
'''

def maxDistToClosest(self, seats: List[int]) -> int:
    max_dist = 0
    n = len(seats)
    prev = -1  # index of last seen person (seat with 1)

    for i in range(n):
        if seats[i] == 1:
            if prev == -1:
                # Case 1: leading zeros from start to first person
                max_dist = i
            else:
                # Case 2: middle gap between two people
                gap = (i - prev) // 2
                max_dist = max(max_dist, gap)
            prev = i

    # Case 3: trailing zeros after the last person
    max_dist = max(max_dist, n - 1 - prev)

    return max_dist

