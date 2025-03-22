'''
Strategy: Greedy with Max Heap (Priority Queue)

Key Idea:
To avoid adjacent duplicates, we want to spread out the most frequent characters.
Use a max heap to always place the character with the highest remaining frequency.
After placing one character, store the previously used character (if it still has frequency) and reinsert it in the next step.



Time: O(n log k), where n = len(s), k = number of unique characters
Space: O(n + k) for output and heap
'''
import heapq
from collections import Counter


def reorganizeString(s: str) -> str:
    # Step 1: Count the frequency of each character
    freq_map = Counter(s)

    # Step 2: Build a max heap using negative frequencies
    max_heap = [(-freq, char) for char, freq in freq_map.items()]
    heapq.heapify(max_heap) # ← This line is O(k), where k is number of unique characters

    # Result list to build the output string
    result = []

    # Variable to store the previous character and its remaining count
    prev_freq, prev_char = 0, ''

    # Step 3: Greedily build the result using the heap
    while max_heap:
        freq, char = heapq.heappop(max_heap)
        result.append(char)
        print(result)
        print('prev', prev_freq,prev_char)

        # If there's a previous character waiting to be reinserted, do it now
        # 如果只有一个字符了，而且已经被pop出来了，那么就没有东西push到heap里面了
        if prev_freq < 0:
            heapq.heappush(max_heap, (prev_freq, prev_char))

        # Decrease the frequency since we've used one occurrence of this char
        prev_freq = freq + 1  # +1 because freq is negative
        prev_char = char

    # Step 4: Check if the output length is valid
    if len(result) != len(s):
        return ""

    return "".join(result)

s="aaaab"
print(reorganizeString(s))