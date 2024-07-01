'''
题目
https://leetcode.com/discuss/interview-question/4245529/Pinterest-or-Phone-or-Count-Pins
'''

'''
Pinterest app screen is two columns of images (pins).
Each pin has a top and bottom index (top < bottom), and has a column represented by "L" or "R".
Given an unsorted list of non-overlapping pins like
and a screen_length of screen_len
Return the maximum number of pins the user can possibly see (fully).

pin 的top是上边缘的位置，bottom是下边缘的位置
screen_len 是屏幕的高度
'''

import heapq
# 这个解法assume，L or R 这边不会有重叠的pins, emphasize on 一个屏幕内左右一共塞得进去多少pins
def get_max_pins(pins, screen_len):
    # Sort pins based on their end position
    pins = sorted(pins, key=lambda x: x[1])


    fitted_pins_heap = [] # Create this heap, to push out the pin with smallest start position
    max_pins = 0
    n=len(pins)
    for i in range(n):
        # screen_start,screen_end
        s_start, s_end = pins[i][1]-screen_len, pins[i][1]
        # Add current pin to fitted
        if pins[i][0] >= s_start:
            # 当前的pins 的开始是在屏幕下面的
            heapq.heappush(fitted_pins_heap, pins[i])
        # Pushes out pins that does not fit any more with the new end position
        while fitted_pins_heap and fitted_pins_heap[0][0] < s_start:
            heapq.heappop(fitted_pins_heap)
        max_pins = max(max_pins, len(fitted_pins_heap))
    return max_pins


pins=[(1, 4, 'L'), (2, 3, 'R'), (4, 8, 'R'), (6, 10, 'L')] 
screen_len=5
pins=[(1, 9, 'L'), (10, 18, 'R')] 
screen_len=7

pins=[(1, 4, 'L'), (4, 15, 'R'), (5, 6, 'L'), (2, 3, 'R')] 
screen_len=7
#Expected: 3 Output: 3

print(get_max_pins(pins, screen_len))


''''
Pins: [(1, 4, 'L'), (2, 3, 'R'), (4, 8, 'R'), (6, 10, 'L')] 
Screen Length=5
#Expected: 2 Output: 2
Pins: [(1, 4, 'L'), (2, 3, 'R'), (5, 6, 'L'), (4, 10, 'R')] Screen Length: 7
#Expected: 3 Output: 3
Pins: [(1, 4, 'L'), (4, 15, 'R'), (5, 6, 'L'), (2, 3, 'R')] Screen Length: 7
#Expected: 3 Output: 3
Pins: [(1, 3, 'L'), (2, 4, 'R'), (3, 6, 'L'), (4, 7, 'R'), (6, 9, 'L')] Screen Length: 5
#Expected: 3 Output: 3
Pins: [(1, 3, 'L'), (3, 6, 'R')] Screen Length: 5
#Expected: 2 Output: 2
Pins: [(0, 4, 'L'), (2, 6, 'R'), (4, 8, 'L')] Screen Length: 8
#Expected: 3 Output: 3
Pins: [(1, 4, 'L'), (2, 8, 'R'), (3, 9, 'L')] Screen Length: 6
#Expected: 1 Output: 1
Pins: [(1, 4, 'L'), (2, 7, 'R'), (3, 9, 'L')] Screen Length: 6
#Expected: 2 Output: 2
Pins: [(1, 3, 'L'), (2, 4, 'R'), (3, 6, 'L'), (4, 7, 'R'), (6, 9, 'L')] Screen Length: 5
#Expected: 3 Output: 3
Pins: [(1, 9, 'L'), (10, 18, 'R')] Screen Length: 7
#Expected: 0 Output: 0
'''