'''
Music player
给一串字符串代表music playlist，判断是随机播放还是列表播放
注意列表播放 一轮轮完后下次的顺序可能不同
"ABCD CABD BCDA" 这是列表播放
"ABACBA" 这是随机播放
这道题考点可能在clarification，多给面试官提供example问问是不是expected：
1.两个methods的返回值都是boolean
2. 第一个method检测input是否有可能是random mode生成的
3. 第二个method检测input是否有可能是shuffle mode生成的
解题思路：
method1: always true，因为没有反例
method2: 没有duplicate就是true
follow-up：如果input是stream，怎么定义这两个method？
解题思路：和面试官clarify清楚，每个人可能解法不同。我这里用的是类似LFU来检测frequency

'''

def is_possible_random_mode(playlist):
    # In random mode, anything is possible (no constraint)
    return True

def is_possible_shuffle_mode(playlist):
    # Step 1: Split into segments (by space)
    cycles = playlist.split()
    
    if not cycles:
        return False

    # Step 2: Build expected set of unique songs from first cycle
    expected_set = set(cycles[0])

    for cycle in cycles:
        # Check length and uniqueness
        if set(cycle) != expected_set or len(cycle) != len(expected_set):
            return False

    return True

# ---- Follow up -----
'''
You can only read one song at a time.
You don’t know when a cycle ends unless song count matches previous cycle.
We need to track frequency.
'''
class PlaybackDetector:
    def __init__(self):
        self.freq = {}
        self.unique_set = set()
        self.last_song = None
        self.play_count = 0

    def process(self, song):
        self.freq[song] = self.freq.get(song, 0) + 1
        self.unique_set.add(song)
        self.play_count += 1

    def is_possible_shuffle(self):
        # All frequencies should either be equal, or at most one element has fewer
        counts = list(self.freq.values())
        return max(counts) - min(counts) <= 1

# how to apply on test case   
detector = PlaybackDetector()
for c in "A B C D A B C".split():
    detector.process(c)
    print(detector.is_possible_shuffle())



