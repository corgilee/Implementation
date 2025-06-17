'''
meeting room II Follow up given a list of time, query the number of rooms in used at that timestamp

'''
import bisect

class MeetingRoomCounter:
    def __init__(self, intervals):
        # O(nlog n)
        self.starts = sorted(s for s,_ in intervals)
        self.ends   = sorted(e for _,e in intervals)

    def query(self, t: int) -> int:
        # bisect 就是找到<=x 的个数
        cnt_start = bisect.bisect_right(self.starts, t) #在<=t时间开始的会
        cnt_end   = bisect.bisect_right(self.ends,   t) #在<=t时间结束的会
        return cnt_start - cnt_end
    

# test case
# 1) Basic example from LC 253
intervals = [[0, 30], [5, 10], [15, 20]]
counter = MeetingRoomCounter(intervals)
# At t=0: only [0,30) ⇒ 1 room
assert counter.query(0)   == 1
# At t=5: [0,30), [5,10) ⇒ 2 rooms
assert counter.query(5)   == 2
# At t=10: [0,30) still open (10 < 30), but [5,10) ends at 10 ⇒ count as closed ⇒ 1
assert counter.query(10)  == 1
# At t=15: [0,30), [15,20) ⇒ 2
assert counter.query(15)  == 2
# At t=20: [15,20) ends at 20 ⇒ closed, so 1
assert counter.query(20)  == 1
# At t=30: [0,30) ends ⇒ 0
assert counter.query(30)  == 0


# 2) No meetings
empty = MeetingRoomCounter([])
for t in [ -1, 0, 5, 100 ]:
    assert empty.query(t) == 0

# 3) Non–overlapping meetings
intervals = [[1,2], [3,4], [5,6]]
c = MeetingRoomCounter(intervals)
assert c.query(0) == 0
assert c.query(1) == 1  # only [1,2)
assert c.query(2) == 0  # boundary
assert c.query(3) == 1
assert c.query(4) == 0
assert c.query(5) == 1
assert c.query(6) == 0

# 4) Completely nested intervals
intervals = [[0,100], [10,90], [20,80], [30,70]]
c = MeetingRoomCounter(intervals)
assert c.query(0)   == 1
assert c.query(15)  == 2   # [0,100], [10,90]
assert c.query(25)  == 3   # [0,100], [10,90], [20,80]
assert c.query(35)  == 4
assert c.query(70)  == 3   # [30,70) ends at 70, so down to 3
assert c.query(100) == 0

# 5) Meetings back–to–back
intervals = [[1,3], [3,5], [5,7]]
c = MeetingRoomCounter(intervals)
assert c.query(2) == 1
assert c.query(3) == 1  # [1,3) closed at 3; [3,5) open
assert c.query(4) == 1
assert c.query(5) == 1
assert c.query(6) == 1
assert c.query(7) == 0