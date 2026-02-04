'''
lc 253
'''
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        #先sort不要忘了,用开始时间sort, sort 开始时间
        # minheap ((end_time_1),(end_time_2))
        intervals.sort()
        res=1
        q=[]
        heapq.heappush(q, intervals[0][1])
        for start, end in intervals[1:]:
            #option 1 start time<top end
            if start<q[0]:
                heapq.heappush(q,(end))

            #option 2 start time> top end
            else:
                while q and start>=q[0]:
                    heapq.heappop(q)
                heapq.heappush(q,end)

            res=max(res, len(q))
        return res


'''
原题问完又问了一个FOLLOWUP,　让返回LIST OF LIST，每个LIST是一个MEETING ROOM所HOST的meeting的INTERVALS

'''

def assign_meeting_rooms(intervals):
    """
    intervals: List[List[int]] with [start, end]
    returns: rooms, where rooms[i] is the list of intervals hosted by room i
    """
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])

    heap = []  # (end_time, room_id)
    rooms = [] # rooms[room_id] = list of [start, end]

    for start, end in intervals:
        if heap and heap[0][0] <= start:
            _, room_id = heapq.heappop(heap)
        else:
            room_id = len(rooms)
            rooms.append([])

        rooms[room_id].append([start, end])
        heapq.heappush(heap, (end, room_id))

    return rooms