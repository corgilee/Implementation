'''
Count frequency of each task.
Use a max-heap to always pick the most frequent task.
Use a cooldown queue to track tasks waiting to be reused.
Simulate time steps — at each step, pick from heap or idle if nothing is available.

Time: O(N log N) where N is number of tasks
'''
from collections import Counter, deque
import heapq

def leastInterval(self, tasks: List[str], n: int) -> int:
        # Step 1: Count frequency of each task
        counter = Counter(tasks)
        
        # Step 2: Use max-heap (invert count to make Python's min-heap into max-heap)
        max_heap = [-cnt for cnt in counter.values()]
        heapq.heapify(max_heap)

        # Cooldown queue: stores (ready_time, count)
        cooldown = deque()
        time = 0

        while max_heap or cooldown:
            time += 1

            if max_heap:
                cnt = heapq.heappop(max_heap) + 1  # do task, so reduce count
                if cnt != 0:
                    # Add to cooldown, available after `time + n`
                    cooldown.append((time + n, cnt))
            
            # Check if any task is ready to be reused
            if cooldown and cooldown[0][0] == time:
                heapq.heappush(max_heap, cooldown.popleft()[1])

        return time