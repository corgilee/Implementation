'''
Use a deque (from collections) to store timestamps of hits.
Each time a hit is recorded, append the timestamp.
On getHits(timestamp), remove old timestamps that are older than timestamp - 300.

time:
hit(t),O(1), Append to the queue
getHits(t), O(n), worst case	Remove outdated timestamps
Space, O(n), Store at most n hits in 300s
'''

class HitCounter:
    def __init__(self):
        # Initialize a deque to store timestamps of hits
        self.hits = deque()
        
    def hit(self, timestamp: int) -> None:
        # Add current hit's timestamp to the deque
        self.hits.append(timestamp)

    def getHits(self, timestamp: int) -> int:
        # Remove timestamps older than 300 seconds from the front
        while self.hits and self.hits[0] <= timestamp - 300:
            self.hits.popleft()
        # Remaining timestamps are within the past 300 seconds
        return len(self.hits)


# Your HitCounter object will be instantiated and called as such:
# obj = HitCounter()
# obj.hit(timestamp)
# param_2 = obj.getHits(timestamp)