from collections import defaultdict
from collections import deque


def getRatio(start, end, data):
    dict = defaultdict(list)
    for node in data:
        dict[node[0]].append([node[1], node[2]])
        dict[node[1]].append([node[0], 1.0 / node[2]])
    queue = deque()
    queue.append((start, 1.0))
    visited = set()
    while queue:
        curr, num = queue.popleft()
        if curr in visited:
            continue
        visited.add(curr)
        if curr in dict:
            values = dict.get(curr)
            next = {}
            for val in values:
                next[val[0]] = val[1]
            for key in next:
                if key not in visited:
                    if key == end:
                        return num * next[key]
                    queue.append((key, num * next[key]))
    return -1

