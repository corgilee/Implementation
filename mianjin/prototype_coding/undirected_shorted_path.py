import heapq

def shortest_path_undirected(graph, start, end):
    """
    graph: dict, 例如 { 'A': [('B', 5), ('C', 2)], ... }
    start: 起点 node
    end:   终点 node
    
    return 最短距离，如果 unreachable 则返回 None
    """
    # dist 初始化
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    
    pq = [(0, start)]  # (time, node)

    while pq:
        time_u, u = heapq.heappop(pq)
        if time_u > dist[u]:
            continue  # stale entry

        if u == end:
            return time_u   # 提前结束：到终点了

        for v, w in graph[u]:
            new_t = time_u + w
            if new_t < dist[v]:
                dist[v] = new_t
                heapq.heappush(pq, (new_t, v))
    
    return None  # end unreachable

### test case ###

graph = {
    'A': [('B', 5), ('C', 2)],
    'B': [('A', 5), ('C', 1)],
    'C': [('A', 2), ('B', 1)]
}

print(shortest_path_undirected(graph, 'A', 'B'))

graph = {
    'A': [('B', 3)],
    'B': [('A', 3)],
    'C': []  # isolated
}

print(shortest_path_undirected(graph, 'A', 'C'))