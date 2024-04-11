from collections import defaultdict, deque

class NodeCount:
    def __init__(self, node, count):
        self.node = node
        self.count = count

def sorted_by_number_of_child(graphs):
    if not graphs:
        return []

    child_map = defaultdict(set)
    connect_map = defaultdict(list)

    for g in graphs:
        child_map[g[0]].add(g[1])
        connect_map[g[1]].append(g[0])

    queue = deque()

    for node, children in child_map.items():
        if len(children) == 0:
            queue.append(node)

    while queue:
        print(f"Current queue: {[node for node in queue]}")  # Print the current queue
        cur = queue.popleft()
        for parent in connect_map[cur]:
            child_map[parent].update(child_map[cur])
            if parent not in queue:
                queue.append(parent)

    res = [NodeCount(node, len(descendants)) for node, descendants in child_map.items()]
    res.sort(key=lambda x: x.count, reverse=True)

    return res

# Example usage
if __name__ == "__main__":
    graphs = [
        ["A", "B"],
        ["B", "C"],
        ["C", "D"],
        ["A", "E"]
    ]
    result = sorted_by_number_of_child(graphs)
    for node_count in result:
        print(f"Node: {node_count.node}, Descendant Count: {node_count.count}")
