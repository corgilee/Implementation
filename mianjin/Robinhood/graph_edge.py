
'''
返回每个node的 孩子数量
'''

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
        if not children:
            print('outstanding',node,children)
            queue.append(node)
    
    while queue:
        print('queue',queue)
        cur = queue.popleft()
        for parent in connect_map[cur]:
            child_map[parent].update(child_map[cur])
            queue.append(parent)
    
    res = [NodeCount(node, len(children)) for node, children in child_map.items()]
    return sorted(res, key=lambda x: -x.count)

# Example usage
if __name__ == "__main__":
    graphs1 = [
        ["A", "B"],
        ["B", "C"],
        ["C", "D"]
    ]
    res1 = sorted_by_number_of_child(graphs1)
    print('graph_1')
    for node_count in res1:
        print(node_count.node, node_count.count)


    graphs2 = [
        ["A", "B"],
        ["A", "C"],
        ["B", "C"]
    ]
    res2 = sorted_by_number_of_child(graphs2)
    print('graph_2')
    for node_count in res2:
        print(node_count.node, node_count.count)

