edges = [[0,1,4],[1,2,3],[1,3,2],[3,4,6]]

print(edges)

edges.sort(key=lambda x: x[2])

print(edges)