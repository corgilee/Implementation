'''
maximize task profits by their ddl.  input: [(taskName, ddl, profit)], output: ([taskOrder], maxProfit)
每个task只需要花一天时间，一天只可以做一个task
'''

'''
按 deadline 从小到大遍历任务
用 min-heap 存当前被选中的任务利润（和名字）
每加入一个任务，就表示“我想把它也做了”
如果当前选中任务数 > 当前 deadline，说明时间不够做这么多任务 → 把 heap 里 profit 最小的那个踢掉
最后 heap 里剩下的任务集合就是最大利润集合
复杂度： O(n log n)，
'''

import heapq


def maximize_task_profit_heap(tasks):
    # sort by deadline
    tasks = sorted(tasks, key=lambda x: x[1]) # (name, ddl, profit)

    heap = [] # min-heap of (profit, name)
    total = 0

    for name, ddl, profit in tasks:
    heapq.heappush(heap, (profit, name))
    total += profit

    # too many tasks for available days up to ddl
    if len(heap) > ddl:
    p, _ = heapq.heappop(heap)
    total -= p

    # heap contains chosen tasks, but NOT a concrete day-by-day schedule yet
    chosen = [name for _, name in heap]
    return chosen, total