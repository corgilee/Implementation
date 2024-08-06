from collections import defaultdict, deque

class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        '''
        Time complexity:
        O(V+E), V is the total number of course, E the number of prerequisites
        https://blog.csdn.net/fuxuemingzhu/article/details/83302328

        
        这个是topological sorting 问题，不要用backtracking思想(backtracking 思想适用于find all possiblity)
        不管是course reschedule i/ii 都用这个解法
        build a list of n to remember state (0/1/2), shows the state of each course
        dfs -> bool, 
        '''
        # Build a graph of prerequisites
        graph = collections.defaultdict(list)
        for crs, pre in prerequisites:
            graph[crs].append(pre)

        # build a list of state
        # States: 0 = unvisited, 1 = visiting, 2 = visited
        visit = [0] * numCourses
        res = []

        def dfs(crs):
            if visit[crs] == 1:  # Cycle detected
                return False
            if visit[crs] == 2:  # Already visited
                return True

            # Mark the node as visiting
            visit[crs] = 1
            for pre in graph[crs]:
                if not dfs(pre):
                    return False

            # Mark the node as visited
            visit[crs] = 2
            res.append(crs)
            return True

        # Perform DFS for each course
        for i in range(numCourses):
            if not dfs(i):
                return []

        return res
