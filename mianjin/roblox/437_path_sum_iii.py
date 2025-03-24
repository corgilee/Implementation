'''
Optimized Solution: Prefix Sum + Hash Map

We apply a technique similar to the "subarray sum equals k" problem:
Traverse the tree using DFS.
Use a prefix sum hash map prefix_sum_count to store how many times a sum has occurred from the root to the current node.

For each node:
    Let curr_sum be the sum from the root to the current node.
    If there’s a prefix sum curr_sum - targetSum, then there exists a path from some ancestor node to the current node whose sum is targetSum.

This approach allows us to compute valid paths in O(n) time.

Time	O(n) — Each node visited once
Space	O(h + n), h is the height of the tree
'''

from collections import defaultdict

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def pathSum(root, targetSum):
    prefix_sum_count = defaultdict(int) #dict 存的是不同sum的个数
    prefix_sum_count[0] = 1  # Base case: a path starting at the root

    def dfs(node, curr_sum):
        if not node:
            return 0

        curr_sum += node.val
        count = prefix_sum_count[curr_sum - targetSum]  # Paths ending here that sum to target

        prefix_sum_count[curr_sum] += 1  # Add current prefix sum to map

        # Recurse left and right
        count += dfs(node.left, curr_sum)
        count += dfs(node.right, curr_sum)

        prefix_sum_count[curr_sum] -= 1  # Backtrack

        return count

    return dfs(root, 0)
