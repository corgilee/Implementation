
#https://www.1point3acres.com/bbs/thread-577581-1-1.html

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class MaxSubTree:
    max_sum = float('-inf')

    def max_sub_tree(root):
        if root is None:
            return 0
        MaxSubTree.helper(root)
        return MaxSubTree.max_sum


    def helper(node):
        if node is None:
            return 0
        # Calculate the sum of values in the current subtree
        cur = node.val + MaxSubTree.helper(node.left) + MaxSubTree.helper(node.right)
        print(cur)
        # Update the maximum sum found
        MaxSubTree.max_sum = max(cur, MaxSubTree.max_sum)
        return cur

# Example usage
if __name__ == "__main__":
    node1 = TreeNode(-100)
    node2 = TreeNode(20)
    node3 = TreeNode(10)
    node4 = TreeNode(-1)
    node5 = TreeNode(3)
    node6 = TreeNode(2)
    node7 = TreeNode(5)

    node1.left = node2
    node1.right = node3
    node2.left = node4
    node2.right = node5
    node3.left = node6
    node3.right = node7

    max_sum_found = MaxSubTree.max_sub_tree(node1)
    print("test")
    print(max_sum_found)
