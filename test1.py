class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def max_leaf_to_leaf_sum(root):
    # We want the max path sum where BOTH endpoints are leaves ("alive nodes").
    # We'll do a postorder traversal:
    # - dfs(node) returns the best sum from node down to ANY leaf in its subtree.
    # - Meanwhile, if node has both children, we can form a leaf->node->leaf path.

    if not root:
        return 0  # depending on spec; could also be -inf

    best = float("-inf")      # global best leaf-to-leaf path
    leaf_count = 0            # to handle "only one leaf exists" case safely

    def dfs(node):
        nonlocal best, leaf_count
        if not node:
            return float("-inf")  # impossible path (so it won't be chosen accidentally)

        # If it's a leaf, it's an "alive node"
        if not node.left and not node.right:
            leaf_count += 1
            return node.val

        # Compute best node->leaf sums from children
        left_down = dfs(node.left) if node.left else float("-inf")
        right_down = dfs(node.right) if node.right else float("-inf")

        # If both children exist, we can connect a leaf on the left + node + a leaf on the right
        if node.left and node.right:
            candidate = left_down + node.val + right_down
            best = max(best, candidate)

        # Return the best "downward" path from this node to a leaf
        return node.val + max(left_down, right_down)

    best_root_to_leaf = dfs(root)

    # If we had fewer than 2 leaves (e.g., single node or a linked-list-shaped tree),
    # there is no "two-leaf" path. We'll treat answer as best root-to-leaf (leaf to itself).
    return best if leaf_count >= 2 else best_root_to_leaf


# ---- 1 test case (the screenshot example) ----
#       5
#     /   \
#    2     8
#   /     / \
# 25     14  15
root = Node(5,
            Node(2, Node(25)),
            Node(8, Node(14), Node(15)))

print(max_leaf_to_leaf_sum(root))  # expected 47  (25 + 2 +
