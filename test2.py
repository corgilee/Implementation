def max_leaf_to_leaf_sum(root):
    if not root:
        return 0

    best = float("-inf")
    leaf_count = 0

    def dfs(node):
        nonlocal best, leaf_count
        if not node:
            return None  # means "no path"

        # leaf
        if not node.left and not node.right:
            leaf_count += 1
            return node.val

        left = dfs(node.left)
        right = dfs(node.right)

        # if both sides exist, we can form a leaf-to-leaf path through this node
        if left is not None and right is not None:
            best = max(best, left + node.val + right)
            return node.val + max(left, right)

        # otherwise only one side exists -> must go down that side
        if left is not None:
            return node.val + left
        else:
            return node.val + right  # right must exist here

    root_to_leaf = dfs(root)
    return best if leaf_count >= 2 else root_to_leaf
