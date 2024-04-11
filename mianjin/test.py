class TreeNode:
    def __init__(self,val):
        self.val=val
        self.left=None
        self.right=None

class MaxSubTree(Node):
    max_sum=float(-inf)

    def helper(Node):
        if Node is None:
            return 0
        cur=Node.val+self.helper(Node.left)+self.helper(Node.right)
        max_sum=max(cur,max_sum)

    if root is None:
        return 0
        
    helper(root)
    return max_sum

        



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

    max_sum_found = MaxSubTree(node1)
    print("test")
    print(max_sum_found)
