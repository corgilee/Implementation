'''
https://docs.google.com/document/d/19SVzrDQ5UKxC03Tco2eaXVlePT6O9hYE--HUt_pisFM/edit
'''


# solution from chatgpt
class DecisionTree:
    def __init__(self, value=None):
        self.value = value
        self.left = None
        self.right = None
        self.split_condition = None

    def add_split(self, leaf, signal_name, constant):
        # Add a split condition to the given leaf node
        leaf.left = DecisionTree()
        leaf.right = DecisionTree()
        leaf.split_condition = (signal_name, constant)
        return leaf.left, leaf.right

    def set_leaf_value(self, leaf, value):
        # Set the return value for a leaf node
        leaf.value = value

    def evaluate(self, signals):
        # Evaluate the tree on a mapping of signal_name -> signal_value
        if self.split_condition:
            signal_name, constant = self.split_condition
            signal_value = signals.get(signal_name)
            if signal_value is None:
                raise ValueError(f"Signal {signal_name} not found in signals.")
            if signal_value < constant:
                return self.left.evaluate(signals)
            else:
                return self.right.evaluate(signals)
        else:
            return self.value

# Example usage and test cases:
if __name__ == "__main__":
    # Grow the example tree
    tree = DecisionTree(value='Y')
    left, right = tree.add_split(tree, 'X1', 3)
    left, _ = tree.add_split(left, 'X2', 1)
    tree.set_leaf_value(left.left, 'N')
    tree.set_leaf_value(left.right, 'Y')
    right, _ = tree.add_split(right, 'X1', 6)
    _, right = tree.add_split(right, 'X3', 2)
    tree.set_leaf_value(right.left, 'Y')
    tree.set_leaf_value(right.right, 'N')

    # Test cases
    signals1 = {'X1': 2, 'X2': 1, 'X3': 11}
    print("Result for signals1:", tree.evaluate(signals1))  # Expected: Y

    signals2 = {'X1': 8, 'X2': 4, 'X3': 12}
    print("Result for signals2:", tree.evaluate(signals2))  # Expected: N


