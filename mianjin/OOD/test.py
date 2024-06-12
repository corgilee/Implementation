import numpy as np
def is_numeric(value):
    return isinstance(value, (int, float))

# Example usage
print(is_numeric(10))       # True
print(is_numeric(10.5))     # True
print(is_numeric(np.nan))     # False
print(is_numeric([10]))     # False
