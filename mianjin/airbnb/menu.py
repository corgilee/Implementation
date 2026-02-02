'''

Build a function that lets the user determine the most cost-effective order. Input includes the menu of the restaurant that contains the item name and its price. And the item the user wants to order. Except for single items, the menu also offers combo items, which are groups of several items at a discounted price. Return the lowest price at which they can get all of their desired items and the item the user needs to order on the menu.
Constraint: The user can want a maximum of 3 unique items.
e.g.
Menu:
[[5.00, "pizza"],
[8.00, "sandwich,coke"],
[4.00, "pasta"],
[2.00, "coke"],
[6.00, "pasta,coke,pizza"],
[8.00, "burger,coke,pizza"],
[5.00, "sandwich"]]
user_wants: ["sandwich", "pasta", "coke"]
Expected answer: lowest cost = 11, order path = [ [ 2, 3, 6], [4,6] ]
用DP+bitmask。想练手的可以参考LC638

'''