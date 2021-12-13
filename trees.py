"""
Tags: IMPORTANT STUDYGUIDE
Ex: 1
Source: CTCI pg 49
Description: Sum values of nodes in balanced binary search tree
Time:  O(N)
    Straightforward: adds each of N nodes
    Recursive Formula: branches=2, depth=logN. O(2^(logN)) = O(N)
Space: O(logN)
"""
def sum(node : Node) -> float:
    if (node == null):
        return 0
    
    return sum(node.left) + node.value + sum(node.right)