import numpy as np
tree = __import__('7_Tree')

""" Ex: 
    Source: CTCI pg 150
    Description: Given sorted array that has been rotated, find idx of ele
    Time:  

    Example: [7,9,10,1,3,4], 3 -> 4
    """


def GroupAnagrams(strings):
    groups = {}

    for string in strings:
        sorted_string = ''.join(sorted(string))  # O(slogs)
        if sorted_string not in groups:
            groups[sorted_string] = []
        groups[sorted_string].append(string)

    sorted_list = []
    for group in groups:  # O(n)
        sorted_list.extend(groups[group])

    return sorted_list


# A = [1, 3, 5, 7, None, None, None]
# B = [2, 4, 6]
# print(SortedMerge(A, B))

print(GroupAnagrams(['friend', 'listen', 'firend', 'silent']))
