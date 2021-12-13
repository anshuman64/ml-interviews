"""
Tags: STUDYGUIDE
Ex: 1
Source: CTCI pg 48
Description: Reverse an array in place
Time:  O(N)
Space: O(N)
"""
def reverse(arr : list) -> list:
    for i in range(len(arr)/2):
        other_idx = len(arr) - i - 1
        temp = arr[i]

        arr[i] = arr[other_idx]
        arr[other_idx] = temp 

    return arr