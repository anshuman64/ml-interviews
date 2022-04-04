"""
Tags: STUDYGUIDE
Ex: 1
Source: CTCI pg 46
Time:  O(N) - two independent loops
Space: O(1)
"""


def foo(arr: list):
    for i in range(len(arr)):
        print(i)

    for i in range(len(arr)):
        print(j)


"""
Tags: STUDYGUIDE
Ex: 2
Source: CTCI pg 46
Time:  O(N^2) - two independent loops
Space: O(1)
"""


def foo(arr: list):
    sum = 0
    product = 1

    for i in range(len(arr)):
        for j in range(len(arr)):
            print(i, j)


"""
Tags: IMPORTANT STUDYGUIDE
Ex: 3
Source: CTCI pg 46
Time:  O(N^2)
    Infinite Sum: N-1 + N-2 + ... + 2 + 1 = N(N-1)/2 = O(N^2)
    Draw It: Half of a NxN matrix of pairs of numbers
    Average Work: Half-th iteration would be N/2*N = O(N^2)
Space: O(1)
"""


def foo(arr: list):
    sum = 0
    product = 1

    for i in range(len(arr)):
        for j in range(i, len(arr)):
            print(i, j)


"""
Tags: STUDYGUIDE
Ex: 4
Source: CTCI pg 46
Time:  O(AB)
Space: O(1)
"""


def foo(arr1: list, arr2: list):
    sum = 0
    product = 1

    for i in range(len(arr1)):
        for j in range(i, len(arr2)):
            print(i, j)


"""
Tags: IMPORTANT STUDYGUIDE
Ex: 5
Source: CTCI pg 45
Description: Add binary tree of numbers
Time:  O(2^N) - binary tree of depth N
Space: O(N) - max width of tree = N
"""


def f(n: int):
    if (n <= 1):
        return 1

    return f(n-1) + f(n-1)
