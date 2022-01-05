#####################
## Complexity
#####################

"""
Ex: 1
Source: CTCI pg 41
Description: Recursively adds numbers from n to 0
Time:  O(N)
Space: O(N) - each recursion takes more space
"""
def sum(n : int) -> int:
    if (n <= 0):
        return 0

    return n + sum(n-1)

"""
Ex: 2
Source: CTCI pg 41
Description: Iteratively add pairs of numbers
Time:  O(N)
Space: O(1) - replacing variable in for loop
"""
def pairSumSequence(n : int) -> int:
    sum = 0

    for i in range(n):
        sum += pairSum(i, i+1)
    
    return sum 

def pairSum(a, b) -> int:
    return a + b


"""
Ex: 8
Source: CTCI pg 46
Time:  O(AB) = O(1000AB)
Space: O(1)
"""
def foo(arr1 : list, arr2 : list):
    sum = 0
    product = 1

    for i in range(len(arr1)):
        for j in range(i, len(arr2)):
            for k in range(1000):
                print(i, j, k)

