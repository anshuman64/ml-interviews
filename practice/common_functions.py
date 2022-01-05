import numpy as np

"""
Tags: STUDYGUIDE
Ex: 1
Source: CTCI pg 50
Description: Determine if number is prime. Only need to check to sqrt(N)
Time:  O(sqrt(N))
Space: O(1)
"""
def isPrime(n : int) -> bool:
    for i in range(2, np.sqrt(n)):
        if (n % i == 0):
            return False
    
    return True

"""
Tags: STUDYGUIDE
Ex: 2
Source: CTCI pg 50
Description: Compute factorial
Time:  O(N)
Space: O(N)
"""
def factorial(n : int) -> int:
    if (n < 0):
        return -1
    elif (n == 0):
        return 1
    else:
        return n * factorial(n-1)

"""
Tags: STUDYGUIDE
Ex: 3
Source: CTCI pg 52
Description: Compute nth Fibonacci number
Time:  O(2^N) - branches = 2, depth = N
    Technically O(1.6^N), because bottom of tree sometimes has only 1 recursive call
Space: -
"""
def fib(n : int) -> int:
    if (n <= 0):
        return 0
    elif (n == 1): 
        return 1
    
    return fib(n-1) + fib(n-2)

"""
Tags: STUDYGUIDE
Ex: 4
Source: CTCI pg 52
Description: Brute force print all Fibonacci numbers from 1 to N
Time:  O(2^N) - O(2) + O(4) + ... + O(2^N) --> O(2^N)
Space: -
"""
def allFib(n : int):
    for i in range(n):
        print(i, fib(i))

"""
Tags: STUDYGUIDE
Ex: 5
Source: CTCI pg 53
Description: Cache print all Fibonacci numbers from 1 to N
Time:  O(N) - only compute two values per level
Space: -
"""
def allFib(n : int):
    memo = [0]*(n+1)

    for i in range(n):
        print(i, fib(i, memo))

def fib(n : int, memo : list) -> int:
    if (n <= 0): 
        return 0
    elif (n == 1): 
        return 1
    elif (memo[n] > 0): 
        return memo[n]
    else:
        memo[n] = fib(n-1, memo) + fib(n-2, memo)
        return memo[n]

"""
Tags: STUDYGUIDE
Ex: 6
Source: CTCI pg 53
Description: Prints powers of 2 from 1 to n
Time:  O(logN) - divide by two
Space: -
"""
def powersOf2(n : int) -> int:
    if (n < 1):
        return 0
    elif (n == 1):
        print(n)
        return 1
    else:
        prev = powersOf2(n / 2)
        curr = prev * 2
        print(curr)
        return curr

"""
Tags: STUDYGUIDE
Ex: 7
Source: CTCI pg 134
Description: Fibonacci with bottom up memoization 
Time:  O(n)
Space: -
"""
def fib(n);
    if (n == 0):
        return 0
    elif (n == 1):
        return 1

    memo = [0] * n
    memo[0] = 0
    memo[1] = 1

    for i in range(2, n):
        memo[i] = memo[i-1] + memo[i-2]

    return memo[n-1] + memo[n-2]

def fib(n):
    if (n == 0):
        return 0

    a = 0
    b = 1

    for i in range(2, n):
        c = a + b
        a = b
        b = c

    return a + b
