"""
Tags: STUDYGUIDE
Ex: 1
Source: CTCI pg 51
Description: Print all permutations of string
Time:  O(N * N!) - N! permutations, print = O(N) 
"""
def permutation(s : str, prefix : str = ""):
    if (len(s) == 0):
        print(prefix)
    else:
        for i in range(len(s)):
            remaining = s[:i] + s[i+1:]
            permutation(remaining, prefix + s[i])