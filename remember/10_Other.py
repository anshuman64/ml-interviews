##############
# LRU Cache
##############

from collections import Counter, OrderedDict


class LRUCache:
    def __init__(self, capacity: int):
        self.dict = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.dict:
            return -1

        self.dict.move_to_end(key)
        return self.dict[key]

    def put(self, key: int, value: int) -> None:
        self.dict[key] = value
        self.dict.move_to_end(key)

        if len(self.dict) > self.capacity:
            self.dict.popitem(last=False)


################
# Cyclic Sort
################

def CyclicSort(nums):
    """
    Description: sort array of numbers in range to find missing numbers
    Runtime: O(n)
    """
    i = 0

    # Loop through array
    while i < len(nums):
        # Find idx of num
        j = nums[i] - 1

        # Swap if idx exists and not equal
        if 0 <= j < len(nums) and nums[i] != nums[j]:
            nums[i], nums[j] = nums[j], nums[i]
        # Else, increase i
        else:
            i += 1

    # Find missing value
    for i in range(len(nums)):
        if nums[i] != i + 1:
            return i + 1

    # Edge case: missing value is last num
    return len(nums) + 1


###################
# Sliding Window
###################

def SlidingWindow(self, s: str, t: str) -> str:
    """
    Description: Determine shortest subsequence eligible
    Runtime: O(n)
    Pseudocode:
        l = r = ok = 0

        for r in range(len(arr)):
            if arr[r] == condition:
                ok += 1

            while l <= r and ok == condition:
                add to return

                if arr[l] == condition:
                    ok -= 1

                l += 1
    """
    t_dict = dict(Counter(t))
    s_dict = {char: 0 for char in t_dict}

    ok = 0
    l = 0
    to_return = s+t

    ### Expand ###
    for r in range(len(s)):
        if s[r] in s_dict:
            s_dict[s[r]] += 1

            if s_dict[s[r]] == t_dict[s[r]]:
                ok += 1

        ### Contract ###
        while l <= r and ok == len(t_dict):
            if len(s[l:r+1]) < len(to_return):
                to_return = s[l:r+1]

            if s[l] in s_dict:
                s_dict[s[l]] -= 1

                if s_dict[s[l]] < t_dict[s[l]]:
                    ok -= 1

            l += 1

    return to_return if len(to_return) != len(s)+len(t) else ''
