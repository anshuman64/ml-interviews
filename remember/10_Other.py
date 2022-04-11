##############
# LRU Cache
##############

from collections import OrderedDict


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


###################
# Sliding Window
###################

def SlidingWindow(self, s: str, t: str) -> str:
    t_dict = dict(Counter(t))
    s_dict = {char: 0 for char in t_dict}

    ok = 0
    l = 0
    r = 0
    to_return = s+t

    ### Expand ###
    while r < len(s):
        if s[r] in s_dict:
            s_dict[s[r]] += 1

            if s_dict[s[r]] == t_dict[s[r]]:
                ok += 1

        ### Contract ###
        while l <= r and ok == len(t_dict):
            to_return = s[l:r+1] if len(s[l:r+1]
                                        ) < len(to_return) else to_return

            if s[l] in s_dict:
                s_dict[s[l]] -= 1

                if s_dict[s[l]] < t_dict[s[l]]:
                    ok -= 1

            l += 1

        r += 1

    return to_return if len(to_return) != len(s)+len(t) else ''


################
# Cyclic Sort
################

def CyclicSort(nums):
    i = 0
    n = len(nums)

    while i < n:
        j = nums[i] - 1

        # Swap if idx exists
        if 0 <= j < n and nums[i] != nums[j]:
            nums[i], nums[j] = nums[j], nums[i]
        else:
            i += 1

    # Find missing value
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1

    return n + 1
