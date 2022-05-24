################
# Bubble Sort
################

def bubbleSort(arr):
    did_swap = False

    for i in range(1, len(arr)):
        # Swap values if previous is larger than current
        if arr[i-1] > arr[i]:
            arr[i-1], arr[i] = arr[i], arr[i-1]

            did_swap = True

    # Re-run if swapped once
    if did_swap:
        bubbleSort(arr)


################
# Selection Sort
################

def selectionSort(arr):
    for swap_idx in range(len(arr)):

        # Find index of smallest value of remaining array
        smallest_idx = swap_idx
        for j in range(swap_idx+1, len(arr)):
            if arr[j] < arr[smallest_idx]:
                smallest_idx = j

        # Swap with current index
        arr[swap_idx], arr[smallest_idx] = arr[smallest_idx], arr[swap_idx]


################
# Insertion Sort
################

def insertionSort(arr):
    for swap_idx in range(1, len(arr)):
        swap_value = arr[swap_idx]

        # Loop backward from swap_idx
        for j in range(swap_idx-1, -1, -1):

            # Push up values bigger than swap_value
            if arr[j] > swap_value:
                arr[j+1] = arr[j]

            # Place swap value in correct spot
            else:
                arr[j+1] = swap_value
                break


################
# MergeSort
################

def mergeSort(arr):
    # Base case!
    if len(arr) == 1:
        return arr

    # Split array in half
    mid_idx = len(arr)//2
    left_arr = arr[:mid_idx]
    right_arr = arr[mid_idx:]

    mergeSort(left_arr)
    mergeSort(right_arr)
    mergeSortHelper(arr, left_arr, right_arr)


def mergeSortHelper(arr, left_arr, right_arr):
    l = r = a = 0

    # Replace values of arr with smaller value of left or right
    while l < len(left_arr) and r < len(right_arr):
        if left_arr[l] < right_arr[r]:
            arr[a] = left_arr[l]
            l += 1
        else:
            arr[a] = right_arr[r]
            r += 1

        a += 1

    # Handle remaining values
    while l < len(left_arr):
        arr[a] = left_arr[l]
        l += 1
        a += 1

    while r < len(right_arr):
        arr[a] = right_arr[r]
        r += 1
        a += 1


################
# QuickSort
################

def quickSort(arr):
    quickSortHelper(arr, 0, len(arr))


def quickSortHelper(arr, start_idx, end_idx):
    if start_idx >= end_idx:
        return

    # Partition
    pivot_idx = start_idx
    for i in range(start_idx+1, end_idx):
        if arr[i] < arr[pivot_idx]:
            # Increase pivot position
            pivot_idx += 1
            # Swap with smaller value
            arr[i], arr[pivot_idx] = arr[pivot_idx], arr[i]

    # Swap new pivot position with initial pivot
    arr[start_idx], arr[pivot_idx] = arr[pivot_idx], arr[start_idx]

    quickSortHelper(arr, start_idx, pivot_idx-1)
    quickSortHelper(arr, pivot_idx+1, end_idx)


################
# Binary Search
################

def binarySearch(arr, value):
    low = 0
    high = len(arr) - 1

    while low <= high:
        # Avoid overflow
        mid = low + (high - low) // 2

        if (arr[mid] < value):
            low = mid + 1
        elif (arr[mid] > value):
            high = mid - 1
        else:
            return True

    return False


def binarySearch2(arr, value):
    low = 0
    high = len(arr)

    while low < high:
        # Avoid overflow
        mid = low + (high - low) // 2

        if arr[mid] < value:
            low = mid + 1
        else:
            high = mid

    return low


def binarySearchRecursive(arr, value):
    return binarySearchHelper(arr, value, 0, len(arr)-1)


def binarySearchHelper(arr, value, low, high):
    if (low > high):
        return False

    mid = low + (high - low) // 2
    if (arr[mid] < value):
        return binarySearchHelper(arr, value, mid+1, high)
    elif (arr[mid] > value):
        return binarySearchHelper(arr, value, low, mid-1)
    else:
        return True


#################
# Quick Select
#################

def quickSelect(arr, k):
    """
    Description: Find top K values with partial sorting
    Runtime:
        - Average: O(n)
        - Worst: O(n^2) if pivot is smallest/largest element
    """
    # Initialize two pointers on opposite ends
    left = 0
    right = len(arr) - 1
    # Initialize sorted_idx
    sorted_idx = len(arr)

    # While we haven't sorted until k
    while sorted_idx != k:
        # Sort & get new sorted_idx
        sorted_idx = quickSelectHelper(arr, left, right)

        # If sorted_idx < k, sort the rest
        if sorted_idx < k:
            left = sorted_idx  # left is past the sorted
        # Else, make sure first half properly sorted
        else:
            right = sorted_idx - 1  # look at only the sorted stuff

    return arr[:k]


def quickSelectHelper(arr, left, right):
    # Given: two pointers on opposite ends
    # Find pivot as middle
    pivot = arr[left + (right - left) // 2]

    while left < right:
        # If left >= pivot
        if arr[left] >= pivot:
            # Swap
            arr[left], arr[right] = arr[right], arr[left]
            right -= 1
        else:
            left += 1

    # Ensure left is past all points lower than pivot
    if arr[left] <= pivot:
        left += 1

    # Return index we've sorted until
    return left


################
# Testing
################

arr = [1, 9, 3, 3, 2, 8, 6]
# arr = [3,1,4,2,5]
# bubbleSort(arr)
# print(arr)
# selectionSort(arr)
# print(arr)
# insertionSort(arr)
# print(arr)
# mergeSort(arr)
# print(arr)
# quickSort(arr)
# print(arr)
print(quickSelect(arr, 3))

# arr = [1, 2, 3, 3, 6, 8, 9]
# assert binarySearchRecursive(arr, 9)
# assert binarySearchRecursive(arr, 1)
# assert binarySearchRecursive(arr, 6)
# assert not binarySearchRecursive(arr, 7)
