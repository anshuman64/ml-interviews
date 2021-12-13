################
## Bubble Sort
################

def bubbleSort(arr):
    did_swap = False
    
    for i in range(1, len(arr)):
        # Swap values if previous is larger than current
        if arr[i-1] > arr[i]:
            temp = arr[i-1]
            arr[i-1] = arr[i]
            arr[i] = temp

            did_swap = True
    
    # Re-run if swapped once
    if did_swap:
        bubbleSort(arr)


################
## Selection Sort
################

def selectionSort(arr):
    for swap_idx in range(len(arr)):

        # Find index of smallest value of remaining array
        smallest_idx = swap_idx
        for j in range(swap_idx+1, len(arr)):
            if arr[j] < arr[smallest_idx]:
                smallest_idx = j
        
        # Swap with current index
        temp = arr[swap_idx]
        arr[swap_idx] = arr[smallest_idx]
        arr[smallest_idx] = temp

################
## Insertion Sort
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
## QuickSort
################

# def quickSort(arr):


################
## MergeSort
################

def mergeSort(arr):
    if len(arr) == 1:
        return arr
    
    # Split array in half
    mid_idx = len(arr)//2
    left_arr  = arr[:mid_idx]
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
## Binary Search
################

def binarySearch(arr, value):
    low = 0
    high = len(arr) - 1

    while(low <= high):
        mid = (low + high) // 2

        if (arr[mid] < value):
            low = mid + 1
        elif (arr[mid] > value):
            high = mid - 1
        else:
            return True

    return False

def binarySearchRecursive(arr, value):
    return binarySearchHelper(arr, value, 0, len(arr)-1)

def binarySearchHelper(arr, value, low, high):
    if (low > high):
        return False
    
    mid = (low + high) // 2
    if (arr[mid] < value):
        return binarySearchHelper(arr, value, mid+1, high)
    elif (arr[mid] > value):
        return binarySearchHelper(arr, value, low, mid-1)
    else:
        return True


            

arr = [1, 9, 3, 3, 2, 8, 6]
bubbleSort(arr)
print(arr)
selectionSort(arr)
print(arr)
insertionSort(arr)
print(arr)
mergeSort(arr)
print(arr)

# arr = [1, 2, 3, 3, 6, 8, 9]
# assert binarySearchRecursive(arr, 9)
# assert binarySearchRecursive(arr, 1)
# assert binarySearchRecursive(arr, 6)
# assert not binarySearchRecursive(arr, 7)
