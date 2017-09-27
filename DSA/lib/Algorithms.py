import numpy as np
# Search Algorithm


def binary_search(arr, value):
    """
     Function: Return index in array of element that matches value.
               return None if no elements in array match

     Note: Can only be performed on a sorted list
           BigO --> O(logn)

     Parameters
     ----------------------
     arr:   list of sorted values
     value: target value

     Output
     ----------------------
     array_out = sorted list
    """
    # Handle Corner Case 1
    if len(arr) < 0:
        return None
    # Handle Corner Case 2
    if len(arr) == 0:
        if arr[0] == value:
            return 0
        else:
            return None

    # Binary Search algorithm
    firstidx = 0
    lastidx = len(arr) - 1
    mididx = firstidx + (lastidx - firstidx) // 2

    while True:
        # stop criteria: when mididx isn't updated
        if mididx > lastidx:
            return None
        # If condition is met, return mididx
        dcmpr = arr[mididx] - value
        if dcmpr == 0:
            return mididx
        # If we over-shot our estimate...
        if dcmpr > 0:
            lastidx = mididx
            mididx = firstidx + (lastidx - firstidx) // 2
        if dcmpr < 0:
            firstidx = mididx
            mididx = (firstidx + 1) + (lastidx - firstidx) // 2
# reference: https://en.wikipedia.org/wiki/Sorting_algorithm
# Sort Algorithms

# reference: https://en.wikipedia.org/wiki/Sorting_algorithm
# Sort Algorithms


def selection_sort(array_in):
    """
    BigO -> O(N^2) ~  (N-1)(N-1) ALWAYS

    Parameters
     ----------------------
     arr:   list of unsorted values

     Output
     ----------------------
     array_out = sorted list
     idx_out   = indices of sorted list
    """
    # Initialize variables
    N = len(array_in)             # Length of array
    idx_array = np.arange(0, N)    # index array

    # Begin sorting algorithm
    for idx1 in range(0, N - 1):
        cur_idx = idx_array[idx1]  # index array
        cur_val = array_in[idx1]
        for idx2 in range(idx1 + 1, N):
            cmp_idx = idx_array[idx2]  # index array
            cmp_val = array_in[idx2]
    # swap values if out of order
            if cur_val > cmp_val:
                # sorted array
                array_in[idx2] = cur_val
                array_in[idx1] = cmp_val
                cur_val = cmp_val
                # index array
                idx_array[idx2] = cur_idx
                idx_array[idx1] = cmp_idx
                cur_idx = cmp_idx
    # place in current position in output array
    return array_in, idx_array


def bubble_sort(array_in):
    """
     BigO -> O(N^2)
      * Fastest when sorted.  BEST ~ O(N)

    Parameters
     ----------------------
     arr:   list of unsorted values

     Output
     ----------------------
     array_out = sorted list
     idx_out   = indices of sorted list
    """
    # Initialize variables
    N = len(array_in)             # Length of array
    idx_array = np.arange(0, N)    # index array
    swapped = True                # stopping criteria
    # Begin sorting algorithm
    while swapped:
        swapped = False
        # Create bubble to compare between idx1 and idx2 values
        for idx1 in range(0, N - 1):
            cur_val = array_in[idx1]
            next_val = array_in[idx1 + 1]
            # index array
            cur_idx = idx_array[idx1]
            next_idx = idx_array[idx1 + 1]
            # swap values if out of order
            if cur_val > next_val:
                array_in = _swap(array_in, idx1 + 1, idx1)
                idx_array = _swap(idx_array, idx1 + 1, idx1)  # index array
                swapped = True
    return array_in, idx_array


def insertion_sort(array_in):
    """
     BigO -> O(N^2)
      * Fastest when sorted.  BEST ~ O(N)
    Parameters
     ----------------------
     arr:   list of unsorted values

     Output
     ----------------------
     array_out = sorted list
     idx_out   = indices of sorted list
    """
    # Initialize variables
    N = len(array_in)             # Length of array
    idx_array = np.zeros(N)      # index array

    array_out = np.zeros(N)       # empty output array, with first element
    array_out[0] = array_in[0]
    idx_array[0] = 1
    # Begin sorting algorithm
    for curidx in range(1, N):
        tempidx = curidx
        array_out[tempidx] = array_in[curidx]
        # decrement through updated sorted list to make sure new value is inserted in correct place
        while (tempidx > 0) and (array_out[tempidx - 1] > array_out[tempidx]):
            # swap values if not in proper order
            array_out = _swap(array_out, tempidx - 1, tempidx)
            # index array swap
            idx_array = _swap(idx_array, tempidx - 1, tempidx)
            # evaluate next position
            tempidx -= 1
    return array_out, idx_array


def merge_sort(array_in):
    """
     BigO -> O(NlogN)
     Descrition: 1.  [a][b][c][d][e][f][g][h]          -> Recursively Reduce down to individual elements
                      Left   Right /  Left   Right
                 2.  [a-b]<->[c-d]<->[e-f]<->[g-h]     -> Recursively Build up into sorted Left/Right pairs
                        Left       Right
                 3.  [a-b-c-d]<->[e-f-g-h]             -> At first funciton call, build sorted array from
                                                        sorted Left/Right pairs

    Parameters
     ----------------------
     arr:   list of unsorted values

     Output
     ----------------------
     array_out = sorted list
    """
    # split array into 2 halves until only 1 element
    if len(array_in) > 1:
        m = len(array_in) // 2  # half of array length
        larray = array_in[:m]
        rarray = array_in[m:]
        # recursively breaks array into individual elements
        merge_sort(larray)
        merge_sort(rarray)
        # Begin sorting once condition is met (array fully broken down);
        ridx = 0
        lidx = 0
        oidx = 0
        # Build up array into sorted fashion
        # Scenario 1: comparing elements in left/right arrays
        while(len(larray) > lidx and len(rarray) > ridx):
            if larray[lidx] < rarray[ridx]:
                array_in[oidx] = larray[lidx]
                lidx += 1
            else:
                array_in[oidx] = rarray[ridx]
                ridx += 1
            oidx += 1
        # Scenario 2: only elements in left array
        while(len(larray) > lidx):
            array_in[oidx] = larray[lidx]
            lidx += 1
            oidx += 1
        # Scenario 3: only elements in right array
        while(len(rarray) > ridx):
            array_in[oidx] = rarray[ridx]
            ridx += 1
            oidx += 1
    return array_in


def quick_sort(array_in):
    """
     BigO -> O(NlogN)
     Descrition: 1.  a-b-c-e-f-g-h-[d]    -> Choose a pivot point
                *2.  a-b-c-[d]-e-f-g-h    -> Insert pivot into position s.t.
                                             pvalue > right values & pvalue < left values
                   Ll[ LEFT ]Lr     Rl[ RIGHT ]Rr            (updates Left/Right bounds)
                 3.  [a-b-c]<-->{d}<->[e-f-g-h]        ->At first funciton call, build sorted array from
                                                        sorted Left/Right pairs
     NOTE
     ----------------------
     This sorting method loses efficiency with repeated elements and short arrays
     Optimization options: switch to a different sorting method for small array sizes

     Parameters
     ----------------------
     arr:   list of unsorted values

     Output
     ----------------------
     array_out = sorted list
     idx_out   = indices of sorted list
    """
    idx_array = np.arange(len(array_in))
    _quick_sort_exe(array_in, idx_array, 0, len(array_in) - 1)
    return array_in, idx_array


def _quick_sort_exe(array_in, idx_array, first, last):

    if first < last:  # stopping criteria within recursion

        new_pivot_idx = _rePivot(array_in, idx_array, first, last)     # updates Left/Right bounds

        _quick_sort_exe(array_in, idx_array, first, new_pivot_idx)  # Sort left-side
        _quick_sort_exe(array_in, idx_array, new_pivot_idx + 1, last)  # Sort right-side


def _rePivot(array_in, idx_array, first, last):
    # Method for choosing Quick Sort pivot idx (choose median of 3 pts)
    pivot_idx = first  # _select_QSpivot(array_in, first, last)
    left = first - 1
    right = last + 1
    while True:
        # This prevents endless loop for repeated elements
        left += 1
        right -= 1
        # Increment from start idx & decrement from last idx s.t. condition *2 is satistfied
        while (array_in[pivot_idx] > array_in[left]):
            left += 1
        while (array_in[pivot_idx] < array_in[right]):
            right -= 1

        # Stopping condition
        if left >= right:
            return right

        # Swap right with left if stopping condition is not met
        array_in = _swap(array_in, left, right)
        # index array
        idx_array = _swap(idx_array, left, right)


def _select_QSpivot(array_in, first, last):
    # WARNING: does not work when implemented
    # 1. Method for choosing Quick Sort pivot idx (choose median of 3 pts)
    pivot_idx = ((last - first) // 2) + first  # index between first and last
    # store first,last and middle indice values of  array
    val_first = array_in[first]
    val_piv = array_in[pivot_idx]
    val_last = array_in[last]
    # keep median
    if val_first > val_piv and val_first < val_last:
        pivot_idx = first

    if val_last > val_piv and val_last < val_first:
        pivot_idx = last

    return pivot_idx

# Generic swap function


def _swap(arr_in, idx1, idx2):
    temp = arr_in[idx1]
    arr_in[idx1] = arr_in[idx2]
    arr_in[idx2] = temp
    return arr_in
