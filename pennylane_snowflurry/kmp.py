from typing import Callable, TypeVar
from pennylane_snowflurry.utility.optimization_utility import T, U

def _compute_lps_array(pattern : list[T], compare : Callable[[T, T], bool]) ->list[int]:
    """
    Compute the longest prefix suffix (LPS) array used in KMP algorithm.
    :param pattern: The pattern for which to compute the LPS array.
    :return: The LPS array.
    """
    m = len(pattern)
    lps = [0] * m
    length = 0  # length of the previous longest prefix suffix
    i = 1
    
    while i < m:
        if compare(pattern[i], pattern[length]):
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
                
    return lps

def kmp_search(array : list[T], pattern : list[T], compare : Callable[[T, T], bool]) -> list[int]:
    """
    Perform KMP search of `pattern` in `array`.
    :param array: The array to search within.
    :param pattern: The pattern to search for.
    :return: the first starting index where the pattern is found in the array.
    """
    n = len(array)
    m = len(pattern)
    lps = _compute_lps_array(pattern, compare)
    
    i = 0  # index for array
    j = 0  # index for pattern
    
    while i < n:
        if compare(pattern[j], array[i]):
            i += 1
            j += 1
        
        if j == m:
            return i - j
            j = lps[j - 1]
        elif i < n and not compare(pattern[j], array[i]):
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
                
    return None
