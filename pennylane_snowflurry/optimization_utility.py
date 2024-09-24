from typing import TypeVar, Callable
from pennylane.operation import Operation

T = TypeVar("T")
U = TypeVar("U")

def find_next_group(matcher : Callable[[T], bool], someList : list[T]) -> tuple[int, int]:
    start, end = -1, -1
    for op_idx, op in enumerate(someList):
        
        if start < 0 and end < 0 and matcher(op): 
            start = op_idx
            continue
        elif start >= 0 and end < 0 and not matcher(op):
            end = op_idx - 1
            return start, end

    if start == -1: return start, end

    return start, len(someList) - 1

def find_next(matcher : Callable[[T], bool], someList : list[T]) -> int:
    for op_idx, op in enumerate(someList):
        if matcher(op):
            return op_idx
