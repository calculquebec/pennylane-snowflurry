from copy import deepcopy
from typing import TypeVar, Callable
import custom_gates as custom
from pennylane.operation import Operation
import pennylane.transforms as transforms
import pennylane as qml
from pennylane import math
from pennylane.tape import QuantumTape
from pennylane.wires import Wires
import numpy as np
import networkx as nx

T = TypeVar("T")
U = TypeVar("U")

def find_next_group(matcher : Callable[[T], bool], someList : list[T]) -> tuple[int, int] | None:
    start, end = -1, -1
    for op_idx, op in enumerate(someList):
        
        if start < 0 and end < 0 and matcher(op): 
            start = op_idx
            continue
        elif start >= 0 and end < 0 and not matcher(op):
            end = op_idx - 1
            return start, end

    if start == -1: return start, end

    if start < 0: return None

    return start, len(someList) - 1

def find_next(matcher : Callable[[T], bool], someList : list[T]) -> int:
    for op_idx, op in enumerate(someList):
        if matcher(op):
            return op_idx

def are_matrices_equivalent(unitary1, unitary2):
    r"""Checks the equivalence of two unitary matrices up to a global phase.

    Args:
        unitary1 (tensor): First unitary matrix.
        unitary2 (tensor): Second unitary matrix.

    Returns:
        bool: True if the two matrices are equivalent up to a global phase,
        False otherwise.

    **Example**

    .. code::

        >>> matrix_T = np.diag([1, 1j])
        >>> matrix_RZ = np.diag([np.exp(-1j * np.pi / 4), np.exp(1j * np.pi / 4)])
        >>> are_mats_equivalent(matrix_T, matrix_RZ)
        True
    """
    mat_product = math.dot(unitary1, math.conj(math.T(unitary2)))

    # If the top-left entry is not 0, divide everything by it and test against identity
    if not math.isclose(mat_product[0, 0], 0.0):
        mat_product = mat_product / mat_product[0, 0]

        if math.allclose(mat_product, math.eye(mat_product.shape[0])):
            return True

    return False

def rz_to_phase_shift(op : Operation, epsilon = 1E-8) -> Operation:
    import numpy as np
    mat = op.matrix()
   
    angle = np.arctan2(np.imag(mat[0][0]), np.real(mat[0][0]))
    imag = np.multiply([-1j], [angle], casting='unsafe')
    mat = np.multiply(mat, np.exp(imag[0]), casting='unsafe')

    angle = np.arctan2(np.imag(mat[1][1]), np.real(mat[1][1]))
    return qml.PhaseShift(angle, op.wires)

def find_previous_gate(index : int, wires : list[int], op_list : list[Operation]) -> int:
    """
    find first operation that shares a list of wires prior to an index in a list
    """
    for i in reversed(range(0, index)):
        if any(w in op_list[i].wires for w in wires):
            return i
    return None

def find_next_gate(index : int, wires : list[int], op_list : list[Operation]) -> int:
    """
    find first operation that shares a list of wires after an index in a list
    """
    for i in range(index+1, len(op_list)):
        if any(w in op_list[i].wires for w in wires):
            return i
    return None

def search_and_apply_equivalences(operations, equivalences):
    
    def apply_identity(gates, pattern, replacement):
        wires = gates[0].wires
        from kmp import kmp_search
        new_gates = []
        list_copy = gates.copy()
        while len(list_copy) > 0:
            op_names = [op.name for op in list_copy]
            index = kmp_search(op_names, pattern, lambda a, b: a == b)
            if index is None:
                new_gates += list_copy
                break
            new_gates += [list_copy.pop(0) for _ in range(index)]
            new_gates += [r(wires) for r in replacement]
            [list_copy.pop(0) for _ in pattern]
        return new_gates
    
    result = operations.copy()
    for assoc in equivalences:
        result = apply_identity(result, assoc[0], assoc[1])
        if len(result) < 1: break
    return result
