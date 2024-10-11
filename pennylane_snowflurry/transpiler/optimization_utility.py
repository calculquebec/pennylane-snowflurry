from copy import deepcopy
from typing import TypeVar, Callable
import custom_gates as custom
from pennylane.operation import Operation
from pennylane.ops import ControlledOp
import pennylane.transforms as transforms
import pennylane as qml
from pennylane import math
from pennylane.tape import QuantumTape
from pennylane.wires import Wires
import numpy as np
import networkx as nx

T = TypeVar("T")
U = TypeVar("U")
            
def expand(tape : QuantumTape, decomps : dict[str, Callable[[Wires], list[Operation]]], iterations = 1) -> QuantumTape:
    list_copy = tape.operations.copy()
    for _ in range(iterations):
        new_operations = []
        for op in list_copy:
            new_operations += decomps[op.name](op.wires) if op.name in decomps else [op]
        if list_copy == new_operations:
            list_copy = new_operations.copy()
            break
        list_copy = new_operations.copy()
    return type(tape)(list_copy, tape.measurements, tape.shots)

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

def is_single_axis_gate(op : Operation, axis : str):
    if op.num_wires != 1: return False
    return op.basis == axis

def is_controlled_axis_gate(op : Operation, axis : str):
    if not isinstance(op, ControlledOp):
        return False
    return op.base.basis == axis

def normalize_angle(angle : float, epsilon = 1E-8) -> float:
        while angle > np.pi * 2 - epsilon:
            angle -= np.pi * 2
        while angle < -epsilon:
            angle += np.pi * 2
        return angle

def to_clifford_t(tape : QuantumTape, epsilon = 1E-8) -> QuantumTape:
    """
    turns X rotations and Z rotations into Z, X, S, SX and T gates
    """
    assocs = [[np.pi, [qml.Z, qml.X]],
              [np.pi/2, [qml.S, qml.SX]],
              [3 * np.pi/2, [qml.adjoint(qml.S), qml.adjoint(qml.SX)]],
              [np.pi/4, [qml.T, None]],
              [7 * np.pi/4, [qml.adjoint(qml.T), None]]]

    
    def identify_axis(op):
        """
        this assumes there are only cz, rx and rz gates
        """

        if is_single_axis_gate(op, "Z"):
            return 0
        elif not is_single_axis_gate(op, "X"):
            return 1
        raise Exception(f"operation {op.name} is not valid in this context. It should be in [RZ, RX, CZ]")
    
    new_operations : list[Operation] = []

    for op in tape.operations:
        # if there are no angles for this operation, add operation directly
        if len(op.parameters) < 1:
            new_operations.append(op)
            continue
        angle = normalize_angle(op.parameters[0])

        # if angle == 0, skip
        if angle < epsilon:
            continue

        choice = list(filter(lambda a: abs(a[0] - angle) < epsilon, assocs))
        # if there is no operation for that rotation, add rotation
        if len(choice) < 1:
            new_operations.append(op)
            continue
        choice = choice[0]
        axis = identify_axis(op)
        choice = choice[1][axis]
        # if there is no operation for that rotation, add rotation
        if choice is None:
            new_operations.append(op)
            continue
        # otherwise, apply new operation
        else:
            new_operations.append(choice(op.wires))
            continue
    return type(tape)(new_operations, tape.measurements, tape.shots)
