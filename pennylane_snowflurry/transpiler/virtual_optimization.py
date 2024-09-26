import numpy as np
from pennylane.operation import Operation
from pennylane.tape import QuantumTape
import pennylane.transforms as transforms

epsilon = 0.00001

def virtual_optimisation(tape : QuantumTape):
    # optimiser les portes du circuit tel-quel avant de mapper les fils et de décomposer les portes non-natives
    # tape = transforms.cancel_inverses(tape)[0][0]
    # tape = transforms.cancel_inverses(tape)[0][0]
    # tape = transforms.
    return tape

def _commutations(operations : list[Operation]):
    new_operations = []
    list_copy = operations.copy()

    while len(list_copy) > 0:
        op0 = list_copy.pop(0)

        # if it's the first operation we can skip it
        if len(operations) - 1 == len(list_copy) and _is_single_z_gate(op0):
            continue


def _is_single_z_gate(op : Operation):
    mat = op.matrix()
    return op.num_wires == 1 and [0][1] < epsilon and mat[1][0] < epsilon

def _is_single_x_gate(op : Operation):
    mat = op.matrix()
    return mat[0][0] < epsilon and mat[1][1] < epsilon and np.abs(_angle(mat[0][1]) - _angle(mat[1][0])) < epsilon

def _angle(complex_num):
    return np.arctan2(np.imag(complex_num), np.real(complex_num))