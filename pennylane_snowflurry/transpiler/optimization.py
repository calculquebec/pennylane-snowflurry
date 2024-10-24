import numpy as np
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane_snowflurry.utility.optimization_utility import expand, is_single_axis_gate
import pennylane.transforms as transforms
from .optimization_methods.commute_and_merge import commute_and_merge

def HCZH_cnot(wires):
    return [
        qml.Hadamard(wires[1]),
        qml.CZ(wires),
        qml.Hadamard(wires[1])
    ]

def ZXZ_Hadamard(wires):
    return [
        qml.S(wires),
        qml.SX(wires),
        qml.S(wires)
    ]

def Y_to_ZXZ(op):
    rot_angles = op.single_qubit_rot_angles()
    return [qml.RZ(np.pi/2, op.wires), qml.RX(rot_angles[1], op.wires), qml.RZ(-np.pi/2, op.wires)]

def get_rid_of_y_rotations(tape : QuantumTape):
    list_copy = tape.operations.copy()
    new_operations = []
    for op in list_copy:
        if not is_single_axis_gate(op, "Y"): 
            new_operations += [op]
        else:
            new_operations += Y_to_ZXZ(op)
    return type(tape)(new_operations, tape.measurements, tape.shots)

def optimize(tape : QuantumTape) -> QuantumTape:
    
    tape = commute_and_merge(tape)

    tape = expand(tape, { "CNOT" : HCZH_cnot })
    tape = commute_and_merge(tape)
    
    tape = expand(tape, { "Hadamard" : ZXZ_Hadamard })
    tape = commute_and_merge(tape)
    
    tape = transforms.create_expand_fn(depth=3, stop_at=lambda op: op.name in ["RZ", "RX", "RY", "CZ"])(tape)
    tape = commute_and_merge(tape)
    
    tape = get_rid_of_y_rotations(tape)
    tape = commute_and_merge(tape)
    return tape
