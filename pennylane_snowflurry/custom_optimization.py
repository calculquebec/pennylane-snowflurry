from copy import copy
from functools import partial
import pennylane as qml
import custom_gates as custom
from pennylane.operation import Operation
from pennylane.tape import QuantumTape
from typing import Tuple, Callable
from pennylane.wires import Wires
import numpy as np

def find_next_gate(wires, op_list):
    for op_idx, op in enumerate(op_list):
        if len(Wires.shared_wires([wires, op.wires])) > 0:
            return op_idx

single_qubit_gate_identities = [
    [["Z90", "X90", "Z90", "Z90", "X90", "Z90"], []],
    [["Z90", "Z90"], [qml.PauliZ]],
    [["X90", "X90"], [qml.PauliX]],
    [["Y90", "Y90"], [qml.PauliY]],
    [["PauliZ", "Z90"], [custom.ZM90]],
    [["PauliX", "X90"], [custom.XM90]],
    [["PauliY", "Y90"], [custom.YM90]],
    [["PauliZ", "PauliZ"], []],
    [["PauliX", "PauliX"], []],
    [["PauliY", "PauliY"], []],
    [["PauliZ"], [partial(qml.PhaseShift, np.pi)]],
    [["Z90"], [partial(qml.PhaseShift, np.pi/2)]],
    [["ZM90"], [partial(qml.PhaseShift, -np.pi/2)]],
    [["T"], [partial(qml.PhaseShift, np.pi/4)]],
    [["TDagger"], [partial(qml.PhaseShift, -np.pi/4)]]
]

phaseshift_association = [
    [np.pi, qml.PauliZ],
    [np.pi/2, custom.Z90],
    [-np.pi/2, custom.ZM90],
    [np.pi/4, qml.T],
    [-np.pi/4, custom.TDagger],
    [0, None]
]

def revert_to_Z(operations : list[Operation]):
    new_operations = []
    for current in operations:
        if(current.name != "PhaseShift"):
            new_operations += [current]
            continue

        angle = current.parameters[0]
        while angle < 0: angle += np.pi * 2
        angle %= np.pi * 2
        new_op = [p for p in phaseshift_association if np.abs(p[0] - current.parameters[0]) < 0.001]
        
        if len(new_op) < 1: 
            new_operations += [current]
            continue
        new_op = new_op[0]
        if new_op is None: continue

        new_operations += [new_op[1](current.wires)]
    return new_operations

def merge_phaseshifts(operations : list[Operation]):
    new_operations = []
    list_copy = operations.copy()

    while len(list_copy) > 0:
        current = list_copy.pop(0)
        if(current.name != "PhaseShift"):
            new_operations += [current]
            continue

        combined_angle = current.parameters[0]
        while len(list_copy) >= 1 and list_copy[0].name == "PhaseShift":
            other = list_copy.pop(0)
            combined_angle += other.parameters[0]
            
        combined_angle %= (2 * np.pi)
        new_operations += [qml.PhaseShift(combined_angle, wires=current.wires)]
    return new_operations


def search_and_apply_identities(operations):
    def apply_identity(gates, pattern, replacement):
        wires = gates[0].wires
        from kmp import kmp_search
        new_gates = []
        list_copy = copy(gates)
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
    
    result = copy(operations)
    for assoc in single_qubit_gate_identities:
        result = apply_identity(result, assoc[0], assoc[1])
        if len(result) < 1: break
    return result

@qml.transform
def thunderhead_optimize(tape: QuantumTape) -> Tuple[list[QuantumTape], Callable]:
    new_operations = []
    list_copy = tape.operations.copy()

    with qml.QueuingManager.stop_recording():
        with qml.tape.QuantumTape() as _:
            while len(list_copy) > 0:
                current_gate = list_copy.pop(0)

                # Ignore 2-qubit gates
                if len(current_gate.wires) > 1:
                    new_operations.append(current_gate)
                    continue

                next_gate_idx = find_next_gate(current_gate.wires, list_copy)

                if next_gate_idx is None:
                    new_operations.append(current_gate)
                    continue

                gates_to_apply = [current_gate]

                while next_gate_idx is not None:
                    next_gate = list_copy[next_gate_idx]

                    
                    if len(next_gate.wires) > 1:
                        break
                    
                    gates_to_apply.append(next_gate)
                    list_copy.pop(next_gate_idx)
                    next_gate_idx = find_next_gate(current_gate.wires, list_copy)

                if len(gates_to_apply) == 1:
                    new_operations += gates_to_apply
                else:
                    appendage = search_and_apply_identities(gates_to_apply)
                    appendage = merge_phaseshifts(appendage)
                    appendage = revert_to_Z(appendage)
                    new_operations += appendage

    new_tape = type(tape)(new_operations, tape.measurements, shots=tape.shots)

    return [new_tape], lambda results: results[0]
