import custom_gates as custom
from pennylane.tape import QuantumTape
from pennylane.operation import Operation
from pennylane.transforms.optimization.optimization_utils import find_next_gate
import pennylane as qml

native_gate_identities = [
        [["Z90", "Z90"], [qml.PauliZ]],
        [["X90", "X90"], [qml.PauliX]],
        [["Y90", "Y90"], [qml.PauliY]],
        [["PauliZ", "Z90"], [custom.ZM90]],
        [["PauliX", "X90"], [custom.XM90]],
        [["PauliY", "Y90"], [custom.YM90]],
        [["PauliZ", "Z90", "T"], [custom.TDagger]],
        [["Z90", "PauliX", "ZM90"], ["PauliY"]], 
        [["ZM90", "PauliX", "Z90"], ["PauliY"]], 
        [["Z90", "X90", "ZM90"], [custom.Y90]],
        [["Z90", "XM90", "ZM90"], [custom.YM90]],
        [["Z90", "XM90", "ZM90"], [custom.YM90]],
        [["Z90", "X90", "ZM90"], [custom.Y90]],
    ]

def _apply_equivalence(gates, pattern, replacement):
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
    
def _search_and_apply_equivalences(operations : list[Operation], equivalences):
    
    result = operations.copy()
    for assoc in equivalences:
        result = _apply_equivalence(result, assoc[0], assoc[1])
        if len(result) < 1: break
    return result

def apply_equivalences(tape: QuantumTape, equivalences) -> QuantumTape:
    """
    checks for groups of gates and replaces it by shorter equivalents

    this is deprecated and shouldn't be used
    """
    new_operations = []
    list_copy = tape.operations.copy()

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
            appendage = _search_and_apply_equivalences(gates_to_apply, equivalences)
            new_operations += appendage

    return type(tape)(new_operations, tape.measurements, shots=tape.shots)

