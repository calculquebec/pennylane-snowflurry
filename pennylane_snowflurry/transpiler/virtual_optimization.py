from typing import Callable, Tuple
import numpy as np
import pennylane as qml
from pennylane.wires import Wires
from pennylane.operation import Operation
from pennylane.tape import QuantumTape
from zx_dag import ZX_DAG
from pennylane_snowflurry.optimization_utility import find_previous_gate, find_next_gate, search_and_apply_equivalences
from pennylane_snowflurry.debug_utility import to_qasm
import pennylane.transforms as transforms
import custom_gates as custom
epsilon = 0.00001

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

def _is_y_gate(op : Operation, epsilon = 1E-8):
    if op.num_wires != 1: return False
    rot_angles = op.single_qubit_rot_angles()
    return abs(rot_angles[0]) < epsilon and abs(rot_angles[2]) < epsilon

def get_rid_of_y_rotations(tape : QuantumTape):
    list_copy = tape.operations.copy()
    new_operations = []
    for op in list_copy:
        if not _is_y_gate(op): 
            new_operations += [op]
        else:
            new_operations += Y_to_ZXZ(op)
    return type(tape)(new_operations, tape.measurements, tape.shots)
            
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

def to_phaseshift(tape : QuantumTape):
    from optimization_utility import rz_to_phase_shift
    list_copy = tape.operations.copy()
    new_operations = []

    for op in list_copy:
        if _is_single_z_gate(op):
            new_operations.append(rz_to_phase_shift(op))
        else:
            new_operations.append(op)

    return type(tape)(new_operations, tape.measurements, tape.shots)

def merge_phaseshifts(tape : QuantumTape):
    from pennylane.transforms.optimization.optimization_utils import find_next_gate
    list_copy = tape.operations.copy()
    new_operations = []
    
    op = list_copy.pop(0)
    while len(list_copy) > 0:
        if op.name != "PhaseShift":
            new_operations.append(op)
            op = list_copy.pop(0)
            continue

        next_op_idx = find_next_gate(op.wires, list_copy)

        if next_op_idx is None:
            new_operations.append(op)
            op = list_copy.pop(0)
            continue

        next_op = list_copy.pop(next_op_idx)
        if next_op.name != "PhaseShift":
            new_operations += [op, next_op]
            op = list_copy.pop(0)
            continue

        op = qml.PhaseShift(op.parameters[0] + next_op.parameters[0], op.wires)

    new_operations.append(op)
    return type(tape)(new_operations, tape.measurements, tape.shots)

def revert_to_Z(tape : QuantumTape):
    phaseshift_association = [
        [np.pi, qml.PauliZ],
        [np.pi/2, qml.S],
        [3*np.pi/2, qml.adjoint(qml.S)],
        [np.pi/4, qml.T],
        [7*np.pi/4, qml.adjoint(qml.T)],
        [0, None],
        [np.pi*2, None]
    ]
    list_copy = tape.operations.copy()
    new_operations = []

    for current in list_copy:
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
        if new_op is None or new_op[1] is None: 
            continue

        new_operations += [new_op[1](current.wires)]
    return type(tape)(new_operations, tape.measurements, tape.shots)

def _is_single_z_gate(op : Operation, epsilon = 1E-8):
    if op.num_wires != 1: return False
    mat = op.matrix()
    return abs(mat[0][1]) < epsilon  and abs(mat[1][0]) < epsilon 

def _is_z_gate(op : Operation):
    mat = op.matrix()
    for i in range(2 ** op.num_wires):
        for j in range(2 ** op.num_wires):
            if i == j:
                continue
            if abs(mat[i][j]) > epsilon:
                return False
    return True

def _to_clifford_t(tape : QuantumTape, epsilon = 1E-8) -> QuantumTape:
    """
    turns X rotations and Z rotations into Z, X, S, SX and T gates
    """
    assocs = [[np.pi, [qml.Z, qml.X]],
              [np.pi/2, [qml.S, qml.SX]],
              [3 * np.pi/2, [qml.adjoint(qml.S), qml.adjoint(qml.SX)]],
              [np.pi/4, [qml.T, None]],
              [7 * np.pi/4, [qml.adjoint(qml.T), None]]]

    def normalize_angle(angle : float) -> float:
        while angle > np.pi * 2 - epsilon:
            angle -= np.pi * 2
        while angle < -epsilon:
            angle += np.pi * 2
        return angle

    def identify_axis(op):
        """
        this assumes there are only cz, rx and rz gates
        """

        if _is_single_z_gate(op):
            return 0
        elif not _is_z_gate(op):
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

def remove_root_zs(tape : QuantumTape, iterations = 3) -> QuantumTape:
    """
    removes all heading z operations
    TODO : add unit tests
    """
    new_operations = tape.operations.copy()
    for i in range(iterations):
        list_copy = new_operations.copy()
        new_operations = []

        for i, op in enumerate(list_copy):
            if op.num_wires != 1 or op.basis != "Z" or find_previous_gate(i, op.wires, list_copy) is not None:
                new_operations.append(op)

        if new_operations == list_copy:
            break
    return type(tape)(new_operations, tape.measurements, tape.shots)

def remove_leaf_zs(tape : QuantumTape, iterations = 5) -> QuantumTape:
    """
    removes all tailing z operations
    TODO : add unit tests
    """
    new_operations = tape.operations.copy()

    
    for i in range(iterations):
        list_copy = new_operations.copy()
        new_operations = []
        for i in reversed(range(len(list_copy))):
            op = list_copy[i]
            if op.num_wires != 1 or op.basis != "Z" or find_next_gate(i, op.wires, list_copy) is not None:
                new_operations.insert(0, op)
                continue

        if new_operations == list_copy:
            break
    return type(tape)(new_operations, tape.measurements, tape.shots)
    
def base_optimisation(tape : QuantumTape) -> QuantumTape:
    """
    expands the circuit to rz, rx and cz gates incrementally, and optimizes at each expansion step
    TODO : add pattern matching
    """
    iterations = 3

    tape = expand(tape, { "CNOT" : HCZH_cnot })
    for _ in range(iterations):
        new_tape = remove_root_zs(tape)
        new_tape = remove_leaf_zs(new_tape)
        new_tape = transforms.commute_controlled(new_tape)[0][0]
        new_tape = transforms.cancel_inverses(new_tape)[0][0]
        new_tape = transforms.merge_rotations(new_tape)[0][0]
        if tape.operations == new_tape.operations:
            tape = new_tape
            break;
        else:
            tape = new_tape
    
    
    tape = expand(tape, { "Hadamard" : ZXZ_Hadamard })
    for _ in range(iterations):
        new_tape = remove_root_zs(tape)
        new_tape = remove_leaf_zs(new_tape)
        new_tape = transforms.commute_controlled(new_tape)[0][0]
        new_tape = transforms.cancel_inverses(new_tape)[0][0]
        new_tape = transforms.merge_rotations(new_tape)[0][0]
        if tape.operations == new_tape.operations:
            tape = new_tape
            break;
        else:
            tape = new_tape
    
    tape = transforms.create_expand_fn(depth=3, stop_at=lambda op: op.name in ["RZ", "RX", "RY", "CZ"])(tape)
    tape = get_rid_of_y_rotations(tape)
    for _ in range(iterations):
        new_tape = remove_root_zs(tape)
        new_tape = remove_leaf_zs(new_tape)
        new_tape = transforms.commute_controlled(new_tape)[0][0]
        new_tape = transforms.cancel_inverses(new_tape)[0][0]
        new_tape = transforms.merge_rotations(new_tape)[0][0]
        if tape.operations == new_tape.operations:
            tape = new_tape
            break;
        else:
            tape = new_tape
    
    return tape

def zx_dag_optimisation(tape : QuantumTape):
    """
    uses a dag containing only RZ, RX and CZ gates to optimize by commuting and merging rotations. 
    deprecated : this should not be used
    """
    from optimization_utility import to_qasm
    import pyperclip

    dag = ZX_DAG(tape)
    
    dag = dag.simplify(20)
    
    print(to_qasm(dag.to_tape(), False))

    return tape

def zx_calculus(tape : QuantumTape) -> QuantumTape:
    import pyzx
    from pennylane.transforms import to_zx, from_zx
    
    g = to_zx(tape)
    pyzx.simplify.teleport_reduce(g)
    test = from_zx(g)
    return test

def apply_equivalences(tape: QuantumTape) -> Tuple[list[QuantumTape], Callable]:
    single_qubit_gate_identities = [
        [["Z90", "Z90"], [qml.PauliZ]],
        [["X90", "X90"], [qml.PauliX]],
        [["Y90", "Y90"], [qml.PauliY]],
        [["PauliZ", "Z90"], [custom.ZM90]],
        [["PauliX", "X90"], [custom.XM90]],
        [["PauliY", "Y90"], [custom.YM90]],
        [["PauliZ", "Z90", "T"], [custom.TDagger]],
        [["X", "Z"], ["Y"]]
    ]

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
                    appendage = search_and_apply_equivalences(gates_to_apply, single_qubit_gate_identities)
                    appendage = merge_phaseshifts(appendage)
                    appendage = revert_to_Z(appendage)
                    new_operations += appendage

    new_tape = type(tape)(new_operations, tape.measurements, shots=tape.shots)

    return [new_tape], lambda results: results[0]

def optimize(tape : QuantumTape) -> QuantumTape:
    
    print(to_qasm(tape, False))
    print("barrier q;")
    tape = base_optimisation(tape)
    print(to_qasm(tape, False))
    return tape