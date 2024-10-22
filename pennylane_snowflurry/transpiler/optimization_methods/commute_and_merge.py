from pennylane.tape import QuantumTape
from pennylane_snowflurry.utility.optimization_utility import  find_previous_gate, find_next_gate
import pennylane.transforms as transforms
import numpy as np

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

def remove_leaf_zs(tape : QuantumTape, iterations = 3) -> QuantumTape:
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

def remove_trivials(tape : QuantumTape, iteration = 3, epsilon = 1E-8):
    new_operations = []
    for op in tape.operations:
        if len(op.parameters) > 0:
            angle = op.parameters[0]
            while angle > 2 * np.pi - epsilon: angle -= 2 * np.pi
            while angle < 0: angle += 2 * np.pi
            if abs(angle) > epsilon:
                new_operations.append(type(op)(angle, wires=op.wires))
        else:
           new_operations.append(op)
    return type(tape)(new_operations, tape.measurements, tape.shots)

def base_optimisation(tape : QuantumTape) -> QuantumTape:
    """
    expands the circuit to rz, rx and cz gates incrementally, and optimizes at each expansion step
    TODO : add pattern matching
    """
    iterations = 3

    for _ in range(iterations):
        new_tape = tape
        new_tape = remove_root_zs(tape)
        new_tape = remove_leaf_zs(new_tape)
        new_tape = transforms.commute_controlled(new_tape)[0][0]
        new_tape = transforms.cancel_inverses(new_tape)[0][0]
        new_tape = transforms.merge_rotations(new_tape)[0][0]
        new_tape = remove_trivials(new_tape)
        if tape.operations == new_tape.operations:
            tape = new_tape
            break;
        else:
            tape = new_tape
    return tape
