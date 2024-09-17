from copy import deepcopy
from functools import partial
from typing import Callable, Tuple
import numpy as np
from pennylane import transform
import pennylane.transforms as transforms
from pennylane.operation import Operation
from pennylane.tape import QuantumTape
import custom_gates as custom
import pennylane as qml
from pennylane.ops.op_math import SProd

from pennylane_snowflurry.custom_optimization import thunderhead_optimize

epsilon = 0.001

def custom_toffoli(wires):
    return [
        qml.Hadamard([wires[2]]),
        qml.CNOT([wires[1], wires[2]]),
        qml.adjoint(qml.T)([wires[2]]),
        qml.CNOT([wires[0], wires[2]]),
        qml.T([wires[2]]),
        qml.CNOT([wires[1], wires[2]]),
        qml.adjoint(qml.T)([wires[2]]),
        qml.CNOT([wires[0], wires[2]]),
        qml.T([wires[1]]),
        qml.T([wires[2]]),
        qml.CNOT([wires[0], wires[1]]),
        qml.Hadamard([wires[2]]),
        qml.T([wires[0]]),
        qml.adjoint(qml.T)([wires[1]]),
        qml.CNOT([wires[0], wires[1]]),
    ]

def custom_tdag(wires):
    return [custom.TDagger(wires)]

def custom_s(wires):
    return [custom.Z90(wires)]

def custom_sdag(wires):
    return [custom.ZM90(wires)]

def custom_h(wires):
    return [custom.Z90(wires), custom.X90(wires), custom.Z90(wires)]

def custom_cnot(wires):
    return custom_h(wires[1]) + [qml.CZ(wires)] + custom_h(wires[1])

def custom_cy(wires):
    return custom_h(wires[1]) \
        + custom_s(wires[1]) \
        + custom_cnot(wires) \
        + custom_sdag(wires[1]) \
        + custom_h(wires[1])

def custom_rz(angle : float, wires):
    while angle < 0: angle += np.pi * 2
    angle %= np.pi * 2
    is_close_enough_to = lambda other_angle: np.abs(angle - other_angle) < 0.001

    if is_close_enough_to(0): return []
    elif is_close_enough_to(np.pi/4): return [qml.T(wires = wires)]
    elif is_close_enough_to(-np.pi/4): return [custom.TDagger(wires = wires)]
    elif is_close_enough_to(np.pi/2): return [custom.Z90(wires = wires)]
    elif is_close_enough_to(-np.pi/2): return [custom.ZM90(wires = wires)]
    elif is_close_enough_to(np.pi): return [qml.PauliZ(wires = wires)]

    result = []
    # rotate ±pi/2 while necessary
    while(np.abs(angle) > np.pi/2):
        if(angle < 0):
            result += custom_sdag(wires = wires)
            angle += np.pi/2
        else:
            result += custom_s(wires = wires)
            angle -= np.pi/2

    # rotate ±pi/4 if necessary
    while(np.abs(angle) > np.pi/4):
        if(angle < 0):
            result += custom.TDagger(wires=wires)
            angle += np.pi/4
        else:
            result += [qml.T(wires = wires)]
            angle -= np.pi/4

    # add the remaining of the angle
    if(np.abs(angle) > epsilon): result += [qml.PhaseShift(angle, wires = wires)]

    return result

def custom_rx(angle : float, wires):
    while angle < 0: angle += np.pi * 2
    angle %= np.pi * 2
    is_close_enough_to = lambda other_angle: np.abs(angle - other_angle) < 0.001

    if is_close_enough_to(0): return []
    elif is_close_enough_to(np.pi/2): return [custom.X90(wires = wires)]
    elif is_close_enough_to(-np.pi/2): return [custom.XM90(wires = wires)]
    elif is_close_enough_to(np.pi): return [qml.PauliX(wires = wires)]

    result = []
    # rotate ±pi/2 while necessary
    while(np.abs(angle) > np.pi/2):
        if(angle < 0):
            result += custom_sdag(0)
            angle += np.pi/2
        else:
            result += custom_s(0)
            angle -= np.pi/2

    result += custom_h(wires = wires)
    # rotate ±pi/4 if necessary
    while(np.abs(angle) > np.pi/4):
        if(angle < 0):
            result += custom_tdag(0)
            angle += np.pi/4
        else:
            result += [qml.T(wires = wires)]
            angle -= np.pi/4

    # add the remaining of the angle
    if(np.abs(angle) > epsilon): result += [qml.PhaseShift(angle, wires = wires)]

    result += custom_h(wires = wires)

    return result

def custom_ry(angle : float, wires):
    while angle < 0: angle += np.pi * 2
    angle %= np.pi * 2
    is_close_enough_to = lambda other_angle: np.abs(angle - other_angle) < 0.001

    if is_close_enough_to(0): return []
    elif is_close_enough_to(np.pi/2): return [custom.Y90(wires = wires)]
    elif is_close_enough_to(-np.pi/2): return [custom.YM90(wires = wires)]
    elif is_close_enough_to(np.pi): return [qml.PauliY(wires = wires)]

    result = []
    # rotate ±pi/2 while necessary
    while(np.abs(angle) > np.pi/2):
        if(angle < 0):
            result += custom.YM90(wires = wires)
            angle += np.pi/2
        else:
            result += custom.Y90(wires = wires)
            angle -= np.pi/2

    result += custom_h(wires = wires)
    result += custom_s(wires = wires)
    # rotate ±pi/4 if necessary
    while(np.abs(angle) > np.pi/4):
        if(angle < 0):
            result += custom_tdag(0)
            angle += np.pi/4
        else:
            result += [qml.T(wires = wires)]
            angle -= np.pi/4

    # add the remaining of the angle
    if(np.abs(angle) > epsilon): result += [qml.PhaseShift(angle, wires = wires)]

    result += custom_sdag(wires = wires)
    result += custom_h(wires = wires)

    return result


decomp_map = {
    "Adjoint(T)" : custom_tdag,
    "S" : custom_s,
    "Adjoint(S)" : custom_sdag,
    "Hadamard" : custom_h,
    "CNOT" : custom_cnot,
    "CY" : custom_cy,
    "RZ" : custom_rz,
    "RX" : custom_rx,
    "RY" : custom_ry,
    "Toffoli" : custom_toffoli
}

@qml.transform
def convert_to_monarq_gates(tape: QuantumTape, exclude_list=None) -> Tuple[list[QuantumTape], Callable]:
    if exclude_list is None:
        exclude_list = []
    
    new_operations = []

    # We define a custom expansion function here to convert everything except
    # the gates identified in exclude_list to gates with known decompositions.
    def stop_at(op):
        return op.name in decomp_map or op.name in exclude_list

    custom_expand_fn = qml.transforms.create_expand_fn(depth=9, stop_at=stop_at)

    with qml.QueuingManager.stop_recording():
        expanded_tape = custom_expand_fn(tape)

        for op in expanded_tape.operations:
            if op.name not in exclude_list and op.name in decomp_map:
                if op.num_params > 0:
                    new_operations.extend(decomp_map[op.name](*op.data, op.wires))
                else:
                    new_operations.extend(decomp_map[op.name](op.wires))
            else:
                new_operations.append(op)

    new_operations = [n.data[0][0] if isinstance(n, SProd) else n for n in new_operations]
    new_tape = type(tape)(new_operations, tape.measurements, shots=tape.shots)

    return [new_tape], lambda results : results[0]

@transform
def thunderhead_decompose(tape : QuantumTape):
    def stop_at(op : Operation):
        return op.name in ["T", "PauliX", "PauliY", "PauliZ", "S", "Hadamard", "CZ"]

    custom_expand_fn = transforms.create_expand_fn(depth=9, stop_at=stop_at)

    optimized_tape = deepcopy(tape)
    with qml.QueuingManager.stop_recording():
        optimized_tape = custom_expand_fn(optimized_tape, 3)
        optimized_tape = transforms.cancel_inverses(optimized_tape)[0][0]
        optimized_tape = transforms.merge_rotations(optimized_tape)[0][0]
        optimized_tape = partial(convert_to_monarq_gates)(optimized_tape)[0][0]
        optimized_tape = thunderhead_optimize(optimized_tape)[0][0]
    new_tape = type(tape)(optimized_tape.operations, tape.measurements, shots=tape.shots)
    
    return [new_tape], lambda results : results[0]