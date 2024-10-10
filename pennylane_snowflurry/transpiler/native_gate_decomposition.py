from pennylane.tape import QuantumTape
from pennylane.operation import Operation
import pennylane as qml
import custom_gates as custom
import numpy as np


epsilon = 0.001

def _custom_sx(wires):
    return [custom.X90(wires)]

def _custom_sxdag(wires):
    return [custom.XM90(wires)]

def _custom_tdag(wires):
    return [custom.TDagger(wires)]

def _custom_s(wires):
    return [custom.Z90(wires)]

def _custom_sdag(wires):
    return [custom.ZM90(wires)]

def _custom_h(wires):
    return [custom.Z90(wires), custom.X90(wires), custom.Z90(wires)]

def _custom_cnot(wires):
    return _custom_h(wires[1]) + [qml.CZ(wires)] + _custom_h(wires[1])

def _custom_cy(wires):
    return _custom_h(wires[1]) \
        + _custom_s(wires[1]) \
        + _custom_cnot(wires) \
        + _custom_sdag(wires[1]) \
        + _custom_h(wires[1])

def _compare(a : float, b : float, epsilon = 1E-8):
    return a - b > epsilon or abs(a-b) < epsilon

def _custom_rz(angle : float, wires, epsilon = 1E-8):
    
    def update_angle(angle : float, result : list[Operation]):
        if _compare(angle, 7 * np.pi/4):
            angle -= 7 * np.pi/4
            result.extend([custom.TDagger(wires)])
        elif _compare(angle, 3 * np.pi/2):
            angle -= 3 * np.pi/2
            result.extend([custom.ZM90(wires)])
        elif _compare(angle, np.pi):
            angle -= np.pi
            result.extend([qml.PauliZ(wires)])
        elif _compare(angle, np.pi/2):
            angle -= np.pi/2
            result.extend([custom.Z90(wires)])
        elif _compare(angle, np.pi/4):
            angle -= np.pi/4
            result.extend([qml.T(wires)])
        return angle

    while angle < 0: angle += np.pi * 2
    while angle > 2*np.pi - epsilon: angle -= np.pi * 2

    result = []
    # rotate ±pi/2 while necessary
    while(angle > np.pi/4 - epsilon):
        angle = update_angle(angle, result)

    # add the remaining of the angle
    if(np.abs(angle) > epsilon): 
        result.extend([qml.PhaseShift(angle, wires = wires)])

    return result

def _custom_rx(angle : float, wires, epsilon = 1E-8):

    def update_angle(angle : float, result : list[Operation]):
        if _compare(angle, 3 * np.pi/2):
            angle -= 3 * np.pi/2
            result.extend([custom.XM90(wires)])
        elif _compare(angle, np.pi):
            angle -= np.pi
            result.extend([qml.PauliX(wires)])
        elif _compare(angle, np.pi/2):
            angle -= np.pi/2
            result.extend([custom.X90(wires)])
        return angle

    while angle < 0: angle += np.pi * 2
    while angle > 2*np.pi - epsilon: angle -= np.pi * 2

    result = []
    # rotate ±pi/2 while necessary
    while(angle > np.pi/2 - epsilon):
        angle = update_angle(angle, result)

    # add the remaining of the angle
    if(np.abs(angle) > epsilon): 
        result.extend(_custom_h(wires = wires))
        result.extend([qml.PhaseShift(angle, wires = wires)])
        result.extend(_custom_h(wires = wires))

    return result

def _custom_ry(angle : float, wires, epsilon = 1E-8):

    def update_angle(angle : float, result : list[Operation]):
        if _compare(angle, 3 * np.pi/2):
            angle -= 3 * np.pi/2
            result.extend([custom.YM90(wires)])
        elif _compare(angle, np.pi):
            angle -= np.pi
            result.extend([qml.PauliY(wires)])
        elif _compare(angle, np.pi/2):
            angle -= np.pi/2
            result.extend([custom.Y90(wires)])
        return angle

    while angle < 0: angle += np.pi * 2
    while angle > 2*np.pi - epsilon: angle -= np.pi * 2

    result = []
    # rotate ±pi/2 while necessary
    while(angle > np.pi/2 - epsilon):
        angle = update_angle(angle, result)

    # add the remaining of the angle
    if(np.abs(angle) > epsilon): 
        result.extend(_custom_h(wires = wires))
        result.extend(_custom_s(wires = wires))
        result.extend([qml.PhaseShift(angle, wires = wires)])
        result.extend(_custom_sdag(wires = wires))
        result.extend(_custom_h(wires = wires))

    return result

def _custom_swap(wires):
    return _custom_cnot(wires) + _custom_h(wires[0]) + _custom_h(wires[1]) + _custom_cnot(wires) + _custom_h(wires[0]) + _custom_h(wires[1]) + _custom_cnot(wires)

def _custom_sx(wires):
    return [custom.X90(wires)]

def _custom_sxdag(wires):
    return [custom.XM90(wires)]

_decomp_map = {
    "Adjoint(T)" : _custom_tdag,
    "S" : _custom_s,
    "Adjoint(S)" : _custom_sdag,
    "Hadamard" : _custom_h,
    "CNOT" : _custom_cnot,
    "CY" : _custom_cy,
    "RZ" : _custom_rz,
    "RX" : _custom_rx,
    "RY" : _custom_ry,
    "SWAP" : _custom_swap,
    "SX" : _custom_sx,
    "Adjoint(SX)" : _custom_sxdag
}

def native_gate_decomposition(tape : QuantumTape, exclude_list : list[str] = None):
    from pennylane.ops.op_math import SProd
    # décomposer toutes les portes non-natives en porte natives
    if exclude_list is None:
        exclude_list = []
    
    new_operations = []

    with qml.QueuingManager.stop_recording():
        for op in tape.operations:
            if op.name not in exclude_list and op.name in _decomp_map:
                if op.num_params > 0:
                    new_operations.extend(_decomp_map[op.name](*op.data, op.wires))
                else:
                    new_operations.extend(_decomp_map[op.name](op.wires))
            else:
                new_operations.append(op)

    new_operations = [n.data[0][0] if isinstance(n, SProd) else n for n in new_operations]
    new_tape = type(tape)(new_operations, tape.measurements, shots=tape.shots)

    return new_tape