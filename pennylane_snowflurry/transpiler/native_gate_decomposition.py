from pennylane.tape import QuantumTape
import pennylane as qml
import custom_gates as custom
import numpy as np


epsilon = 0.001

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

def _custom_rz(angle : float, wires):
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
            result += _custom_sdag(wires = wires)
            angle += np.pi/2
        else:
            result += _custom_s(wires = wires)
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

def _custom_rx(angle : float, wires):
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
            result += _custom_sdag(0)
            angle += np.pi/2
        else:
            result += _custom_s(0)
            angle -= np.pi/2

    result += _custom_h(wires = wires)
    # rotate ±pi/4 if necessary
    while(np.abs(angle) > np.pi/4):
        if(angle < 0):
            result += _custom_tdag(0)
            angle += np.pi/4
        else:
            result += [qml.T(wires = wires)]
            angle -= np.pi/4

    # add the remaining of the angle
    if(np.abs(angle) > epsilon): result += [qml.PhaseShift(angle, wires = wires)]

    result += _custom_h(wires = wires)

    return result

def _custom_ry(angle : float, wires):
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

    result += _custom_h(wires = wires)
    result += _custom_s(wires = wires)
    # rotate ±pi/4 if necessary
    while(np.abs(angle) > np.pi/4):
        if(angle < 0):
            result += _custom_tdag(0)
            angle += np.pi/4
        else:
            result += [qml.T(wires = wires)]
            angle -= np.pi/4

    # add the remaining of the angle
    if(np.abs(angle) > epsilon): result += [qml.PhaseShift(angle, wires = wires)]

    result += _custom_sdag(wires = wires)
    result += _custom_h(wires = wires)

    return result

def _custom_swap(wires):
    return _custom_cnot(wires) + _custom_h(wires[0]) + _custom_h(wires[1]) + _custom_cnot(wires) + _custom_h(wires[0]) + _custom_h(wires[1]) + _custom_cnot(wires)


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
    "SWAP" : _custom_swap
}

def native_gate_decomposition(tape : QuantumTape, exclude_list : list[str] = None):
    from pennylane.ops.op_math import SProd
    # décomposer toutes les portes non-natives en porte natives
    if exclude_list is None:
        exclude_list = []
    
    new_operations = []

    # We define a custom expansion function here to convert everything except
    # the gates identified in exclude_list to gates with known decompositions.
    def stop_at(op):
        return op.name in _decomp_map or op.name in exclude_list

    custom_expand_fn = qml.transforms.create_expand_fn(depth=9, stop_at=stop_at)

    with qml.QueuingManager.stop_recording():
        expanded_tape = custom_expand_fn(tape)

        for op in expanded_tape.operations:
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