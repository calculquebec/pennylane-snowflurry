from pennylane.tape import QuantumTape
import pennylane as qml
import pennylane_snowflurry.custom_gates as custom
import numpy as np


def _custom_tdag(wires):
    """
    a native implementation of the adjoint(T) operation
    """
    return [custom.TDagger(wires)]

def _custom_s(wires):
    """
    a native implementation of the S operation
    """
    return [custom.Z90(wires)]

def _custom_sdag(wires):
    """
    a native implementation of the adjoint(S) operation
    """
    return [custom.ZM90(wires)]

def _custom_h(wires):
    """
    a native implementation of the Hadamard operation
    """
    return [custom.Z90(wires), custom.X90(wires), custom.Z90(wires)]

def _custom_cnot(wires):
    """
    a native implementation of the CNOT operation
    """
    return _custom_h(wires[1]) + [qml.CZ(wires)] + _custom_h(wires[1])

def _custom_cy(wires):
    """
    a native implementation of the CY operation
    """
    return _custom_h(wires[1]) \
        + _custom_s(wires[1]) \
        + _custom_cnot(wires) \
        + _custom_sdag(wires[1]) \
        + _custom_h(wires[1])

def _custom_rz(angle : float, wires, epsilon = 1E-8):
    """
    a native implementation of the RZ operation
    """
    while angle < 0: angle += np.pi * 2
    angle %= np.pi * 2
    is_close_enough_to = lambda other_angle: np.abs(angle - other_angle) < epsilon

    if is_close_enough_to(0): return []
    elif is_close_enough_to(np.pi/4): return [qml.T(wires = wires)]
    elif is_close_enough_to(-np.pi/4): return [custom.TDagger(wires = wires)]
    elif is_close_enough_to(np.pi/2): return [custom.Z90(wires = wires)]
    elif is_close_enough_to(-np.pi/2): return [custom.ZM90(wires = wires)]
    elif is_close_enough_to(np.pi): return [qml.PauliZ(wires = wires)]

    result = []
    # rotate �pi/2 while necessary
    while(np.abs(angle) > np.pi/2):
        if(angle < 0):
            result += _custom_sdag(wires = wires)
            angle += np.pi/2
        else:
            result += _custom_s(wires = wires)
            angle -= np.pi/2

    # rotate �pi/4 if necessary
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

def _custom_rx(angle : float, wires, epsilon = 1E-8):
    """
    a native implementation of the RX operation
    """
    while angle < 0: angle += np.pi * 2
    angle %= np.pi * 2
    is_close_enough_to = lambda other_angle: np.abs(angle - other_angle) < epsilon

    if is_close_enough_to(0): return []
    elif is_close_enough_to(np.pi/2): return [custom.X90(wires = wires)]
    elif is_close_enough_to(-np.pi/2): return [custom.XM90(wires = wires)]
    elif is_close_enough_to(np.pi): return [qml.PauliX(wires = wires)]

    result = []
    # rotate �pi/2 while necessary
    while(np.abs(angle) > np.pi/2):
        if(angle < 0):
            result += _custom_sdag(0)
            angle += np.pi/2
        else:
            result += _custom_s(0)
            angle -= np.pi/2
    
    result += _custom_h(wires = wires)
    # rotate �pi/4 if necessary
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

def _custom_ry(angle : float, wires, epsilon = 1E-8):
    """
    a native implementation of the RY operation
    """
    while angle < 0: angle += np.pi * 2
    angle %= np.pi * 2
    is_close_enough_to = lambda other_angle: np.abs(angle - other_angle) < epsilon

    if is_close_enough_to(0): return []
    elif is_close_enough_to(np.pi/2): return [custom.Y90(wires = wires)]
    elif is_close_enough_to(-np.pi/2): return [custom.YM90(wires = wires)]
    elif is_close_enough_to(np.pi): return [qml.PauliY(wires = wires)]
    else: return [qml.PhaseShift(angle, wires = wires)]

    result = []
    # rotate �pi/2 while necessary
    while(np.abs(angle) > np.pi/2):
        if(angle < 0):
            result += custom.YM90(wires = wires)
            angle += np.pi/2
        else:
            result += custom.Y90(wires = wires)
            angle -= np.pi/2

    result += _custom_h(wires = wires)
    result += _custom_s(wires = wires)
    # rotate �pi/4 if necessary
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
    """
    a native implementation of the SWAP operation
    """
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
    """
    decomposes all non-native gate to an equivalent set of native gates
    """
    from pennylane.ops.op_math import SProd
    # d�composer toutes les portes non-natives en porte natives
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