import pennylane as qml
import numpy as np

def U(wires, angle = 2 * np.pi / 5):
    return qml.PhaseShift(angle, wires=wires)

def circuit_qpe(num_wires = 5, angle = 2 * np.pi / 5):
    wires = [i for i in range(num_wires)]
    estimation_wires = wires[:-1]
    # initialize to state |1>
    qml.PauliX(wires=num_wires - 1)

    for wire in estimation_wires:
        qml.Hadamard(wires=wire)

    qml.ControlledSequence(U(num_wires - 1, angle), control=estimation_wires)
    qml.adjoint(qml.QFT)(wires=estimation_wires)

    return qml.probs(wires=estimation_wires)

def commutation_test():
    qml.RZ(0.1, wires=[0])
    qml.RX(0.2, wires=[0])
    qml.RZ(0.3, wires=[0])
    qml.CZ(wires=[0, 1])
    qml.RZ(0.4, wires=[1])
    qml.RZ(0.5, wires=[0])
    qml.RX(0.6, wires=[1])
    qml.RX(0.7, wires=[0])
    qml.RZ(0.8, wires=[1])
    qml.RZ(0.9, wires=[0])
    qml.CZ(wires=[1, 2])
    qml.RZ(1.0, wires=[2])
    qml.RZ(1.1, wires=[1])
    qml.RX(1.2, wires=[2])
    qml.RX(1.3, wires=[1])
    qml.RZ(1.4, wires=[2])
    qml.RZ(1.5, wires=[1])
    qml.CZ(wires=[0, 2])
    qml.RZ(1.6, wires=[2])
    qml.RZ(1.7, wires=[0])
    qml.RX(1.8, wires=[2])
    qml.RX(1.9, wires=[0])
    qml.RZ(2.0, wires=[2])
    qml.RZ(2.1, wires=[0])
    qml.CZ(wires=[1, 2])
    qml.RZ(2.2, wires=[1])
    qml.RX(2.3, wires=[1])
    qml.RZ(2.4, wires=[1])

    return qml.probs(wires=[0,1,2])

def GHZ(num_wires):
    qml.Hadamard(0)
    [qml.CNOT([0, i]) for i in range(num_wires)]
    return qml.probs()

def bernstein_varizani(number : int):
    value = []
    while number > 0:
        value.insert(0, (number & 1) != 0)
        number = number >> 1

    num_wires = len(value) + 1
    [qml.Hadamard(i) for i in range(num_wires)]
    qml.Z(num_wires-1)
        
    # Uf
    [qml.CNOT([i, num_wires - 1]) for i, should in enumerate(value) if should]
        
    [qml.Hadamard(i) for i in range(num_wires - 1)]
    return qml.probs(wires=[i for i in range(num_wires - 1)])
    