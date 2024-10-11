from pennylane.tape import QuantumTape

def to_qasm(tape : QuantumTape) -> str:
    eq = {
        "PauliX" : "x", "PauliY" : "y", "PauliZ" : "z", "Identity" : "id",
        "RX" : "rx", "RY" : "ry", "RZ" : "rz", "PhaseShift" : "p", "Hadamard" : "h",
        "S" : "s", "Adjoint(S)" : "sdg", "SX" : "sx", "Adjoint(SX)" : "sxdg", "T" : "t", "Adjoint(T)" : "tdg", 
        "CNOT" : "cx", "CY" : "cy", "CZ" : "cz", "SWAP" : "swap"
    }
    return "\n".join([eq[op.name] \
        + (f"({op.parameters[0]}) " if len(op.parameters) > 0 else " ") \
        + " ".join([f"q[{w}]" for w in op.wires]) \
        + ";" for op in tape.operations])
