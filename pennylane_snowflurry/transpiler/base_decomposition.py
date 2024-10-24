from pennylane.tape import QuantumTape
from pennylane.operation import Operation

import pennylane.transforms as transforms

base_gates = ["Adjoint(T)", "Adjoint(S)", "SX", "Adjoint(SX)", 
                  "T", "PauliX", "PauliY", "PauliZ", "S", "Hadamard", 
                  "CZ", "CNOT", "RZ", "RX", "RY"]

def base_decomposition(tape : QuantumTape):
    """decompose every non-standard gates (arbitrary unitaries and 3+ qubits gates) into a base set of gates"""

    def stop_at(op : Operation):
        # TODO : voir quelles portes on veut stop at
        return op.name in base_gates

    # pennylane create_expand_fn does the job for us 
    custom_expand_fn = transforms.create_expand_fn(depth=9, stop_at=stop_at)
    tape = custom_expand_fn(tape)
    return tape