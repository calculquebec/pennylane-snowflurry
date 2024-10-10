from pennylane.tape import QuantumTape
from pennylane.operation import Operation

import pennylane.transforms as transforms
def multiple_gate_decomposition(tape : QuantumTape):
    # on veut que toutes les portes � >= 3 qubits soient simplifi� � <= 2 qubits

    def stop_at(op : Operation):
        # TODO : voir quelles portes on veut stop at
        return op.name in ["T", "PauliX", "PauliY", "PauliZ", "S", "Hadamard", "CZ", "CNOT", "RZ", "RX", "RY"]

    # pennylane create_expand_fn does the job for us 
    custom_expand_fn = transforms.create_expand_fn(depth=9, stop_at=stop_at)
    tape = custom_expand_fn(tape)
    return tape