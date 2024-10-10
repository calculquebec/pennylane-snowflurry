from pennylane.tape import QuantumTape
import pennylane.transforms as transforms
def physical_optimization(tape : QuantumTape, iterations = 3):
    # optimiser les portes natives
    for i in range(iterations):
        comparison = tape.operations.copy()
        tape = transforms.cancel_inverses(tape)[0][0]
        tape = transforms.merge_rotations(tape)[0][0]
        tape = transforms.merge_amplitude_embedding(tape)[0][0]
        tape = transforms.commute_controlled(tape, "right")[0][0]
        if comparison == tape.operations:
            break
    return tape