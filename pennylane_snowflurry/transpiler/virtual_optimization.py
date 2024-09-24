from pennylane.tape import QuantumTape


def virtual_optimisation(tape : QuantumTape):
    # optimiser les portes du circuit tel-quel avant de mapper les fils et de décomposer les portes non-natives
    return tape