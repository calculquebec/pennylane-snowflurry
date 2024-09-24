import numpy as np
from pennylane.tape import QuantumTape
import pennylane.transforms as transforms
def virtual_optimisation(tape : QuantumTape):
    # optimiser les portes du circuit tel-quel avant de mapper les fils et de décomposer les portes non-natives
    tape = transforms.cancel_inverses(tape)[0][0]
    tape = transforms.cancel_inverses(tape)[0][0]
    # tape = transforms.
    return tape
