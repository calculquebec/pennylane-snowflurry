from copy import deepcopy
from pennylane.tape import QuantumTape
import pennylane as qml
from pennylane.transforms import transform
import transpiler.virtual_optimization as step1
import transpiler.multi_gate_decomposition as step2
import transpiler.physical_placement as step3
import transpiler.swap_routing as step4
import transpiler.native_gate_decomposition as step5
import transpiler.physical_optimization as step6

@transform
def transpile(tape : QuantumTape):
    optimized_tape = deepcopy(tape)
    with qml.QueuingManager.stop_recording():
        optimized_tape = step2.multiple_gate_decomposition(optimized_tape)
        optimized_tape = step1.virtual_optimisation(optimized_tape)
        optimized_tape = step3.physical_placement(optimized_tape)
        optimized_tape = step4.swap_routing(optimized_tape)
        optimized_tape = step5.native_gate_decomposition(optimized_tape)
        optimized_tape = step6.physical_optimization(optimized_tape)
    new_tape = type(tape)(optimized_tape.operations, optimized_tape.measurements, shots=optimized_tape.shots)
    return [new_tape], lambda results : results[0]