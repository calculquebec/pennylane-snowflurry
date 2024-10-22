from copy import deepcopy
from pennylane.tape import QuantumTape
import pennylane as qml
from pennylane.transforms import transform
import pennylane_snowflurry.transpiler.simple_decomposition as step1
import pennylane_snowflurry.transpiler.placement as step2
import pennylane_snowflurry.transpiler.routing as step3
import pennylane_snowflurry.transpiler.optimization as step4
import pennylane_snowflurry.transpiler.native_decomposition as step5


def get_transpiler(baseDecomposition = True, 
                   placeAndRoute = True, 
                   optimization = True, 
                   nativeDecomposition = True, 
                   benchmark : dict[str, any] = None):
    def transpile(tape : QuantumTape):
        """
        goes through different passes of transpilation until the code is usable on monarq. 
        every phase is optional, leaving modularity to the end user
        TODO : boolean flags could be replaced by enums. this way we could add other ways of optimizing or routing
        """
        optimized_tape = deepcopy(tape)
        with qml.QueuingManager.stop_recording():
            if baseDecomposition: 
                optimized_tape = step1.simple_decomposition(optimized_tape)
            if placeAndRoute: 
                optimized_tape = step2.placement_vf2(optimized_tape, benchmark)
            if placeAndRoute: 
                optimized_tape = step3.swap_routing(optimized_tape, benchmark)
            
            if optimization: 
                optimized_tape = step1.simple_decomposition(optimized_tape)
                optimized_tape = step4.optimize(optimized_tape)
            if nativeDecomposition: 
                optimized_tape = step5.native_gate_decomposition(optimized_tape)
        new_tape = type(tape)(optimized_tape.operations, optimized_tape.measurements, shots=optimized_tape.shots)
        return [new_tape], lambda results : results[0]

    return transform(transpile)