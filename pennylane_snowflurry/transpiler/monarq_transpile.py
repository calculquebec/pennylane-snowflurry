from copy import deepcopy
from pennylane.tape import QuantumTape
import pennylane as qml
from pennylane.transforms import transform
import pennylane_snowflurry.transpiler.base_decomposition as step1
import pennylane_snowflurry.transpiler.placement as step2
import pennylane_snowflurry.transpiler.routing as step3
import pennylane_snowflurry.transpiler.optimization as step4
import pennylane_snowflurry.transpiler.native_decomposition as step5


def get_transpiler(baseDecomposition = True, 
                   placeAndRoute = True, 
                   optimization = True, 
                   nativeDecomposition = True,
                   use_benchmark = True):
    """
        returns a transform that goes through 5 transpilation steps
        end circuit should be executable on MonarQ
        every step is optional, leaving modularity to the end user

        Args
            baseDecomposition (bool) : defines if preliminary decomposition should be applied
            placeAndRoute (bool) : defines if placing and routing should be applied
            optimization (bool) : defines if optimization should be applied
            nativeDecomposition (bool) : defines if native decomposition should be applied
            use_benchmark (bool) : defines if benchmark informations should be used for placing and routing
        """
        # TODO : boolean flags could be replaced by a data structure or kwargs. 
    def transpile(tape : QuantumTape):
        """
        goes through 5 transpilation steps
        end circuit should be executable on MonarQ
        every step is optional, leaving modularity to the end user
        """
        optimized_tape = deepcopy(tape)
        with qml.QueuingManager.stop_recording():
            if baseDecomposition: 
                optimized_tape = step1.base_decomposition(optimized_tape)
            if placeAndRoute: 
                optimized_tape = step2.placement_astar(optimized_tape, use_benchmark)
            if placeAndRoute: 
                optimized_tape = step3.swap_routing(optimized_tape, use_benchmark)
                if baseDecomposition:
                    optimized_tape = step1.base_decomposition(optimized_tape)
            
            if optimization:
                optimized_tape = step4.optimize(optimized_tape)

            if nativeDecomposition:
                optimized_tape = step5.native_gate_decomposition(optimized_tape)
                if optimization:
                    optimized_tape_before = optimized_tape
                    optimized_tape = step4.optimize(optimized_tape)
                    optimized_tape = step5.native_gate_decomposition(optimized_tape)
                    while len(optimized_tape_before.operations) > len(optimized_tape.operations):
                        optimized_tape_before = optimized_tape
                        optimized_tape = step4.optimize(optimized_tape)
                        optimized_tape = step5.native_gate_decomposition(optimized_tape)
                    optimized_tape = optimized_tape_before

        new_tape = type(tape)(optimized_tape.operations, optimized_tape.measurements, shots=optimized_tape.shots)
        return [new_tape], lambda results : results[0]

    return transform(transpile)