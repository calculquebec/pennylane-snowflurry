from julia import Snowflurry
from julia import Main
import pennylane as qml
from pennylane.typing import Result, ResultBatch
#https://snowflurrysdk.github.io/Snowflurry.jl/dev/library.html#Snowflurry.sigma_z
SNOWFLURRY_OPERATION_MAP = {
    # native PennyLane native to snowflurry
    "PauliX": NotImplementedError,
    "PauliY": NotImplementedError,
    "PauliZ": NotImplementedError,
    "Hadamard": NotImplementedError,
    "CNOT": NotImplementedError,
    "CZ": NotImplementedError,
    "SWAP": NotImplementedError,
    "ISWAP": NotImplementedError,
    "RX": NotImplementedError,
    "RY": NotImplementedError,
    "RZ": NotImplementedError,
    "Identity": NotImplementedError,
    "CSWAP": NotImplementedError,
    "CRX": NotImplementedError,
    "CRY": NotImplementedError,
    "CRZ": NotImplementedError,
    "PhaseShift": NotImplementedError,
    "QubitStateVector": NotImplementedError,
    "StatePrep": NotImplementedError,
    "Toffoli": NotImplementedError,
    "QubitUnitary": NotImplementedError,
    "U1": NotImplementedError,
    "U2": NotImplementedError,
    "U3": NotImplementedError,
    "IsingZZ": NotImplementedError,
    "IsingYY": NotImplementedError,
    "IsingXX": NotImplementedError,
}















class PennylaneConverter:
    def __init__(self, circuit: qml.tape.QuantumScript, rng=None, debugger=None, interface=None ) -> Result:
        c = Snowflurry.QuantumCircuit(qubit_count=3)

        state, is_state_batched = get_final_state(circuit, debugger=debugger, interface=interface)
        return self.measure_final_state(circuit, state, is_state_batched, rng=rng)
    def getGateFromCircuit(circuit):
        return list(circuit)
    
    def applyOperation(ops):
        c = Snowflurry.QuantumCircuit(qubit_count=3)#3 to be replaced by the items in the loop
        julia_operation = ""
        for key, value in ops.items():
            if SNOWFLURRY_OPERATION_MAP[key] is NotImplementedError:
                print(f"!! {key} is not implemented, skipping!!")
                continue
            julia_operation.append(SNOWFLURRY_OPERATION_MAP[key](*value))
        return Main.eval(f"Snowflurry.QuantumCircuit(qubit_count=3)\n{julia_operation}")


    def get_final_state(circuit, debugger, interface):
        return circuit.eval()
    


    def measure_final_state(circuit, state, is_state_batched, rng, prng_key):
        return NotImplementedError
    
    def get_final_state(circuit: QuantumTape, debugger=None, interface=None):
        """
        Get the final state for the SnowflurryQubitDevice.

        Args:
            circuit (QuantumTape): The circuit to simulate.
            debugger (optional): Debugger instance, if debugging is needed.
            interface (str, optional): The interface to use for any necessary conversions.

        Returns:
            Tuple[TensorLike, bool]: A tuple containing the final state of the quantum script and
                a boolean indicating if the state has a batch dimension.
        """
        # Process the circuit to extract the final state

        # pseudocode:
        # state = initialize_state(...)
        # for op in circuit.operations:
        #     state = apply_op_to_state(op, state)
        # is_state_batched = check_if_state_is_batched(state)

        return state, is_state_batched
