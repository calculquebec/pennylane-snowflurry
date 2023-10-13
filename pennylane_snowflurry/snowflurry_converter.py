from julia import Snowflurry
from julia import Main
import pennylane as qml
from pennylane.tape import QuantumTape
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

        state, is_state_batched = self.get_final_state(circuit, debugger=debugger, interface=interface)
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


    # def get_final_state(circuit, debugger, interface):
    #     return circuit.eval()
    


    def measure_final_state(circuit, state, is_state_batched, rng, prng_key):
        return NotImplementedError
    
    def get_final_state(self, circuit: qml.tape.QuantumScript, debugger=None, interface=None):
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
        sf_circuit = Snowflurry.QuantumCircuit(qubit_count=1)
        print(f"muhtype : {type(circuit)}")
        prep = None
        if len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase):
            prep = circuit[0]
        # Add gates to Snowflurry circuit
        for op in circuit.map_to_standard_wires().operations[bool(prep) :]:
            # Here you'll need to map PennyLane gate names to Snowflurry gate functions
            # For example, if op.name is 'PauliX' and it's applied on wire 0:
            if op.name == "PauliX":
                current_wire = 0
                julia.eval(f"push!(sf_circuit,sigma_x({current_wire}))")
            # Add similar conditions for other gate types

        # Simulate the circuit using Snowflurry
        final_state = Snowflurry.simulate(sf_circuit)

        # Convert the final state to a NumPy array
        # Note: Adjust the conversion based on how Snowflurry represents states
        final_state_np = np.array(final_state)

        return final_state_np, False
