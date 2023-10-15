from julia import Snowflurry
from julia import Main
import julia
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.typing import Result, ResultBatch
import numpy as np
#https://snowflurrysdk.github.io/Snowflurry.jl/dev/library.html#Snowflurry.sigma_z
SNOWFLURRY_OPERATION_MAP = {
    # native PennyLane native to snowflurry
    "PauliX": "sigma_x",
    "PauliY": "sigma_y",
    "PauliZ": "sigma_z",
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
        self.circuit = circuit
        self.rng = rng
        self.debugger = debugger
        self.interface = interface


    def simulate(self):
        state, is_state_batched = self.get_final_state(self.circuit, debugger=self.debugger, interface=self.interface)
        return self.measure_final_state(self.circuit, state, is_state_batched, self.rng)

    def getGateFromCircuit(circuit):
        return list(circuit)

    # def get_final_state(circuit, debugger, interface):
    #     return circuit.eval()
    


    def measure_final_state(self, circuit, state, is_state_batched, rng):
        return state
    
    def get_final_state(self, pennylane_circuit: qml.tape.QuantumScript, debugger=None, interface=None):
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
        Main.sf_circuit = sf_circuit

        prep = None
        if len(pennylane_circuit) > 0 and isinstance(pennylane_circuit[0], qml.operation.StatePrepBase):
            prep = pennylane_circuit[0]
        # Add gates to Snowflurry circuit
        for op in pennylane_circuit.map_to_standard_wires().operations[bool(prep) :]:
            if op.name in SNOWFLURRY_OPERATION_MAP:
                current_wire = 0
                print(f"placed {op.name}")
                Snowflurry.eval(f"push!(sf_circuit,{SNOWFLURRY_OPERATION_MAP[op.name]}({current_wire}))")

        # Simulate the circuit using Snowflurry
        final_state = Snowflurry.simulate(sf_circuit)

        # Convert the final state to a NumPy array
        # Note: Adjust the conversion based on how Snowflurry represents states
        final_state_np = np.array(final_state)

        return final_state_np, False
