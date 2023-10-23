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
    "Hadamard": "hadamard",
    "CNOT": NotImplementedError,
    "CZ": NotImplementedError,
    "SWAP": NotImplementedError,
    "ISWAP": NotImplementedError,
    "RX": NotImplementedError,
    "RY": NotImplementedError,
    "RZ": NotImplementedError,
    "Identity": "identity_gate",
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
    def __init__(self, circuit: qml.tape.QuantumScript, rng=None, debugger=None, interface=None, auth='') -> Result:
        self.circuit = circuit
        self.rng = rng
        self.debugger = debugger
        self.interface = interface
        self.auth = auth


    def simulate(self):
        state, is_state_batched = self.get_final_state(self.circuit, debugger=self.debugger, interface=self.interface)
        return self.measure_final_state(self.circuit, state, is_state_batched, self.rng)

    def getGateFromCircuit(circuit):
        return list(circuit)
    


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
        Main.eval("using Snowflurry")
        Main.sf_circuit = Main.QuantumCircuit(qubit_count=1)
        current_wire = 1

        prep = None
        if len(pennylane_circuit) > 0 and isinstance(pennylane_circuit[0], qml.operation.StatePrepBase):
            prep = pennylane_circuit[0]

        # Add gates to Snowflurry circuit
        for op in pennylane_circuit.map_to_standard_wires().operations[bool(prep) :]:
            if op.name in SNOWFLURRY_OPERATION_MAP:
                print(f"placed {op.name}")
                if SNOWFLURRY_OPERATION_MAP[op.name] == NotImplementedError:
                    print(f"{op.name} is not implemented yet, skipping...")
                    continue
                Main.eval(f"push!(sf_circuit,{SNOWFLURRY_OPERATION_MAP[op.name]}({current_wire}))")



        final_state = Main.simulate(Main.sf_circuit)
        print("states: ")
        print(final_state)
        probabilities = Main.get_measurement_probabilities(final_state)
        print(probabilities)
        # Convert the final state to a NumPy array
        final_state_np = np.array(final_state)

        return final_state_np, False
