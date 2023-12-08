from julia import Snowflurry
from julia import Main
import julia
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.typing import Result, ResultBatch
import numpy as np
from collections import Counter
from pennylane.measurements import (
    StateMeasurement,
    MeasurementProcess,
    MeasurementValue,
    ExpectationMP,
    CountsMP,
)
from pennylane.typing import TensorLike
from typing import Callable
from pennylane.ops import Sum, Hamiltonian
#https://snowflurrysdk.github.io/Snowflurry.jl/dev/library.html#Snowflurry.sigma_z
SNOWFLURRY_OPERATION_MAP = {
    # native PennyLane native to snowflurry
    "PauliX": "sigma_x({0})",
    "PauliY": "sigma_y({0})",
    "PauliZ": "sigma_z({0})",
    "Hadamard": "hadamard({0})",
    "CNOT": "controlled(sigma_x({1}),{0})",
    "CZ": NotImplementedError,
    "SWAP": NotImplementedError,
    "ISWAP": NotImplementedError,
    "RX": NotImplementedError,
    "RY": NotImplementedError,
    "RZ": NotImplementedError,
    "Identity": "identity_gate({0})",
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





"""
if auth is left blank, the code will be ran on the simulator
if auth is filled, the code will be sent to Anyon's API
"""
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


    """
    supported measurements : 
    counts([op, wires, all_outcomes])
    expval(op)
    state()

    currently not supported measurements : 
    sample([op, wires])
    probs([wires, op])
    var(op)
    density_matrix(wires)
    vn_entropy(wires[, log_base])
    mutual_info(wires0, wires1[, log_base])
    purity(wires)
    classical_shadow(wires[, seed])
    shadow_expval(H[, k, seed])
    """
    def measure_final_state(self, circuit, state, is_state_batched, rng):
            """
            Perform the measurements required by the circuit on the provided state.

            This is an internal function that will be called by the successor to ``default.qubit``.

            Args:
                circuit (.QuantumScript): The single circuit to simulate
                state (TensorLike): The state to perform measurement on
                is_state_batched (bool): Whether the state has a batch dimension or not.
                rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
                    seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
                    If no value is provided, a default RNG will be used.

            Returns:
                Tuple[TensorLike]: The measurement results
            """
            #circuit.shots can return the total number of shots with .total_shots or 
            #it can return ShotCopies with .shot_vector
            #the case with ShotCopies is not handled as of now
            
            circuit = circuit.map_to_standard_wires()
            shots = circuit.shots.total_shots
            if shots is None:
                shots = 1
            print(circuit.measurements)
            print(circuit.measurements[0])
            if len(circuit.measurements) == 1:
                pass
            else:
                tuple(print(mp) for mp in circuit.measurements)
            if isinstance(circuit.measurements[0], ExpectationMP):
                if circuit.measurements[0].obs is not None and circuit.measurements[0].obs.has_matrix:
                    observable_matrix = circuit.measurements[0].obs.compute_matrix()
                    return Main.expected_value(Main.DenseOperator(observable_matrix), Main.result_state)
                
            # actual sampling cases
            
            if isinstance(circuit.measurements[0], CountsMP):
                print("supge")
            if isinstance(circuit.measurements[0], StateMeasurement):
                print("supgestate")
                return state
            shots_results = Main.simulate_shots(Main.sf_circuit, shots)
            result = dict(Counter(shots_results))
            return result
    
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
        wires_nb = len(pennylane_circuit.op_wires)
        Main.sf_circuit = Main.QuantumCircuit(qubit_count=wires_nb)

        prep = None
        if len(pennylane_circuit) > 0 and isinstance(pennylane_circuit[0], qml.operation.StatePrepBase):
            prep = pennylane_circuit[0]

        # Add gates to Snowflurry circuit
        for op in pennylane_circuit.map_to_standard_wires().operations[bool(prep) :]:
            if op.name in SNOWFLURRY_OPERATION_MAP:
                if SNOWFLURRY_OPERATION_MAP[op.name] == NotImplementedError:
                    print(f"{op.name} is not implemented yet, skipping...")
                    continue
                gate = SNOWFLURRY_OPERATION_MAP[op.name].format(*[i+1 for i in op.wires.tolist()])
                print(f"placed {gate}")
                Main.eval(f"push!(sf_circuit,{gate})")


        Main.result_state = Main.simulate(Main.sf_circuit)
        # Convert the final state to a NumPy array
        final_state_np = np.array([element for element in Main.result_state])

        return final_state_np, False