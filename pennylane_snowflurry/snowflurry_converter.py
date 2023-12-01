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
)
from pennylane.typing import TensorLike
from typing import Callable

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
    not supported measurements : 
    expval(op)
    sample([op, wires])
    var(op)
    probs([wires, op])
    state()
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
            print(circuit.measurements)
            if shots is None:
                shots = 1
            
            shots_results = Main.simulate_shots(Main.sf_circuit, shots)
            result = dict(Counter(shots_results))
            return result
    
            if not circuit.shots:
                # analytic case

                if len(circuit.measurements) == 1:
                    return measure(circuit.measurements[0], state, is_state_batched=is_state_batched)

                return tuple(
                    measure(mp, state, is_state_batched=is_state_batched) for mp in circuit.measurements
                )

            # finite-shot case

            rng = default_rng(rng)
            results = measure_with_samples(
                circuit.measurements,
                state,
                shots=circuit.shots,
                is_state_batched=is_state_batched,
                rng=rng,
                prng_key=prng_key,
            )

            if len(circuit.measurements) == 1:
                if circuit.shots.has_partitioned_shots:
                    return tuple(res[0] for res in results)

                return results[0]

            return results
            
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
        Main.sf_circuit = Main.QuantumCircuit(qubit_count=len(pennylane_circuit.op_wires))

        prep = None
        if len(pennylane_circuit) > 0 and isinstance(pennylane_circuit[0], qml.operation.StatePrepBase):
            prep = pennylane_circuit[0]

        # Add gates to Snowflurry circuit
        for op in pennylane_circuit.map_to_standard_wires().operations[bool(prep) :]:
            if op.name in SNOWFLURRY_OPERATION_MAP:
                if SNOWFLURRY_OPERATION_MAP[op.name] == NotImplementedError:
                    print(f"{op.name} is not implemented yet, skipping...")
                    continue
                gate = SNOWFLURRY_OPERATION_MAP[op.name].format(*op.wires.tolist())
                print(f"placed {gate}")
                Main.eval(f"push!(sf_circuit,{gate})")



        print("testing")
        Main.final_state = Main.eval("simulate(sf_circuit)")
        print("testing2")
        # Convert the final state to a NumPy array
        final_state_np = np.array(Main.final_state)

        return final_state_np, False

# new sections 
    

    def get_measurement_function(self,
        measurementprocess: MeasurementProcess, state: TensorLike
    ) -> Callable[[MeasurementProcess, TensorLike], TensorLike]:
        """Get the appropriate method for performing a measurement.

        Args:
            measurementprocess (MeasurementProcess): measurement process to apply to the state
            state (TensorLike): the state to measure
            is_state_batched (bool): whether the state is batched or not

        Returns:
            Callable: function that returns the measurement result
        """
        if isinstance(measurementprocess, StateMeasurement):
            if isinstance(measurementprocess.mv, MeasurementValue):
                return state_diagonalizing_gates

            if isinstance(measurementprocess, ExpectationMP):
                if measurementprocess.obs.name == "SparseHamiltonian":
                    return csr_dot_products

                backprop_mode = math.get_interface(state, *measurementprocess.obs.data) != "numpy"
                if isinstance(measurementprocess.obs, Hamiltonian):
                    # need to work out thresholds for when its faster to use "backprop mode" measurements
                    return sum_of_terms_method if backprop_mode else csr_dot_products

                if isinstance(measurementprocess.obs, Sum):
                    if backprop_mode:
                        # always use sum_of_terms_method for Sum observables in backprop mode
                        return sum_of_terms_method
                    if (
                        measurementprocess.obs.has_overlapping_wires
                        and len(measurementprocess.obs.wires) > 7
                    ):
                        # Use tensor contraction for `Sum` expectation values with non-commuting summands
                        # and 8 or more wires as it's faster than using eigenvalues.

                        return csr_dot_products

            if measurementprocess.obs is None or measurementprocess.obs.has_diagonalizing_gates:
                return state_diagonalizing_gates

        raise NotImplementedError