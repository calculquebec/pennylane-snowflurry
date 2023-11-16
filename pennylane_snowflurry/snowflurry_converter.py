from julia import Snowflurry
from julia import Main
import julia
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.typing import Result, ResultBatch
import numpy as np
from collections import Counter
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
                prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
                    the key to the JAX pseudo random number generator. Only for simulation using JAX.
                    If None, the default ``sample_state`` function and a ``numpy.random.default_rng``
                    will be for sampling.

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
            
            shots_results = Main.simulate_shots(Main.sf_circuit, shots)
            result = dict(Counter(shots_results))
            return result
    
            if not circuit.shots:
                # analytic case

                if len(circuit.measurements) == 1:
                    probabilities = Main.get_measurement_probabilities(Main.final_state)
                    print("supge")
                    print(probabilities)
                    return probabilities #measure(circuit.measurements[0], state, is_state_batched=is_state_batched)

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



        Main.final_state = Main.simulate(Main.sf_circuit)
        # Convert the final state to a NumPy array
        final_state_np = np.array(Main.final_state)

        return final_state_np, False
