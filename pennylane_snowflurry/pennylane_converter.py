# from julia import Snowflurry
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
    ProbabilityMP,
    SampleMP,
    ExpectationMP,
    CountsMP,
)
import time
from pennylane.typing import TensorLike
from typing import Callable
from pennylane.ops import Sum, Hamiltonian

# Dictionary mapping PennyLane operations to Snowflurry operations
# The available Snowflurry operations are listed here:
# https://snowflurrysdk.github.io/Snowflurry.jl/dev/library/quantum_toolkit.html
# https://snowflurrysdk.github.io/Snowflurry.jl/dev/library/quantum_gates.html
# https://snowflurrysdk.github.io/Snowflurry.jl/dev/library/quantum_circuit.html
SNOWFLURRY_OPERATION_MAP = {
    "PauliX": "sigma_x({0})",
    "PauliY": "sigma_y({0})",
    "PauliZ": "sigma_z({0})",
    "Hadamard": "hadamard({0})",
    "CNOT": "control_x({0},{1})",
    "CZ": "control_z({0},{1})",
    "SWAP": "swap({0},{1})",
    "ISWAP": "iswap({0},{1})",
    "RX": "rotation_x({1},{0})",
    "RY": "rotation_y({1},{0})",
    "RZ": "rotation_z({1},{0})",  # NOTE : rotation_z is not implemented in snowflurry, phase_shift is the closest thing
    "Identity": "identity_gate({0})",
    "CSWAP": NotImplementedError,
    "CRX": "controlled(rotation_x({1},{0}),{1})",  # gates using controlled probably wont work, might have to do a special operations for those cases.
    "CRY": NotImplementedError,
    "CRZ": NotImplementedError,
    "PhaseShift": "phase_shift({1},{0})",
    "QubitStateVector": NotImplementedError,
    "StatePrep": NotImplementedError,
    "Toffoli": "toffoli({0},{1},{2})",  # order might be wrong on that one
    "QubitUnitary": NotImplementedError,
    "U1": NotImplementedError,
    "U2": NotImplementedError,
    "U3": "universal({3},{0},{1},{2})",
    "IsingZZ": NotImplementedError,
    "IsingYY": NotImplementedError,
    "IsingXX": NotImplementedError,
    "T": "pi_8({0})",
    "Rot": "rotation({3},{0},{1})",  # theta, phi but no omega so we skip {2}, {3} is the wire
    "QubitUnitary": NotImplementedError,  # might correspond to apply_gate!(state::Ket, gate::Gate) from snowflurry
    "QFT": NotImplementedError,
}


"""
if host, user, access_token are left blank, the code will be ran on the simulator
if host, user, access_token are filled, the code will be sent to Anyon's API
"""


class PennylaneConverter:
    """
    supported measurements :
    counts([op, wires, all_outcomes]) arguments have no effect
    expval(op)
    state()
    sample([op, wires]) arguments have no effect
    probs([wires, op]) arguments have no effect

    currently not supported measurements :
    var(op)
    density_matrix(wires)
    vn_entropy(wires[, log_base])
    mutual_info(wires0, wires1[, log_base])
    purity(wires)
    classical_shadow(wires[, seed])
    shadow_expval(H[, k, seed])
    """

    def __init__(
        self,
        circuit: qml.tape.QuantumScript,
        rng=None,
        debugger=None,
        interface=None,
        host="",
        user="",
        access_token="",
    ) -> Result:
        self.circuit = circuit
        self.rng = rng
        self.debugger = debugger
        self.interface = interface
        if len(host) != 0 and len(user) != 0 and len(access_token) != 0:
            Main.currentClient = Main.Eval(
                "Client(host={host},user={user},access_token={access_token})"
            )  # TODO : I think this pauses the execution, check if threading is needed
        else:
            Main.currentClient = None

    def simulate(self):
        sf_circuit, is_state_batched = self.convert_circuit(
            self.circuit, debugger=self.debugger, interface=self.interface
        )
        return self.measure_final_state(
            self.circuit, sf_circuit, is_state_batched, self.rng
        )

    def convert_circuit(
        self, pennylane_circuit: qml.tape.QuantumScript, debugger=None, interface=None
    ):
        """
        Convert the received pennylane circuit into a snowflurry device in julia.
        It is then store into Main.sf_circuit

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
        if len(pennylane_circuit) > 0 and isinstance(
            pennylane_circuit[0], qml.operation.StatePrepBase
        ):
            prep = pennylane_circuit[0]

        # Add gates to Snowflurry circuit
        for op in pennylane_circuit.map_to_standard_wires().operations[bool(prep) :]:
            if op.name in SNOWFLURRY_OPERATION_MAP:
                if SNOWFLURRY_OPERATION_MAP[op.name] == NotImplementedError:
                    print(f"{op.name} is not implemented yet, skipping...")
                    continue
                parameters = op.parameters + [i + 1 for i in op.wires.tolist()]
                gate = SNOWFLURRY_OPERATION_MAP[op.name].format(*parameters)
                print(f"placed {gate}")
                Main.eval(f"push!(sf_circuit,{gate})")
            else:
                print(f"{op.name} is not supported by this device. skipping...")

        Main.print(Main.sf_circuit)

        return Main.sf_circuit, False

    def apply_readouts(self, wires_nb):
        """
        Apply readouts to all wires in the snowflurry circuit.

        Args:
            sf_circuit: The snowflurry circuit to modify.
            wires_nb (int): The number of wires in the circuit.
        """
        for wire in range(wires_nb):
            Main.eval(f"push!(sf_circuit, readout({wire + 1}, {wire + 1}))")

    def measure_final_state(self, circuit, sf_circuit, is_state_batched, rng):
        """
        Perform the measurements required by the circuit on the provided state.

        This is an internal function that will be called by the successor to ``default.qubit``.

        Args:
            circuit (.QuantumScript): The single circuit to simulate
            sf_circuit : The snowflurry circuit used
            is_state_batched (bool): Whether the state has a batch dimension or not.
            rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
                seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
                If no value is provided, a default RNG will be used.

        Returns:
            Tuple[TensorLike]: The measurement results
        """
        # circuit.shots can return the total number of shots with .total_shots or
        # it can return ShotCopies with .shot_vector
        # the case with ShotCopies is not handled as of now

        circuit = circuit.map_to_standard_wires()
        shots = circuit.shots.total_shots
        if shots is None:
            shots = 1

        if len(circuit.measurements) == 1:
            return self.measure(circuit.measurements[0], sf_circuit, shots)
        else:
            return tuple(
                self.measure(mp, sf_circuit, shots) for mp in circuit.measurements
            )

    def measure(self, mp: MeasurementProcess, sf_circuit, shots):
        # if measurement is a qml.counts
        if isinstance(mp, CountsMP):  # this measure can run on hardware
            if Main.currentClient is None:
                # since we use simulate_shots, we need to add readouts to the circuit
                self.apply_readouts(len(self.circuit.op_wires))
                shots_results = Main.simulate_shots(Main.sf_circuit, shots)
                result = dict(Counter(shots_results))
                return result
            else:  # if we have a client, we try to use the real machine
                # NOTE : THE FOLLOWING WILL VERY LIKELY NOT WORK AS IT WAS NOT TESTED
                # I DID NOT RECEIVE THE AUTHENTICATION INFORMATION IN TIME TO TEST IT.
                # WHOEVER WORK ON THIS ON THE FUTURE, CONSIDER THIS LIKE PSEUDOCODE
                # THE CIRCUITID WILL PROBABLY NEED TO BE RAN ON A DIFFERENT THREAD TO NOT STALL THE EXECUTION,
                # YOU CAN MAKE IT STALL IF THE REQUIREMENTS ALLOWS IT
                circuitID = Main.submit_circuit(
                    Main.currentClient, Main.sf_circuit, shots
                )
                status = Main.get_status(circuitID)
                while (
                    status != "succeeded"
                ):  # it won't be "succeeded", need to check what Main.get_status return
                    print(f"checking for status for circuit id {circuitID}")
                    time.sleep(1)
                    status = Main.get_status(circuitID)
                    print(f"current status : {status}")
                    if (
                        status == "failed"
                    ):  # it won't be "failed", need to check what Main.get_status return
                        break
                if status == "succeeded":
                    return Main.get_result(circuitID)

        # if measurement is a qml.sample
        if isinstance(mp, SampleMP):  # this measure can run on hardware
            if Main.currentClient is None:
                # since we use simulate_shots, we need to add readouts to the circuit
                self.apply_readouts(len(self.circuit.op_wires))
                shots_results = Main.simulate_shots(Main.sf_circuit, shots)
                return np.asarray(shots_results).astype(int)
            else:  # if we have a client, we try to use the real machine
                # NOTE : THE FOLLOWING WILL VERY LIKELY NOT WORK AS IT WAS NOT TESTED
                # I DID NOT RECEIVE THE AUTHENTICATION INFORMATION IN TIME TO TEST IT.
                # WHOEVER WORK ON THIS ON THE FUTURE, CONSIDER THIS LIKE PSEUDOCODE
                # THE CIRCUITID WILL PROBABLY NEED TO BE RAN ON A DIFFERENT THREAD TO NOT STALL THE EXECUTION,
                # YOU CAN MAKE IT STALL IF THE REQUIREMENTS ALLOWS IT
                circuitID = Main.submit_circuit(
                    Main.currentClient, Main.sf_circuit, shots
                )
                status = Main.get_status(circuitID)
                while (
                    status != "succeeded"
                ):  # it won't be "succeeded", need to check what Main.get_status return
                    print(f"checking for status for circuit id {circuitID}")
                    time.sleep(1)
                    status = Main.get_status(circuitID)
                    print(f"current status : {status}")
                    if (
                        status == "failed"
                    ):  # it won't be "failed", need to check what Main.get_status return
                        break
                if status == "succeeded":
                    return Main.get_result(circuitID)

        # if measurement is a qml.probs
        if isinstance(mp, ProbabilityMP):
            wires_list = mp.wires.tolist()
            if len(wires_list) == 0:
                return Main.get_measurement_probabilities(Main.sf_circuit)
            else:
                return Main.get_measurement_probabilities(
                    Main.sf_circuit, [i + 1 for i in wires_list]
                )

        # if measurement is a qml.expval
        if isinstance(mp, ExpectationMP):
            Main.result_state = Main.simulate(sf_circuit)
            if mp.obs is not None and mp.obs.has_matrix:
                print(type(mp.obs))
                observable_matrix = qml.matrix(mp.obs)
                return Main.expected_value(
                    Main.DenseOperator(observable_matrix), Main.result_state
                )

        # if measurement is a qml.state
        if isinstance(mp, StateMeasurement):
            Main.result_state = Main.simulate(sf_circuit)
            # Convert the final state from pyjulia to a NumPy array
            final_state_np = np.array([element for element in Main.result_state])
            return final_state_np
        return NotImplementedError
