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
import re
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
    "CY": "controlled(sigma_y({1}),[{0}])",  # 0 is the control qubit, 1 is the target qubit
    "CZ": "control_z({0},{1})",
    "SWAP": "swap({0},{1})",
    "ISWAP": "iswap({0},{1})",
    "RX": "rotation_x({1},{0})",
    "RY": "rotation_y({1},{0})",
    "RZ": "rotation_z({1},{0})",
    "Identity": "identity_gate({0})",
    "CSWAP": "controlled(swap({1},{2}),[{0}])",  # 0 is the control qubit, 1 and 2 are the target qubits
    "CRX": "controlled(rotation_x({2},{0}),[{1}])",  # 0 is the angle, 1 is the control qubit, 2 is the target qubit
    "CRY": "controlled(rotation_y({2},{0}),[{1}])",
    "CRZ": "controlled(rotation_z({2},{0}),[{1}])",
    "PhaseShift": "phase_shift({1},{0})",  # 0 is the angle, 1 is the wire
    "ControlledPhaseShift": "controlled(phase_shift({2},{0}),[{1}])",  # 0 is the angle, 1 is the control qubit, 2 is the target qubit
    "Toffoli": "toffoli({0},{1},{2})",
    "U3": "universal({3},{0},{1},{2})", # 3 is the wire, 0,1,2 are theta, phi, delta respectively
    "T": "pi_8({0})",
    "Rot": "rotation({3},{0},{1})",  # theta, phi but no omega so we skip {2}, {3} is the wire
}


"""
if host, user, access_token are left blank, the code will be ran on the simulator
if host, user, access_token are filled, the code will be sent to Anyon's API
"""


class PennylaneConverter:
    """
    A PennyLane converter for the Snowflurry device.

    Is in charge of interfacing with the Snowflurry.jl package and converting PennyLane circuits to
    Snowflurry circuits.
    """

    ###################################
    # Class attributes used for logic #
    ###################################
    snowflurry_readout_name = "Readout"
    snowflurry_gate_object_name = "Gate Object"

    # Pattern is found in PyCall.jlwrap object of Snowflurry.QuantumCircuit.instructions
    snowflurry_str_search_pattern = r"Gate Object: (.*)\n"

    #################
    # Class methods #
    #################

    def __init__(
        self,
        circuit: qml.tape.QuantumScript,
        rng=None,
        debugger=None,
        interface=None,
        host="",
        user="",
        access_token="",
        project_id="",
    ) -> Result:

        # Instance attributes related to PennyLane
        self.circuit = circuit
        self.rng = rng
        self.debugger = debugger
        self.interface = interface

        # Instance attributes related to Snowflurry
        if (
            len(host) != 0
            and len(user) != 0
            and len(access_token) != 0
            and len(project_id) != 0
        ):
            Main.currentClient = Main.Eval(
                "Client(host={host},user={user},access_token={access_token}, project_id={project_id})"
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
                Main.eval(f"push!(sf_circuit,{gate})")
            else:
                print(f"{op.name} is not supported by this device. skipping...")

        return Main.sf_circuit, False

    def apply_readouts(self, wires_nb, obs):
        """
        Apply readouts to all wires in the snowflurry circuit.

        Args:
            wires_nb (int): The number of wires in the circuit.
            obs (Optional[Observable]): The observable mentioned in the measurement process. If None,
                readouts are applied to all wires because we assume the user wants to measure all wires.
        """

        if obs is None:  # if no observable is given, we apply readouts to all wires
            for wire in range(wires_nb):
                Main.eval(f"push!(sf_circuit, readout({wire + 1}, {wire + 1}))")

        else:
            # if an observable is given, we apply readouts to the wires mentioned in the observable,
            # TODO : could add Pauli rotations to get the correct observable
            self.apply_single_readout(obs.wires[0])

    def get_circuit_as_dictionary(self):
        """
        Take the snowflurry QuantumCircuit.instructions and convert it to an array of operations.
        When instruction is called from Snowflurry, PyCall returns a jlwrap object which is not easily
        iterable. This function is used to convert the jlwrap object to a Python dictionary.

        Returns:
            Dict [str, [int]]: A dictionary containing the operations and an array of the wires they are
                applied to.

        Example:
            >>> Main.sf_circuit.instructions
            [<PyCall.jlwrap Gate Object: Snowflurry.Hadamard
            Connected_qubits        : [1]
            Operator:
            (2, 2)-element Snowflurry.DenseOperator:
            Underlying data ComplexF64:
            0.7071067811865475 + 0.0im    0.7071067811865475 + 0.0im
            0.7071067811865475 + 0.0im    -0.7071067811865475 + 0.0im
            >, <PyCall.jlwrap Gate Object: Snowflurry.ControlX
            Connected_qubits        : [2, 1]
            Operator:
            (4, 4)-element Snowflurry.DenseOperator:
            Underlying data ComplexF64:
            1.0 + 0.0im    0.0 + 0.0im    0.0 + 0.0im    0.0 + 0.0im
            0.0 + 0.0im    1.0 + 0.0im    0.0 + 0.0im    0.0 + 0.0im
            0.0 + 0.0im    0.0 + 0.0im    0.0 + 0.0im    1.0 + 0.0im
            0.0 + 0.0im    0.0 + 0.0im    1.0 + 0.0im    0.0 + 0.0im
            >, <PyCall.jlwrap Explicit Readout object:
            connected_qubit: 1
            destination_bit: 1
            >]

            Becomes:
            [{'gate': 'Snowflurry.Hadamard', 'connected_qubits': [1]},
            {'gate': 'Snowflurry.ControlX', 'connected_qubits': [1, 2]},
            {'gate': 'Readout', 'connected_qubits': [1]}]


        """
        ops = []
        instructions = Main.sf_circuit.instructions  # instructions is a jlwrap object
        gate_str = ""
        gate_name = ""

        for inst in instructions:

            gate_str = str(inst)  # convert the jlwrap object to a string

            try:
                if self.snowflurry_gate_object_name in gate_str:
                    # if the gate is a Gate object, we extract the name and the connected qubits
                    # from the string with a regex
                    gate_name = re.search(
                        self.snowflurry_str_search_pattern, gate_str
                    ).group(1)
                    op_data = {
                        "gate": gate_name,
                        "connected_qubits": list(inst.connected_qubits),
                    }
                if self.snowflurry_readout_name in gate_str:
                    # if the gate is a Readout object, we extract the connected qubit from the string
                    gate_name = self.snowflurry_readout_name
                    op_data = {
                        "gate": gate_name,
                        "connected_qubits": [inst.connected_qubit],
                    }
                # NOTE : attribute for the Gate object is connected_qubits (plural)
                # while the attribute for the Readout object is connected_qubit (singular)

            except:
                raise ValueError(f"Error while parsing {gate_str}")

            ops.append(op_data)

        return ops

    def has_readout(self) -> bool:
        """
        Check if a readout is applied on any of the wires in the snowflurry circuit.

        Returns:
            bool: True if a readout is applied, False otherwise.
        """
        ops = self.get_circuit_as_dictionary()
        for op in ops:
            if op["gate"] == self.snowflurry_readout_name:
                return True
        return False

    def remove_readouts(self):
        """
        Remove all readouts from the snowflurry circuit with pop!() function.

        """
        # RFE : eventually, removing the readouts could be done by making a copy
        # of the instructions vector and removing the readouts from it before
        # contructing a new QuantumCircuit with that vector.
        while self.has_readout():
            Main.eval("pop!(sf_circuit)")

    def apply_single_readout(self, wire):
        """
        Apply a readout to a single wire in the snowflurry circuit.

        Args:
            wire (int): The wire to apply the readout to.
        """
        ops = self.get_circuit_as_dictionary()

        for op in ops:
            # if a readout is already applied to the wire, we don't apply another one
            if op["gate"] == self.snowflurry_readout_name:
                if op["connected_qubits"] == wire - 1:  # wire is 1-indexed in Julia
                    return

        # if no readout is applied to the wire, we apply one while taking into account that
        # the wire number is 1-indexed in Julia
        Main.eval(f"push!(sf_circuit, readout({wire+1}, {wire+1}))")

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
            results = self.measure(circuit.measurements[0], sf_circuit, shots)
        else:
            results = tuple(
                self.measure(mp, sf_circuit, shots) for mp in circuit.measurements
            )

        Main.print(Main.sf_circuit)

        return results

    def measure(self, mp: MeasurementProcess, sf_circuit, shots):
        """
        Measure the quantum state using the provided measurement process.

        Args:
            mp (MeasurementProcess): The measurement process to perform
            sf_circuit : The snowflurry circuit used
            shots (int): The number of shots

        Returns:
            result: The measurement result TODO : type needs to be unified

        Currently supported measurements :
            - counts(works with Snowflurry.simulate_shots)
            - sample(works with Snowflurry.simulate_shots)
            - probs(works with Snowflurry.get_measurement_probabilities)
            - expval(works with Snowflurry.simulate and Snowflurry.expected_value)
            - state(works with Snowflurry.simulate and Snowflurry.result_state)

        """

        # if measurement is a qml.counts
        if isinstance(mp, CountsMP):  # this measure can run on hardware
            if Main.currentClient is None:
                # since we use simulate_shots, we need to add readouts to the circuit
                self.remove_readouts()
                self.apply_readouts(len(self.circuit.op_wires), mp.obs)
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
                self.remove_readouts()
                self.apply_readouts(len(self.circuit.op_wires), mp.obs)
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
            self.remove_readouts()
            wires_list = mp.wires.tolist()
            if len(wires_list) == 0:
                return Main.get_measurement_probabilities(Main.sf_circuit)
            else:
                return Main.get_measurement_probabilities(
                    Main.sf_circuit, [i + 1 for i in wires_list]
                )

        # if measurement is a qml.expval
        if isinstance(mp, ExpectationMP):
            # FIXME : this measurement only works with a single wire
            # Requires some processing to work with larger matrices
            self.remove_readouts()
            Main.result_state = Main.simulate(Main.sf_circuit)
            if mp.obs is not None and mp.obs.has_matrix:
                print(type(mp.obs))
                observable_matrix = qml.matrix(mp.obs)
                return Main.expected_value(
                    Main.DenseOperator(observable_matrix), Main.result_state
                )

        # if measurement is a qml.state
        if isinstance(mp, StateMeasurement):
            self.remove_readouts()
            Main.result_state = Main.simulate(Main.sf_circuit)
            # Convert the final state from pyjulia to a NumPy array
            final_state_np = np.array([element for element in Main.result_state])
            return final_state_np

        return NotImplementedError
