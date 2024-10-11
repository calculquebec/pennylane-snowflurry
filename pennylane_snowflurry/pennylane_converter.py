from juliacall import newmodule
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.typing import Result
from pennylane.measurements import (
    StateMeasurement,
    MeasurementProcess,
    MeasurementValue,
    ProbabilityMP,
    SampleMP,
    ExpectationMP,
    CountsMP,
    StateMP,
)
import time
import re
from pennylane.typing import TensorLike
from typing import Callable, Type
from pennylane.ops import Sum, Hamiltonian

from pennylane_snowflurry.measurements import (
    MeasurementStrategy,
    Sample,
    Counts,
    Probabilities,
    ExpectationValue,
    State
)

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
    "ControlledPhaseShift": "controlled(phase_shift({2},{0}),[{1}])",
    # 0 is the angle, 1 is the control qubit, 2 is the target qubit
    "Toffoli": "toffoli({0},{1},{2})",
    "U3": "universal({3},{0},{1},{2})",  # 3 is the wire, 0,1,2 are theta, phi, delta respectively
    "T": "pi_8({0})",
    "Rot": "rotation({3},{0},{1})",  # theta, phi but no omega so we skip {2}, {3} is the wire
}


"""
if host, user, access_token are left blank, the code will be ran on the simulator
if host, user, access_token are filled, the code will be sent to Anyon's API
"""

##########################################
# Defining namespace for Snowflurry      #
##########################################
Snowflurry = newmodule("Snowflurry")
Snowflurry.seval("using Snowflurry")


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
        pennylane_circuit: QuantumTape,
        debugger=None,
        interface=None,
        host="",
        user="",
        access_token="",
        project_id="",
        realm="",
        wires=None
    ):

        # Instance attributes related to PennyLane
        self.pennylane_circuit = pennylane_circuit
        self.debugger = debugger
        self.interface = interface
        self.wires = wires

        # Instance attributes related to Snowflurry
        self.snowflurry_py_circuit = None
        if (
            len(host) != 0
            and len(user) != 0
            and len(access_token) != 0
            and len(realm) != 0
        ):
            Snowflurry.currentClient = Snowflurry.Client(
                host=host, user=user, access_token=access_token, realm=realm
            )
            Snowflurry.seval('project_id="' + project_id + '"')
        else:
            Snowflurry.currentClient = None

        self.measurementStrategy = None

    def simulate(self):
        self.snowflurry_py_circuit = self.convert_circuit(
            self.pennylane_circuit
        )
        return self.measure_final_state()

    def convert_circuit(
        self, pennylane_circuit: QuantumTape,
    ):
        """
        Convert the received pennylane circuit into a snowflurry device in julia.
        It is then store into Snowflurry.sf_circuit

        Args:
            pennylane_circuit (QuantumTape): The circuit to simulate.

        Returns:
            Tuple[TensorLike, bool]: A tuple containing the final state of the quantum script and
                a boolean indicating if the state has a batch dimension.
        """

        wires_nb = self.wires  # default number of wires in the circuit
        Snowflurry.sf_circuit = Snowflurry.QuantumCircuit(qubit_count=wires_nb)

        prep = None
        if len(pennylane_circuit) > 0 and isinstance(
            pennylane_circuit[0], qml.operation.StatePrepBase
        ):
            prep = pennylane_circuit[0]

        # Add gates to Snowflurry circuit
        for op in pennylane_circuit.operations[bool(prep):]:
            if op.name in SNOWFLURRY_OPERATION_MAP:
                if SNOWFLURRY_OPERATION_MAP[op.name] == NotImplementedError:
                    print(f"{op.name} is not implemented yet, skipping...")
                    continue
                parameters = op.parameters + [i + 1 for i in op.wires.tolist()]
                gate = SNOWFLURRY_OPERATION_MAP[op.name].format(*parameters)
                Snowflurry.seval(f"push!(sf_circuit,{gate})")
            else:
                print(f"{op.name} is not supported by this device. skipping...")

        return Snowflurry.sf_circuit

    def apply_readouts(self, obs):
        """
        Apply readouts to all wires in the snowflurry circuit.

        Args:
            obs (Optional[Observable]): The observable mentioned in the measurement process. If None,
                readouts are applied to all wires because we assume the user wants to measure all wires.
        """

        if obs is None:  # if no observable is given, we apply readouts to all wires
            for wire in range(self.wires):
                Snowflurry.seval(f"push!(sf_circuit, readout({wire + 1}, {wire + 1}))")

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
            >>> Snowflurry.sf_circuit.instructions
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
        instructions = (
            Snowflurry.sf_circuit.instructions
        )  # instructions is a jlwrap object
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
            Snowflurry.seval("pop!(sf_circuit)")

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
        Snowflurry.seval(f"push!(sf_circuit, readout({wire+1}, {wire+1}))")

    def measure_final_state(self):
        """
        Perform the measurements required by the circuit on the provided state.

        This is an internal function that will be called by the successor to ``default.qubit``.

        Returns:
            Tuple[TensorLike]: The measurement results
        """
        # circuit.shots can return the total number of shots with .total_shots or
        # it can return ShotCopies with .shot_vector
        # the case with ShotCopies is not handled as of now

        circuit = self.pennylane_circuit.map_to_standard_wires()
        shots = circuit.shots.total_shots
        if shots is None:
            shots = 1

        if len(circuit.measurements) == 1:
            results = self.measure(
                circuit.measurements[0], shots
            )
        else:
            results = tuple(
                self.measure(mp, shots)
                for mp in circuit.measurements
            )

        # Snowflurry.print(Snowflurry.sf_circuit) # uncomment to print the circuit while debugging

        return results

    def measure(self, mp: MeasurementProcess, shots):
        """
        Measure the quantum state using the provided measurement process.

        Args:
            mp (MeasurementProcess): The measurement process to perform
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
        self.measurementStrategy = self.get_strategy(mp)
        result = self.measurementStrategy.measure(self, mp, shots)
        return result

    def get_strategy(self, mp: MeasurementProcess):
        """
        Get the strategy to use for the measurement process.

        Args:
            mp (MeasurementProcess): The measurement process to perform

        Returns:
            MeasurementStrategy: The strategy to use for the measurement process
        """
        if isinstance(mp, CountsMP):
            return Counts()
        elif isinstance(mp, SampleMP):
            return Sample()
        elif isinstance(mp, ProbabilityMP):
            return Probabilities()
        elif isinstance(mp, ExpectationMP):
            return ExpectationValue()
        elif isinstance(mp, StateMP):
            return State()
        else:
            raise ValueError(f"Measurement process {mp} is not supported by this device.")
