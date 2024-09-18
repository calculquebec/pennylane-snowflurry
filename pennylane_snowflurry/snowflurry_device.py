from typing import Union, Callable, Tuple, Optional, Sequence
import numpy as np
from pennylane import Device
import abc
import pennylane as qml
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.typing import Result, ResultBatch
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.transforms.core import TransformProgram
from pennylane.operation import Operator
from pennylane.devices.preprocess import decompose
from pennylane_snowflurry.pennylane_converter import PennylaneConverter
from pennylane_snowflurry.pennylane_converter import SNOWFLURRY_OPERATION_MAP
from pennylane_snowflurry.execution_config import (
    ExecutionConfig,
    DefaultExecutionConfig,
)
from pennylane.typing import Result, ResultBatch
from ._version import __version__


# The plugin does not support batching yet, but this is a placeholder for future implementation
Result_or_ResultBatch = Union[Result, ResultBatch]
QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[
    QuantumTape, QuantumTapeBatch
]  # type : either a single QuantumTape or a Sequence of QuantumTape


def stopping_condition(op: qml.operation.Operator) -> bool:
    r"""Specify whether or not an Operator object is supported by the device.

      Will be used to determine whether or not to decompose an Operator object.

    Args:
        op (Operator): a PennyLane Operator object.

    Returns:
        bool: True if the Operator is supported by the device, False otherwise.

    Note:
        - MultiControlledX needs work_wires to be decomposed, so any circuit
        containing MultiControlledX must specify work_wires in its hyperparameters.
        This operator should eventually be mapped to a Snowflurry operation and won't
        need to be decomposed.
    """
    if op.name not in SNOWFLURRY_OPERATION_MAP.keys():
        return False
    if op.name == "GroverOperator":
        return False
    if (
        op.name == "MultiControlledX"
    ):  # TODO : remove this condition once MultiControlledX is supported
        return False
    if op.name == "Snapshot":
        return True
    if op.__class__.__name__[:3] == "Pow" and qml.operation.is_trainable(op):
        return False

    return op.has_matrix


class SnowflurryQubitDevice(qml.devices.Device):
    """Snowflurry Qubit PennyLane device for interfacing with Anyon's quantum simulators or quantum Hardware.

    * Extends the PennyLane :class:`~.pennylane.Device` class.
    * Snowflurry API credentials are only required for sending jobs on Anyon System's QPU.
    * Batching is not supported yet.

    Args:
        wires (int, Iterable[Number, str]): Number of wires present on the device, or iterable that
            contains unique labels for the wires as numbers (i.e., ``[-1, 0, 2]``) or strings
            (``['ancilla', 'q1', 'q2']``). Default ``None`` if not specified.
        shots (int, Sequence[int], Sequence[Union[int, Sequence[int]]]): The default number of shots
            to use in executions involving this device.
        seed (Union[str, None, int, array_like[int], SeedSequence, BitGenerator, Generator, jax.random.PRNGKey]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``, or
            a request to seed from numpy's global random number generator.
            The default, ``seed="global"`` pulls a seed from NumPy's global generator. ``seed=None``
            will pull a seed from the OS entropy.
            If a ``jax.random.PRNGKey`` is passed as the seed, a JAX-specific sampling function using
            ``jax.random.choice`` and the ``PRNGKey`` will be used for sampling rather than
            ``numpy.random.default_rng``.
        max_workers (int): A ``ProcessPoolExecutor`` executes tapes asynchronously
            using a pool of at most ``max_workers`` processes. If ``max_workers`` is ``None``,
            only the current process executes tapes. If you experience any
            issue, say using JAX, TensorFlow, Torch, try setting ``max_workers`` to ``None``.
        host (str): URL of the QPU server.
        user (str): Username.
        access_token (str): User access token.
        project_id (str): Used to identify which project the jobs sent to this QPU belong to.

    """  # host, user, access_token, project_id would ideally be keyword args

    def __init__(
        self,
        wires=None,
        shots=None,
        seed="global",
        host="",
        user="",
        access_token="",
        project_id="",
        realm="",
    ) -> None:
        super().__init__(wires=wires, shots=shots)

        seed = np.random.randint(0, high=10000000) if seed == "global" else seed
        self._rng = np.random.default_rng(seed)
        self.host = host
        self.user = user
        self.access_token = access_token
        self.project_id = project_id
        self.realm = realm
        self._debugger = None

    pennylane_requires = ">=0.30.0"

    name = "Snowflurry Qubit Device"
    short_name = "snowflurry.qubit"
    version = __version__
    author = "CalculQuébec"
    observables = {
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
    }  # Update with supported observables

    # Define the supported operations by making a python set of the keys from the SNOWFLURRY_OPERATION_MAP dictionary
    # Ignore keys with value "NotImplementedError"
    operations = {
        key
        for key, value in SNOWFLURRY_OPERATION_MAP.items()
        if value != NotImplementedError
    }

    @property
    def num_wires(self):
        """Get the number of wires.

        Returns:
            int: The number of wires.
        """
        return len(self.wires)

    @property
    def name(self):
        """The name of the device."""
        return "snowflurry.qubit"

    def preprocess(
        self,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Tuple[TransformProgram, ExecutionConfig]:
        """This function defines the device transfrom program to be applied and an updated execution config.

        Args:
            execution_config (Union[ExecutionConfig, Sequence[ExecutionConfig]]): A data structure describing the
            parameters needed to fully describe the execution.

        Returns:
            TransformProgram: A transform program that when called returns QuantumTapes that the device
            can natively execute.
            ExecutionConfig: A configuration with unset specifications filled in.
        """
        config = execution_config

        transform_program = TransformProgram()

        transform_program.add_transform(
            decompose, stopping_condition=stopping_condition, name=self.name
        )

        return transform_program, config

    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Result_or_ResultBatch:
        """
        Execute a batch of quantum circuits or a single circuit on the device.

        Args:
            circuits (QuantumTape or Sequence[QuantumTape]): a single quantum circuit or a batch of quantum
            circuits to execute.
            execution_config (ExecutionConfig): a data structure describing the parameters needed to
            fully describe the execution.

        Returns:
            Result (tuple): a single result if a single circuit is executed, or a tuple of results if a batch of
            circuits is executed.
        """
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = [circuits]

        if self.tracker.active:
            for c in circuits:
                self.tracker.update(resources=c.specs["resources"])
            self.tracker.update(batches=1, executions=len(circuits))
            self.tracker.record()

        # Check if execution_config is an instance of ExecutionConfig
        if isinstance(execution_config, ExecutionConfig):
            interface = (
                execution_config.interface
                if execution_config.gradient_method in {"backprop", None}
                else None
            )
        else:
            # Fallback or default behavior if execution_config is not an instance of ExecutionConfig
            interface = None

        results = tuple(
            PennylaneConverter(
                circuit,
                debugger=self._debugger,
                interface=interface,
                host=self.host,
                user=self.user,
                access_token=self.access_token,
                project_id=self.project_id,
                realm=self.realm,
                wires=self.num_wires,
            ).simulate()
            for circuit in circuits
        )

        return results[0] if is_single_circuit else results


class MonarqDevice(qml.devices.Device):

    def __init__(self):
        super().__init__(wires=1, shots=1)

    @property
    def name(self):
        return "monarq.default"

    def execute(self, tape, **kwargs):
        return "you have accessed the Monarq device"