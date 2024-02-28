from typing import Union, Callable, Tuple, Optional, Sequence
import numpy as np
from pennylane import Device
import abc
import pennylane as qml
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.typing import Result, ResultBatch
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.transforms.core import TransformProgram
from pennylane.devices.preprocess import decompose
from pennylane_snowflurry.pennylane_converter import PennylaneConverter
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
    """Specify whether or not an Operator object is supported by the device."""
    if op.name == "QFT" and len(op.wires) >= 6:
        return False
    if op.name == "GroverOperator" and len(op.wires) >= 13:
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

    """# host, user, access_token, project_id would ideally be keyword args

    def __init__(
        self,
        wires=None,
        shots=None,
        seed="global",
        max_workers=None,
        host="",
        user="",
        access_token="",
        project_id="",
    ) -> None:
        super().__init__(wires=wires, shots=shots)
        self._max_workers = max_workers

        seed = np.random.randint(0, high=10000000) if seed == "global" else seed
        self._rng = np.random.default_rng(seed)
        self.host = host
        self.user = user
        self.access_token = access_token
        self.project_id = project_id
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
    operations = {
        "CNOT",
        "Hadamard",
        "RX",
        "RY",
        "RZ",
        "PauliX",
        "PauliY",
        "PauliZ",
        "PhaseShift",
        "CNOT",
        "CZ",
        "SWAP",
        "ISWAP",
        "Identity",
        "PhaseShift",
        "Toffoli",
        "U3",
        "T",
        "Rot",
    }  # Update with supported operations

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
            can natively execute as well as a postprocessing function to be called after execution.
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
            max_workers = execution_config.device_options.get(
                "max_workers", self._max_workers
            )
            interface = (
                execution_config.interface
                if execution_config.gradient_method in {"backprop", None}
                else None
            )
        else:
            # Fallback or default behavior if execution_config is not an instance of ExecutionConfig
            max_workers = self._max_workers
            interface = None

        if max_workers is None:
            results = tuple(
                PennylaneConverter(
                    c,
                    rng=self._rng,
                    debugger=self._debugger,
                    interface=interface,
                    host=self.host,
                    user=self.user,
                    access_token=self.access_token,
                    project_id=self.project_id,
                ).simulate()
                for c in circuits
            )
        else:
            vanilla_circuits = [convert_to_numpy_parameters(c) for c in circuits]
            seeds = self._rng.integers(2**31 - 1, size=len(vanilla_circuits))
            _wrap_simulate = partial(simulate, debugger=None, interface=interface)
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
            ) as executor:
                exec_map = executor.map(
                    _wrap_simulate,
                    vanilla_circuits,
                    seeds,
                    [self._prng_key] * len(vanilla_circuits),
                )
                results = tuple(exec_map)

            # reset _rng to mimic serial behavior
            self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))

        return results[0] if is_single_circuit else results
