from typing import Union, Callable, Tuple, Optional, Sequence
import numpy as np
from pennylane import Device
import abc
import pennylane as qml
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.typing import Result, ResultBatch
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.transforms.core import TransformProgram
from pennylane_snowflurry.pennylane_converter import PennylaneConverter
from pennylane_snowflurry.execution_config import (
    ExecutionConfig,
    DefaultExecutionConfig,
)
from pennylane.typing import Result, ResultBatch

# The plugin does not support batching yet, but this is a placeholder for future implementation
Result_or_ResultBatch = Union[Result, ResultBatch]
QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[
    QuantumTape, QuantumTapeBatch
]  # type : either a single QuantumTape or a Sequence of QuantumTape


class SnowflurryQubitDevice(qml.devices.Device):
    """Snowflurry Qubit PennyLane device for interfacing with Anyon's quantum computers.

    Extends the PennyLane :class:`~.pennylane.Device` class.
    """

    def __init__(
        self,
        wires=None,
        shots=None,
        seed="global",
        max_workers=None,
        host="",
        user="",
        access_token="",
    ) -> None:
        super().__init__(wires=wires, shots=shots)
        self._max_workers = max_workers

        seed = np.random.randint(0, high=10000000) if seed == "global" else seed
        self._rng = np.random.default_rng(seed)
        self.host = host
        self.user = user
        self.access_token = access_token
        self._debugger = None

    pennylane_requires = ">=0.27.0"

    name = "Snowflurry Qubit Device"
    short_name = "snowflurry.qubit"
    version = "1.0"
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
