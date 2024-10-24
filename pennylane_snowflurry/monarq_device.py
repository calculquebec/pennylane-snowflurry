from typing import Tuple
from pennylane.devices import Device
from pennylane.transforms.core import TransformProgram
from pennylane.tape import QuantumScript, QuantumTape
from pennylane_snowflurry.execution_config import DefaultExecutionConfig, ExecutionConfig
from pennylane_snowflurry.api_job import Job
from pennylane_snowflurry.transpiler.monarq_transpile import get_transpiler

class MonarqDevice(Device):
    """PennyLane device for interfacing with Anyon's quantum Hardware.

    * Extends the PennyLane :class:`~.pennylane.Device` class.
    * Batching is not supported yet.

    Args:
        wires (int, Iterable[Number, str]): Number of wires present on the device, or iterable that
            contains unique labels for the wires as numbers (i.e., ``[-1, 0, 2]``) or strings
            (``['ancilla', 'q1', 'q2']``). Default ``None`` if not specified.
        shots (int, Sequence[int], Sequence[Union[int, Sequence[int]]]): The default number of shots
            to use in executions involving this device.
        host (str): URL of the QPU server.
        user (str): Username.
        access_token (str): User access token.
    """

    name = "MonarQDevice"
    short_name = "calculqc.qubit"
    pennylane_requires = ">=0.30.0"
    author = "CalculQuebec"
    
    realm = "calculqc"
    circuit_name = "test circuit"
    project_id = ""
    machine_name = "yamaska"

    observables = {
        "PauliZ"
    }
    
    def __init__(self, 
                 wires, 
                 shots,  
                 baseDecomposition=True, 
                 placeAndRoute=True, 
                 optimization=True, 
                 nativeDecomposition=True, 
                 use_benchmarking=True) -> None:

        super().__init__(wires=wires, shots=shots)
        self._baseDecomposition = baseDecomposition
        self._placeAndRoute = placeAndRoute
        self._optimization = optimization, 
        self._nativeDecomposition = nativeDecomposition 
        self._use_benchmarking = use_benchmarking
    

    @property
    def name(self):
        return MonarqDevice.short_name
    
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
        transform_program.add_transform(get_transpiler(self._baseDecomposition, 
                                                self._placeAndRoute, 
                                                self._optimization, 
                                                self._nativeDecomposition, 
                                                self._use_benchmarking))
        return transform_program, config

    def execute(self, circuits: QuantumTape | list[QuantumTape], execution_config : ExecutionConfig = DefaultExecutionConfig):
        """
        This function runs provided quantum circuit on MonarQ
        A job is first created, and then ran. Results are returned to the user.
        """
        is_single_circuit : bool = isinstance(circuits, QuantumScript)
        if is_single_circuit:
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
            
        results = [Job(circ).run() for circ in circuits]
        
        return results if not is_single_circuit else results[0]

