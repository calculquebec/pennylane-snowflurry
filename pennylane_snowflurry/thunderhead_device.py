from functools import partial
import numpy
import custom_gates as custom
from typing import Tuple
import pennylane as qml
from pennylane.devices import Device
from pennylane.transforms.core import TransformProgram
from pennylane.tape import QuantumScript, QuantumTape
from execution_config import DefaultExecutionConfig, ExecutionConfig
from pennylane import transform
import pennylane.transforms as transforms
from api_job import Job
from api_adapter import instructions
from custom_decomposition import thunderhead_decompose
from custom_optimization import thunderhead_optimize


class ThunderheadDevice(Device):
    name = "Thunderhead device"
    short_name = "thunderhead.qubit"
    pennylane_requires = ">=0.30.0"
    author = "CalculQuébec"
    
    realm = "calculqc"
    circuit_name = "test circuit"
    project_id = ""
    machine_name = "yamaska"
    
    operations = {
        key for key in instructions.keys()
    }
    
    observables = {
        "PauliZ"
    }
    
    def __init__(self, wires, shots, host = "", user = "", access_token = "") -> None:
        super().__init__(wires=wires, shots=shots)
        self.host = host
        self.user = user
        self.access_token = access_token
    
    @property
    def name(self):
        return ThunderheadDevice.short_name
    
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
        transform_program.add_transform(thunderhead_decompose)
        # transform_program.add_transform(thunderhead_optimize)

        return transform_program, config

    def execute(self, circuits: QuantumTape | list[QuantumTape], execution_config : ExecutionConfig = DefaultExecutionConfig):
        # circuits = [circuits]
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
            
        results = [Job(host=self.host, 
                       user=self.user, 
                       access_token=self.access_token, 
                       realm=ThunderheadDevice.realm)
                   .run(circ, 
                        ThunderheadDevice.circuit_name, 
                        ThunderheadDevice.project_id, 
                        ThunderheadDevice.machine_name) for circ in circuits]
        
        return results if not is_single_circuit else results[0]
    
if __name__ == "__main__":
    class const:
        host = "https://manager.anyonlabs.com"
        user = "stage"
        access_token = "FjjIKjmDMoAMzSO4v2Bu62a+8vD39zib"
        realm = "calculqc"
        machine_name = "yamaska"
        project_id = ""
        circuit_name = "test_circuit"
        
    from thunderhead_device import ThunderheadDevice
    import pennylane as qml

    dev = ThunderheadDevice(3, 1000, const.host, const.user, const.access_token)

    @qml.qnode(dev)
    def circuit():
        qml.X(0)
        qml.CNOT([0, 1])
        qml.CNOT([1, 2])
        return qml.counts(wires = [0, 1, 2])

    circuit()

