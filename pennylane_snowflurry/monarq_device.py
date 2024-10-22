import json
from typing import Tuple
from pennylane.devices import Device
from pennylane.transforms.core import TransformProgram
from pennylane.tape import QuantumScript, QuantumTape
from pennylane_snowflurry.execution_config import DefaultExecutionConfig, ExecutionConfig
from pennylane_snowflurry.api_job import Job
from pennylane_snowflurry.api_adapter import ApiAdapter
from pennylane_snowflurry.transpiler.monarq_transpile import get_transpiler
from functools import partial
import pennylane_snowflurry.monarq_connectivity as con

class MonarqDevice(Device):
    """
    a device created for sending job on Calcul Quebec's MonarQ quantum computer
    """

    benchmark_acceptance=0.7

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
                 host, 
                 user, 
                 access_token, 
                 baseDecomposition=True, 
                 placeAndRoute=True, 
                 optimization=True, 
                 nativeDecomposition=True) -> None:

        super().__init__(wires=wires, shots=shots)
        self.host = host
        self.user = user
        self.access_token = access_token
        self._baseDecomposition = baseDecomposition
        self._placeAndRoute = placeAndRoute
        self._optimization = optimization, 
        self._nativeDecomposition = nativeDecomposition

        self._build_benchmark()
    
    def _build_benchmark(self):
        couplers_tag = "couplers"
        qubits_tag = "qubits"
        readoutState1Fidelity_tag = "readoutState1Fidelity"
        czGateFidelity_tag = "czGateFidelity"

        api = ApiAdapter(self.host, self.user, self.access_token, MonarqDevice.realm)
        qubits_and_couplers = api.get_qubits_and_couplers(self.machine_name)
        connectivity = con.connectivity
        self.benchmark = { qubits_tag : [], couplers_tag : [] }

        for coupler_id in qubits_and_couplers[couplers_tag]:
            benchmark_coupler = qubits_and_couplers[couplers_tag][coupler_id]
            conn_coupler = connectivity[couplers_tag][coupler_id]

            if benchmark_coupler[czGateFidelity_tag] >= MonarqDevice.benchmark_acceptance:
                continue

            self.benchmark[couplers_tag].append(conn_coupler)

        for qubit_id in qubits_and_couplers[qubits_tag]:
            benchmark_qubit = qubits_and_couplers[qubits_tag][qubit_id]

            if benchmark_qubit[readoutState1Fidelity_tag] >= MonarqDevice.benchmark_acceptance:
                continue

            self.benchmark[qubits_tag].append(int(qubit_id))

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
        transform_program.add_transform(get_transpiler(baseDecomposition=self._baseDecomposition, 
                                                placeAndRoute = self._placeAndRoute, 
                                                optimization = self._optimization, 
                                                nativeDecomposition = self._nativeDecomposition,
                                                benchmark=self.benchmark))
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
                       realm=MonarqDevice.realm)
                   .run(circ, 
                        MonarqDevice.circuit_name, 
                        MonarqDevice.project_id, 
                        MonarqDevice.machine_name) for circ in circuits]
        
        return results if not is_single_circuit else results[0]

