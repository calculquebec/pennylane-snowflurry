from pennylane.tape import QuantumTape
from pennylane.operation import Operation
from pennylane.measurements import MeasurementProcess
from dotenv import dotenv_values
import requests
import json

class ApiUtility:
    @staticmethod
    def convert_instruction(instruction : Operation) -> dict[str, any]:
        """converts a Pennylane operation to a dictionary that can be read by the Thunderhead API

        Args:
            instruction (Operation): a Pennylane operation (a gate)
            actual_qubits (list[int]): the mapping from the wires in pennylane to the wires in the physical machine

        Returns:
            dict[str, any]: a dictionary representation of the operation that can be read by the Thunderhead API
        """
        operation = {
            ApiUtility.keys.qubits : [w for w in instruction.wires],
            ApiUtility.keys.type : instructions[instruction.name]
        }
        if instruction.name == "PhaseShift": operation[ApiUtility.keys.parameters] = {"lambda" : instruction.parameters[0]}
            
        return operation
    
    @staticmethod
    def convert_circuit(circuit : QuantumTape) -> dict[str, any]:
        """converts a pennylane quantum script to a dictionary that can be read by the Thunderhead API

        Args:
            tape (tape.QuantumScript): a pennylane quantum script (with informations about the number of wires, the operations and the measurements)

        Returns:
            dict[str, any]: a dictionary representation of the circuit that can be read by the API
        """

        circuit_dict = {
            ApiUtility.keys.bitCount : 24,
            ApiUtility.keys.operations : [ApiUtility.convert_instruction(op) for op in circuit.operations if not isinstance(op, MeasurementProcess)],
            ApiUtility.keys.qubitCount : 24
        }
        for m in circuit.measurements:
            wires = m.wires if len(m.wires) > 0 else [_ for _ in range(len(circuit.wires))]
            for i, w in enumerate(wires):
                if i in [w2[ApiUtility.keys.qubits] for w2 in circuit_dict[ApiUtility.keys.operations]]:
                    continue
                circuit_dict[ApiUtility.keys.operations].append({
                    ApiUtility.keys.qubits : [circuit.wires[i]],
                    ApiUtility.keys.bits : [i],
                    ApiUtility.keys.type : "readout"
                })
        return circuit_dict
    
    @staticmethod
    def basic_auth(username : str, password : str) -> str:
        """create a basic authentication token from a Thunderhead username and access token

        Args:
            username (str): your Thunderhead username
            password (str): your Thunderhead access token

        Returns:
            str: the basic authentification string that will authenticate you with the API
        """
        from base64 import b64encode
        token = b64encode(f"{username}:{password}".encode('ascii')).decode("ascii")
        return f'Basic {token}'
    
    @staticmethod
    def headers(username : str, password : str, realm : str) -> dict[str, any]:
        """the Thunderhead API headers

        Args:
            username (str): your Thunderhead username
            password (str): your Thunderhead access token
            realm (str): your organization identifier with Thunderhead

        Returns:
            dict[str, any]: a dictionary representing the request headers
        """
        return {
            "Authorization" : ApiUtility.basic_auth(username, password),
            "Content-Type" : "application/json",
            "X-Realm" : realm
        }
    
    @staticmethod
    def job_body(circuit : dict[str, any], circuit_name : str, project_id : str, machine_name : str, shots) -> dict[str, any]:
        """the body for the job creation request

        Args:
            circuit (tape.QuantumScript): the script you want to convert
            name (str): the name of your job
            project_id (str): the id for the project for which this job will be run
            machine_name (str): the name of the machine on which this job will be run
            shots (int, optional): the number of shots (-1 will use the circuit's shot number)

        Returns:
            dict[str, any]: the body for the job creation request
        """
        body = {
            ApiUtility.keys.name : circuit_name,
            ApiUtility.keys.projectID : project_id,
            ApiUtility.keys.machineName : machine_name,
            ApiUtility.keys.shotCount : shots,
            ApiUtility.keys.circuit : circuit,
        }
        return body

    class routes:
        jobs = "/jobs"
        projects = "/projects"
        machines = "/machines"
        benchmarking = "/benchmarking"
        machineName = "?machineName"
        
    class keys:
        bitCount = "bitCount"
        qubitCount = "qubitCount"
        operations = "operations"
        circuit = "circuit"
        name = "name"
        machineName = "machineName"
        projectID = "projectID"
        shotCount = "shotCount"
        type = "type"
        bits = "bits"
        qubits = "qubits"
        parameters = "parameters"
        couplers = "couplers"
        readoutState1Fidelity = "readoutState1Fidelity"
        czGateFidelity = "czGateFidelity"
        resultsPerDevice = "resultsPerDevice"
        items = "items"
        id = "id"


instructions : dict[str, str] = {
    "Identity" : "i",
    "PauliX" : "x",
    "X90" : "x_90",
    "XM90" : "x_minus_90",
    "PauliY" : "y",
    "Y90" : "y_90",
    "YM90" : "y_minus_90",
    "PauliZ" : "z",
    "Z90" : "z_90",
    "ZM90" : "z_minus_90",
    "T" : "t",
    "TDagger" : "t_dag",
    "CZ" : "cz",
    "PhaseShift" : "p",
    "Hadamard" : "h",
    "CNOT" : "cnot"
}