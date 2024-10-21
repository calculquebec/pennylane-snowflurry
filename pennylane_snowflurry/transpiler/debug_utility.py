from pennylane.math import quantum
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operation
from pennylane.tape import QuantumTape
import pennylane as qml

from pennylane_snowflurry.pennylane_converter import PennylaneConverter, Snowflurry

def to_qasm(tape : QuantumTape) -> str:
    eq = {
        "PauliX" : "x", "PauliY" : "y", "PauliZ" : "z", "Identity" : "id",
        "RX" : "rx", "RY" : "ry", "RZ" : "rz", "PhaseShift" : "p", "Hadamard" : "h",
        "S" : "s", "Adjoint(S)" : "sdg", "SX" : "sx", "Adjoint(SX)" : "sxdg", "T" : "t", "Adjoint(T)" : "tdg", 
        "CNOT" : "cx", "CY" : "cy", "CZ" : "cz", "SWAP" : "swap",
        "Z90" : "s", "ZM90" : "sdg", "X90" : "sx", "XM90" : "sxdg", "Y90" : "ry(pi/2)", "YM90" : "ry(3*pi/2)",
        "TDagger" : "tdg"
    }
    return "\n".join([eq[op.name] \
        + (f"({op.parameters[0]}) " if len(op.parameters) > 0 else " ") \
        + " ".join([f"q[{w}]" for w in op.wires]) \
        + ";" for op in tape.operations])

class SnowflurryUtility:
    def __init__(self, tape : QuantumTape, host, user, access_token, realm):
        self.converter = PennylaneConverter(tape, 
                                       wires=24,
                                       host=host, 
                                       user=user, 
                                       access_token=access_token, 
                                       realm=realm, 
                                       project_id="default")
    
    def gate_count(self, sf_circuit = None) -> int:
        if sf_circuit is None: sf_circuit = Snowflurry.sf_circuit
        return len(sf_circuit.instructions)

    def to_qasm(self, sf_circuit = None) -> str:
        if sf_circuit is None: sf_circuit = Snowflurry.sf_circuit

        result = []
        
        assoc = {"Snowflurry.Z90" : "s", "Snowflurry.ZM90" : "sdg", 
                 "Snowflurry.X90" : "sx", "Snowflurry.XM90" : "sxdg", 
                 "Snowflurry.Y90" : "ry(pi/2)", "Snowflurry.YM90" : "ry(3*pi/2)",
                 "Snowflurry.SigmaX" : "x", "Snowflurry.SigmaY" : "y", "Snowflurry.SigmaZ" : "z", 
                 "Snowflurry.Pi8" : "t", "Snowflurry.Pi8Dagger" : "tdg", "Snowflurry.ControlZ" : "cz"}

        for i in sf_circuit.instructions:
            if "Readout" in str(i):
                continue
            symbol_and_param = i.symbol
            symbol_and_param = str(symbol_and_param).split("(")
            symbol = symbol_and_param[0]
            param = symbol_and_param[1].split(")")[0] if len(symbol_and_param) > 1 else None
            connected_qubits = i.connected_qubits
            connected_qubits = str(connected_qubits)
            result.append(f"{assoc[symbol]}{"(" + param + ") " if param != "" else " "} {connected_qubits}")
        return "\n".join(result)

    def transpile(self):
        sf_circuit = self.converter.convert_circuit(self.converter.pennylane_circuit)
        [[self.converter.apply_single_readout(w) for w in mp.wires] for mp in self.converter.pennylane_circuit.measurements]

        qpu = Snowflurry.AnyonYamaskaQPU(Snowflurry.currentClient, Snowflurry.seval("project_id"))
        sf_circuit = Snowflurry.transpile(Snowflurry.get_transpiler(qpu), sf_circuit)
        Snowflurry.sf_circuit = sf_circuit

def arbitrary_circuit(tape : QuantumTape):
    def _arbitrary_circuit(operations : list[Operation], measurements : list[MeasurementProcess]):
        [type(op)(op.parameters, op.wires) if len(op.parameters) > 0 else type(op)(op.wires) for op in operations]
        return [type(mp)(wires=[w for w in mp.wires]) for mp in measurements] if len(measurements) > 1 \
            else type(measurements[0])(wires=[w for w in measurements[0].wires] if len(measurements[0].wires) > 0 else tape.wires)
    return _arbitrary_circuit(tape.operations, tape.measurements)
