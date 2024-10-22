from functools import partial
from pennylane.math import quantum
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operation
from pennylane.tape import QuantumTape
import pennylane as qml
import numpy as np
import custom_gates as custom
from pennylane_snowflurry.pennylane_converter import PennylaneConverter, Snowflurry
import matplotlib.pyplot as plt

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

    def to_pennylane(self, sf_circuit = None, measurement_process = qml.probs) -> QuantumTape:
        """
        turns a snowflurry circuit into a Pennylane QuantumTape
        """
        if sf_circuit is None: sf_circuit = Snowflurry.sf_circuit

        new_operations = []
        measurements = {}

        assoc = {"Snowflurry.Z90" : custom.Z90, "Snowflurry.ZM90" : custom.ZM90, 
                 "Snowflurry.X90" : custom.X90, "Snowflurry.XM90" : custom.XM90, 
                 "Snowflurry.Y90" : custom.Y90, "Snowflurry.YM90" : custom.YM90,
                 "Snowflurry.SigmaX" : qml.PauliX, "Snowflurry.SigmaY" : qml.PauliY, "Snowflurry.SigmaZ" : qml.PauliZ, 
                 "Snowflurry.Pi8" : qml.T, "Snowflurry.Pi8Dagger" : custom.TDagger, "Snowflurry.ControlZ" : qml.CZ}

        
        for i in sf_circuit.instructions:
            if "Readout" in str(i):
                qubit = i.connected_qubit
                bit = i.destination_bit
                measurements[int(bit) - 1] = int(qubit) - 1
                continue

            symbol_and_param = i.symbol
            symbol_and_param = str(symbol_and_param).split("(")
            
            symbol = symbol_and_param[0]
            param = symbol_and_param[1].split(")")[0] if len(symbol_and_param) > 1 else None
            connected_qubits = i.connected_qubits

            op = partial(qml.PhaseShift, float(param)) if "PhaseShift" in symbol else assoc[symbol]
            new_operations += [op([int(w) - 1 for w in connected_qubits])]

        measurements = list(sorted(measurements.values(), key=lambda v: [m for m in measurements.items() if m[1] == v][0]))
        measurements = [measurement_process(wires=measurements)]
        return QuantumTape(ops = new_operations, measurements = measurements, shots = 1000)

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


def arbitrary_circuit(tape : QuantumTape, measurement = qml.probs):
    """
    create a quantum function out of a tape and a default measurement to use (overrides the measurements in the qtape)
    """
    def _arbitrary_circuit(operations : list[Operation], measurements : list[MeasurementProcess]):
        for op in operations:
            if len(op.parameters) > 0:
                type(op)(op.parameters, op.wires)
            else:
                type(op)(wires=op.wires)
        
        def get_wires(mp : MeasurementProcess):
            return [w for w in mp.wires] if mp is not None and mp.wires is not None and len(mp.wires) > 0 else tape.wires

        # retourner une liste de mesures si on a plusieurs mesures, sinon retourner une seule mesure
        return [measurement(wires=get_wires(meas)) for meas in measurements] if len(measurements) > 1 \
            else measurement(wires=get_wires(measurements[0] if len(measurements) > 0 else None))
    return _arbitrary_circuit(tape.operations, tape.measurements)

def get_labels(up_to : int):
    num = int(np.log2(up_to)) + 1
    return [format(i, f"0{num}b") for i in range(up_to + 1)]

def bar_plot(labels, *values):
    
    for value in values:
        if len(labels) != len(value):
            raise Exception("all columns should have the same number of lines")
    
    plot_infos = zip(labels, values)

    x = np.arange(len(labels))
    width = 1/len(labels)
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained")

    for attribute, measurement in plot_infos:
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label = attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1
    
    ax.set_ylabel('Counts')
    ax.set_title('Measure comparison between different environments')
    ax.set_xticks(x + width, labels)
    ax.set_ylim(0, 250)
    plt.show()