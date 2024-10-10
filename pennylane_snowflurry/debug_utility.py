import numpy as np
from pennylane.tape import QuantumTape


def to_qasm(tape : QuantumTape, numbered = True):
    """
    turns a quantum circuit into a qasm circuit, readable by quantum composer
    TODO : add headers and readouts
    """
    def get_assoc(op_name):
        assoc = {
        "Hadamard":"h", "PauliX":"x", "PauliY":"y", "PauliZ":"z", 
        "CNOT":"cx", "CY" : "cz", "CZ":"cz", "SWAP":"swap",
        "S":"s", "Adjoint(S)":"sdg", "SX":"sx", "Adjoint(SX)":"sxdg", "T":"t", "Adjoint(T)":"tdg",
        "RZ":"rz", "RY":"ry", "RX":"rx", "Identity":"id", 
        "X90":"sx", "XM90":"sxdg", "Z90":"s", "ZM90":"sdg", "TDagger":"tdg", "PhaseShift":"p"}
        if op_name in assoc:
            return assoc[op_name]
        else:
            raise Exception(f"operation name {op_name} not in assoc")
    return "\n".join([(str(i) + "\t:\t" if numbered else "") + get_assoc(op.name) + (f"({op.parameters[0]}) " if len(op.parameters) > 0 else " ") + ",".join([f"q[{w}]" for w in op.wires]) + ";" for i, op in enumerate(tape.operations)])

def qasm_to_pennylane(code : str):
    """
    turns a qasm string into a pennylane readable circuit string
    TODO : many gates not supported. works for x, y, z, rx, ry, rz, cz, s, sx
    """
    mapping = {
        "x" : "PauliX", "y" : "PauliY", "z" : "PauliZ", "id":"Identity",
        "rx" : "RX", "ry" : "RY", "rz" : "RZ", "p" : "PhaseShift",
        "h" : "Hadamard", "s" : "S", "sx" : "SX", "t" : "T", 
        "sdg" : "adjoint(S)", "sxdg" : "adjoint(SX)", "tdg" : "adjoint(T)",
        "cx" : "CNOT", "cy" : "CY", "cz" : "CZ", "swap" : "SWAP"
    }
    values = code.split("\n")
    result = ""
    for value in values:
        value = value.strip()
        op = value[:2]
        param = "" if len(value.split("(")) == 1 else f"{value.split("(")[1].split(")")[0]}, "
        wires = f"[{value.split("[")[1].split("]")[0]}]" if len(value.split("[")) == 2 else f"[{int(value.split("[")[1].split("]")[0])}, {int(value.split("[")[2].split("]")[0])}]"
        result += f"qml.{mapping[op]}({param}wires={wires})\n"
    return result

def save_circuit_to_file(tape : QuantumTape):
    """
    debug function that saves a ZX_DAG's circuit representation to a file
    deprecated : doesn't work in current state. shouldn't be used
    """
    from pennylane.transforms import commutation_dag, CommutationDAG
    import networkx as nx

    dag : CommutationDAG = commutation_dag(tape)

    strs = ["" for i in dag.tape.wires]
    generations = nx.topological_generations(dag.graph)
    for gen in generations:
        for w in dag.tape.wires:
            if not any([w in dag.nodes[m].wires for m in gen]):
                strs[w] += "-"
        for i in gen:
            op = dag.nodes[i]
                
            for w in op.wires:
                if op.name == "CZ":
                    strs[w] += "c"
                elif op.name == "RZ":
                    if op.parameters[0] > np.pi/2 - 1: 
                        strs[w] += "Z"
                    else:
                        strs[w] += "z"
                elif op.name == "RX":
                    if op.parameters[0] > np.pi/2 - 1: 
                        strs[w] += "X"
                    else:
                        strs[w] += "x"
    strs = "\n".join(strs)
    with open("test.txt", "a") as myfile:
        myfile.write(strs + "\n\n")
