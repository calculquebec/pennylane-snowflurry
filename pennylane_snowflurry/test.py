from functools import partial
from pennylane_snowflurry.transpiler.debug_utility import arbitrary_circuit
import numpy as np
import pennylane as qml
import test_circuits
from pennylane_snowflurry.monarq_device import MonarqDevice
from pennylane_snowflurry.test_device import TestDevice
from pennylane_snowflurry.transpiler.debug_utility import to_qasm, SnowflurryUtility
from pennylane_snowflurry.transpiler.monarq_transpile import get_transpiler
from pennylane_snowflurry.transpiler.optimization_utility import are_matrices_equivalent
from pennylane_snowflurry.transpiler.multi_gate_decomposition import multiple_gate_decomposition
    

if __name__ == "__main__":
    
    dev = qml.device("default.qubit")

    class const:
        host = "https://manager.anyonlabs.com"
        user = "stage"
        access_token = "FjjIKjmDMoAMzSO4v2Bu62a+8vD39zib"
        realm = "calculqc"
        machine_name = "yamaska"
        project_id = ""
        circuit_name = "test_circuit"


    def test_efficiency(circuit):
        regular_results = circuit()
        tape = multiple_gate_decomposition(circuit.tape)
        # snowflurryUtil = SnowflurryUtility(tape, const.host, const.user, const.access_token, const.realm)
        # snowflurryUtil.transpile()

        transpiler = get_transpiler(True, True, False, False)
        optimized_tape = transpiler(circuit.tape)[0][0]

        qnode = qml.QNode(lambda : arbitrary_circuit(tape), dev)
        print(f"regular circuit : \n{qml.draw(qnode)()}")
        qnode = qml.QNode(lambda : arbitrary_circuit(optimized_tape), dev)
        print(f"optimized circuit : \n{qml.draw(qnode)()}")
        optimized_results = qnode()

        
        print(f"regular circuit probs : \n{regular_results}")
        print(f"optimized circuit probs : \n{optimized_results[0]}")
        # print(f"snowflurry transpiler gate count : {snowflurryUtil.gate_count()}")
        print(f"pennylane transpiler gate count : {str(len(optimized_tape.operations) + len(optimized_tape.measurements))}")
        
    @qml.qnode(dev)
    def connectivity_test1(num_wires = 7):
        wires = range(num_wires)
        for w in wires:
            qml.Hadamard(w)
        for w in wires[1:]:
            qml.CNOT(wires=[0, w])
        return qml.probs(wires=wires)
    
    partial_circuit = partial(test_circuits.circuit_qpe)
    circuit = qml.QNode(partial_circuit, dev)
    test_efficiency(circuit)
