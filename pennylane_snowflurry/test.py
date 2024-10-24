from functools import partial
from pennylane_snowflurry.utility.debug_utility import arbitrary_circuit
import numpy as np
import pennylane as qml
import test_circuits
from pennylane_snowflurry.monarq_device import MonarqDevice
from pennylane_snowflurry.test_device import TestDevice
from pennylane_snowflurry.snowflurry_device import SnowflurryQubitDevice
from pennylane_snowflurry.utility.debug_utility import SnowflurryUtility, get_labels
from pennylane_snowflurry.transpiler.monarq_transpile import get_transpiler
from pennylane_snowflurry.transpiler.base_decomposition import base_decomposition

if __name__ == "__main__":
    
    class const:
        realm = "calculqc"
        machine_name = "yamaska"
        project_id = "default"
        circuit_name = "test_circuit"

    num_qubits = 11

    dev = TestDevice()
    dev = qml.device("default.qubit", shots = 1000)
    dev = MonarqDevice(num_qubits, 1000)
    dev = SnowflurryQubitDevice(num_qubits, 1000, "global", const.host, const.user, const.access_token, const.project_id, const.realm)

    def prepare(circuit, dev, regular = True, snowflurry = True, calcul_quebec = True, measurement = qml.probs):
        """
        prepares 3 qnodes :
        1. without any transpilation
        2. with snowflurry transpilation
        3. with calcul quebec transpilation
        overrides measurements with given one
        """

        qnodes = []
        
        circuit()
        tape = base_decomposition(circuit.tape)
        
        if regular:
            qnode = qml.QNode(lambda : arbitrary_circuit(tape, measurement), dev)
            qnodes.append(qnode)

        if snowflurry:
            snowflurryUtil = SnowflurryUtility(tape, const.host, const.user, const.access_token, const.realm)
            snowflurryUtil.transpile()
            sf_tape = snowflurryUtil.to_pennylane()
            sf_qnode = qml.QNode(lambda: arbitrary_circuit(sf_tape, measurement), dev)
            qnodes.append(sf_qnode)

        if calcul_quebec:
            transpiler = get_transpiler()
            optimized_tape = transpiler(circuit.tape)[0][0]
            opt_qnode = qml.QNode(lambda : arbitrary_circuit(optimized_tape, measurement), dev)
            qnodes.append(opt_qnode)

        return qnodes

    def counts_is_same(results1, results2, acceptance_criteria = 10):
        """
        checks if counts are the same for two result dictionaries
        """
        labels = get_labels(2 ** len(results1.items()[0][0]))
        for i in labels:
            if i in results1 and i not in results2:
                return False
            if i not in results1 and i in results2:
                return False
            if np.abs(results1[i] - results2[i]) > acceptance_criteria:
                return False
        return True

    def probs_is_same(results1, results2, acceptance_criteria = 1E-5):
        """
        checks if probabilities are the same for two result arrays
        """
        for i, r1 in enumerate(results1):
            r2 = results2[i]
            if np.abs(r1 - r2) > acceptance_criteria:
                return False
        return True

    def test_veracity(circuit):
        qnodes = prepare(circuit, dev, True, False, True)
        reg_qnode = qnodes[0]
        cq_qnode = qnodes[1]

        reg_results = reg_qnode()
        cq_results = cq_qnode()

        return probs_is_same(reg_results, cq_results)

    def test_efficiency(circuit):
        qnodes = prepare(circuit, dev, False, True, True, measurement=qml.counts)

        sf_qnode = qnodes[0]
        cq_qnode = qnodes[1]

        sf_depth = qml.specs(sf_qnode)()["resources"].depth
        cq_depth = qml.specs(cq_qnode)()["resources"].depth

        print(f"snowflurry depth : {sf_depth}")
        print(f"calcul quebec depth : {cq_depth}")

    
    partial_circuit = partial(test_circuits.GHZ, 4)
    circuit = qml.QNode(partial_circuit, dev)

    print(circuit())
    exit()

    for i in range(3, 16):
        partial_circuit = partial(test_circuits.GHZ, i)
        circuit = qml.QNode(partial_circuit, dev)

        print(f"GHZ with {i} qubits : ")
        test_efficiency(circuit)
        print("\n")
