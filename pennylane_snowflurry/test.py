

if __name__ == "__main__":
    import numpy as np
    import pennylane as qml
    from test_circuits import circuit_qpe, bernstein_vazirani
    from pennylane_snowflurry.monarq_device import MonarqDevice
    from pennylane_snowflurry.test_device import TestDevice
    from pennylane_snowflurry.transpiler.debug_utility import to_qasm, SnowflurryUtility
    from pennylane_snowflurry.transpiler.monarq_transpile import get_transpiler
    from pennylane_snowflurry.transpiler.optimization_utility import are_matrices_equivalent
  
    class const:
        host = "https://manager.anyonlabs.com"
        user = "stage"
        access_token = "FjjIKjmDMoAMzSO4v2Bu62a+8vD39zib"
        realm = "calculqc"
        machine_name = "yamaska"
        project_id = ""
        circuit_name = "test_circuit"

    num_wires = 5
    dev = qml.device("default.qubit", wires=num_wires)
    
    print(qml.draw(bernstein_vazirani)(8))
    circuit = qml.QNode(bernstein_vazirani, dev)
    circuit(8)

    # regular_qasm = to_qasm(circuit.tape)
    # print(regular_qasm)
    # print("barrier q;\n")
    from pennylane_snowflurry.transpiler.multi_gate_decomposition import multiple_gate_decomposition
    tape = multiple_gate_decomposition(circuit.tape)
    snowflurryUtil = SnowflurryUtility(tape, const.host, const.user, const.access_token, const.realm)
    snowflurryUtil.transpile()
    snowflurry_qasm = snowflurryUtil.to_qasm()
    print(snowflurry_qasm)
    print("barrier q;\n")
    transpiled_pennylane = get_transpiler(baseDecomposition=False)(tape)[0][0]
    pennylane_qasm = to_qasm(transpiled_pennylane)
    print(pennylane_qasm)


