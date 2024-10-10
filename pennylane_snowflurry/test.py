import pennylane as qml
import numpy as np
from pennylane.tape import QuantumTape
from pennylane_snowflurry.test_device import TestDevice
import pennylane.transforms as transforms
from optimization_utility import to_qasm
from pennylane_snowflurry.transpiler.virtual_optimization import remove_leaf_zs, remove_root_zs

 
if __name__ == "__main__":
    import pennylane as qml
    from test_circuits import circuit_qpe

    num_wires = 5
    dev2 : qml.Device = TestDevice(num_wires)
    
    def circuit():
        import random
        [qml.Hadamard(i) for i in range(num_wires)]
        qml.Z(num_wires-1)
        
        # Uf
        for i in range(num_wires - 1):
            qml.CNOT([i, num_wires - 1])
        

        [qml.Hadamard(i) for i in range(num_wires - 1)]
        return qml.probs(wires=[i for i in range(num_wires - 1)])
    
    print(qml.draw(circuit_qpe)(num_wires))
    result2 = qml.QNode(circuit_qpe, dev2)(num_wires)
    print(list([float(i) for i in result2]))

