import pennylane_snowflurry
import pennylane as qml
import numpy as np
import logging
from pennylane.tape import QuantumScript
import unittest

class Test_PennylaneClass(unittest.TestCase):
    def test_gate(self):
        ops = [qml.BasisState(np.array([1,1]), wires=(0,"a")),
        qml.RX(0.432, 0),
        qml.RY(0.543, 0),
        qml.CNOT((0,"a")),
        qml.RX(0.133, "a")]
        qscript = QuantumScript(ops, [qml.expval(qml.PauliZ(0))])
        converter = pennylane_snowflurry.PennylaneConverter(qscript)
        converter.getGateFromCircuit(qscript)
        self.assertEqual(ops,qscript)
    

if __name__ == '__main__':
    unittest.main()