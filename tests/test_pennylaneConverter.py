import pennylane_snowflurry
import pennylane as qml
import numpy as np
import logging
from pennylane.tape import QuantumTape
import unittest

class Test_PennylaneConverterClass(unittest.TestCase):
    def test_quantumTape(self):
        ops = [qml.BasisState(np.array([1,1]), wires=(0,"a"))]
        quantumTape = QuantumTape(ops, [qml.expval(qml.PauliZ(0))])
        converter = pennylane_snowflurry.PennylaneConverter(quantumTape)
        self.assertIsInstance(converter.pennylane_circuit, QuantumTape)
    

if __name__ == '__main__':
    unittest.main()