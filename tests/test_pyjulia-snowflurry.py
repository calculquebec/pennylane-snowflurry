import pennylane_snowflurry
from julia import Snowflurry
from julia import Base
from julia import Main
import julia
import pennylane_recent as qml
import unittest

class Test_TestSnowflurryPennylageIntegration(unittest.TestCase):
    def test_basic_julia(self):
        c = Snowflurry.QuantumCircuit(qubit_count=3)
        print(c)
        dev_def = qml.device("snowflurry.qubit", wires=3)
        self.assertEqual(True,True)

    def test_gate_hadamard():
        c = Snowflurry.QuantumCircuit(qubit_count=3)
        Main.eval("jl.eval('push!(c,hadamard(1))')")

    def test_gate_PauliX():
        c = Snowflurry.QuantumCircuit(qubit_count=3)
        Main.eval("jl.eval('push!(c,sigma_x(1))')")

    def test_gate_PauliZ():
        c = Snowflurry.QuantumCircuit(qubit_count=3)
        Main.eval("jl.eval('push!(c,sigma_z(1))')")

if __name__ == '__main__':
    #unittest.main()
    dev_def = qml.device("snowflurry.qubit", wires=1)
    #make quantumtape with rx
    #enumerate gates
    #execute rx gate
    with qml.tape.QuantumTape() as tape:
        qml.RX(0.432, wires=0)
        qml.RY(0.543, wires=0)
        qml.CNOT(wires=[0, 'a'])
        qml.RX(0.133, wires='a')
        qml.expval(qml.PauliZ(wires=[0]))
        print(tape.circuit)

