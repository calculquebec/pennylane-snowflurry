#from snowflurry_device import SnowflurryQubitDevice
from pennylane_snowflurry import SnowflurryQubitDevice
from julia import Snowflurry
from julia import Base
from julia import Main
import julia
import pennylane as qml
import unittest
import numpy as np
from pennylane.tape import QuantumScript


class Test_TestSnowflurryPennylageIntegration(unittest.TestCase):
    def test_basic_julia(self):
        c = Snowflurry.QuantumCircuit(qubit_count=3)
        print(c)
        dev_def = qml.device("snowflurry.qubit", wires=3)
        self.assertEqual(True,True)

    def test_gate_hadamard():
        c = Snowflurry.QuantumCircuit(qubit_count=3)
        j = julia.julia()
        j.eval("jl.eval('push!(c,hadamard(1))')")
    def test_gate_PauliX():
        c = Snowflurry.QuantumCircuit(qubit_count=3)
        j = julia.julia()
        j.eval("jl.eval('push!(c,hadamard(1))')")
    def test_gate_PauliZ():
        c = Snowflurry.QuantumCircuit(qubit_count=3, gates=[sigma_x(1)])
        j = julia.julia()
        j.eval("jl.eval('push!(c,hadamard(1))')")

if __name__ == '__main__':
    #julia.install()
    #unittest.main()
    #dev_def = qml.device("snowflurry.qubit", wires=1)
    dev = SnowflurryQubitDevice(wires=1)
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
    c = Main.eval("""
                using Snowflurry
                QuantumCircuit(qubit_count=3, gates=[sigma_x(1)])
                """)
    print(c)
    # ops = [qml.BasisState(np.array([1,1]), wires=(0,"a")),
    #     qml.RX(0.432, 0),
    #     qml.RY(0.543, 0),
    #     qml.CNOT((0,"a")),
    #     qml.RX(0.133, "a")]

    # qscript = QuantumScript(ops, [qml.expval(qml.PauliZ(0))])
    # for element in qscript:
    #     print(type(element))
    #     print(element)


