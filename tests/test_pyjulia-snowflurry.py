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


class Test_TestSnowflurryPennylaneIntegration(unittest.TestCase):
    def test_basic_julia(self):
        c = Snowflurry.QuantumCircuit(qubit_count=3)
        dev_def = qml.device("snowflurry.qubit", wires=3)
        self.assertEqual(True,True)

    def test_execute_gate_hadamard(self):
        dev_snowflurry = qml.device("snowflurry.qubit", wires=1)
        dev_pennylane = qml.device("default.qubit", wires=1)

        tape_pennylane = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.counts(wires=0)],shots=1)
        tape_snowflurry = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.counts(wires=0)])

        self.assertEqual(type(dev_pennylane.execute(tape_pennylane)), type(dev_snowflurry.execute(tape_snowflurry)), "Hadamard gate, singular shot error")
        
        tape = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.counts(wires=0)], shots=50)
        results = dev_snowflurry.execute(tape)
        
        self.assertEqual(type(dev_pennylane.execute(tape)), type(results), "Hadamard gate, multiple shots errors")
        
        self.assertTrue(abs(results['0'] - results['1']) < 10, "Hadamard gate, statistical error")
        tape = qml.tape.QuantumTape([qml.Hadamard(0)],[qml.expval(qml.PauliZ(0))])
    
        self.assertEqual(dev_pennylane.execute(tape), dev_snowflurry.execute(tape))

        tape = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.expval(qml.PauliZ(0))])
        self.assertEqual(dev_pennylane.execute(tape), dev_snowflurry.execute(tape))



    def test_gate_PauliX(self):
        dev_snowflurry = qml.device("snowflurry.qubit", wires=1)
        tape = qml.tape.QuantumScript([qml.PauliX(0)], [qml.counts(wires=0)], shots=50)
        self.assertTrue(dev_snowflurry.execute(tape)['1'] == 50, "PauliX gate, statistical error") 

        @qml.qnode(dev_snowflurry)
        def snowflurry_circuit():
            qml.PauliX(0)
            return qml.expval(qml.PauliZ(0))
        
        dev_pennylane = qml.device("default.qubit", wires=1)
        @qml.qnode(dev_pennylane)
        def pennylane_circuit():
            qml.PauliX(0)
            return qml.expval(qml.PauliZ(0))
        self.assertEqual(snowflurry_circuit(), pennylane_circuit())

    def test_gate_PauliZ(self):
        c = Snowflurry.QuantumCircuit(qubit_count=3, gates=[sigma_x(1)])
        j = julia.julia()
        j.eval("jl.eval('push!(c,hadamard(1))')")

class Test_TestSnowflurryPennylaneBatch(unittest.TestCase):
    def test_batch_execute(self):
        dev = qml.device('snowflurry.qubit')
        tape = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.counts(wires=0)], shots=50)
        dev.execute(tape)
        

if __name__ == '__main__':
    #julia.install()
    #unittest.main()
    dev_def = qml.device("snowflurry.qubit", wires=1)
    @qml.qnode(dev_def)
    def testfunc():
        qml.PauliX(0)
        return qml.state()


    dev1 = qml.device("default.qubit", wires=1)
    @qml.qnode(dev1)
    def testfunc2():
        qml.PauliX(0)
        return qml.state()

    #tape = qml.tape.QuantumTape([qml.Hadamard(0)],[qml.expval(qml.PauliZ(0))])
    tape = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.expval(qml.PauliZ(0))])
    # tape = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.counts(wires=0)])
    # tape = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.counts(wires=0)], shots=1)
    print(f"results : {dev1.execute(tape)}")
    print(f"results : {dev_def.execute(tape)}")
    # print(type(dev1.execute(tape)))
    #dev = SnowflurryQubitDevice(wires=1)
    #make quantumtape with rx
    #enumerate gates
    #execute rx gate
    # with qml.tape.QuantumTape() as tape:
    #     qml.RX(0.432, wires=0)
    #     qml.RY(0.543, wires=0)
    #     qml.CNOT(wires=[0, 'a'])
    #     qml.RX(0.133, wires='a')
    #     qml.expval(qml.PauliZ(wires=[0]))
    #     print(tape.circuit)
    # c = Main.eval("""
    #             using Snowflurry
    #             QuantumCircuit(qubit_count=3, gates=[sigma_x(1)])
    #             """)
    # print(c)
    # ops = [qml.BasisState(np.array([1,1]), wires=(0,"a")),
    #     qml.RX(0.432, 0),
    #     qml.RY(0.543, 0),
    #     qml.CNOT((0,"a")),
    #     qml.RX(0.133, "a")]

    # qscript = QuantumScript(ops, [qml.expval(qml.PauliZ(0))])
    # for element in qscript:
    #     print(type(element))
    #     print(element)


