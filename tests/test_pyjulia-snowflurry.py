# from snowflurry_device import SnowflurryQubitDevice
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
        self.assertEqual(True, True)  # TODO : complete this test

    def test_execute_gate_hadamard(self):
        dev_snowflurry = qml.device("snowflurry.qubit", wires=1)
        dev_pennylane = qml.device("default.qubit", wires=1)

        tape_pennylane = qml.tape.QuantumScript(
            [qml.Hadamard(0)], [qml.counts(wires=0)], shots=1
        )
        tape_snowflurry = qml.tape.QuantumScript(
            [qml.Hadamard(0)], [qml.counts(wires=0)]
        )

        self.assertEqual(
            type(dev_pennylane.execute(tape_pennylane)),
            type(dev_snowflurry.execute(tape_snowflurry)),
            "Hadamard gate, singular shot error",
        )

        tape = qml.tape.QuantumScript(
            [qml.Hadamard(0)], [qml.counts(wires=0)], shots=50
        )
        results = dev_snowflurry.execute(tape)

        self.assertEqual(
            type(dev_pennylane.execute(tape)),
            type(results),
            "Hadamard gate, multiple shots errors",
        )

        self.assertTrue(
            abs(results["0"] - results["1"]) < 10, "Hadamard gate, statistical error"
        )
        tape = qml.tape.QuantumTape([qml.Hadamard(0)], [qml.expval(qml.PauliZ(0))])

        self.assertEqual(dev_pennylane.execute(tape), dev_snowflurry.execute(tape))

        tape = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.expval(qml.PauliZ(0))])
        self.assertEqual(dev_pennylane.execute(tape), dev_snowflurry.execute(tape))

    def test_circuit_basic_measure(self):
        """Test the basic Measurements functions from PennyLane with the Snowflurry device."""

        dev_snowflurry = qml.device("snowflurry.qubit", wires=1)
        dev_pennylane = qml.device("default.qubit", wires=1)

        def circuit_state():
            qml.Hadamard(0)
            return qml.state()

        r_s = qml.QNode(circuit_state, dev_snowflurry)
        r_p = qml.QNode(circuit_state, dev_pennylane)
        self.assertEqual(type(r_s), type(r_p))

        @qml.qnode(dev_snowflurry)
        def circuit_expval():
            qml.Hadamard(0)
            return qml.expval(qml.PauliZ(0))

        r_s = qml.QNode(circuit_expval, dev_snowflurry)
        r_p = qml.QNode(circuit_expval, dev_pennylane)
        self.assertEqual(type(r_s), type(r_p))

        def circuit_counts():
            qml.Hadamard(0)
            return qml.counts(qml.PauliY(0))

        r_s = qml.QNode(circuit_counts, dev_snowflurry)
        r_p = qml.QNode(circuit_counts, dev_pennylane)
        self.assertEqual(type(r_s), type(r_p))
        self.assertEqual(type(r_s), type(r_p))

        def circuit_probs():
            qml.Hadamard(0)
            return qml.probs(wires=[0])

        r_s = qml.QNode(circuit_probs, dev_snowflurry)
        r_p = qml.QNode(circuit_probs, dev_pennylane)
        self.assertEqual(type(r_s), type(r_p))

    def test_circuit_advanced_measure(self):
        dev_snowflurry = qml.device("snowflurry.qubit", wires=2)
        dev_pennylane = qml.device("default.qubit", wires=2)

        def circuit_density_matrix():
            qml.Hadamard(0)
            return qml.density_matrix([0])

        r_s = qml.QNode(circuit_density_matrix, dev_snowflurry)
        r_p = qml.QNode(circuit_density_matrix, dev_pennylane)
        self.assertEqual(type(r_s), type(r_p))

        def circuit_var():
            qml.Hadamard(0)
            return qml.var(qml.PauliY(0))

        r_s = qml.QNode(circuit_var, dev_snowflurry)
        r_p = qml.QNode(circuit_var, dev_pennylane)

        def circuit_purity():
            qml.Hadamard(0)
            qml.Hadamard(1)
            return qml.purity(wires=[0, 1])

        r_s = qml.QNode(circuit_purity, dev_snowflurry)
        r_p = qml.QNode(circuit_purity, dev_pennylane)
        self.assertEqual(type(r_s), type(r_p))

        def circuit_vn_entropy():
            qml.Hadamard(0)
            return qml.vn_entropy(wires=[0])

        r_s = qml.QNode(circuit_vn_entropy, dev_snowflurry)
        r_p = qml.QNode(circuit_vn_entropy, dev_pennylane)
        self.assertEqual(type(r_s), type(r_p))

        def circuit_mutual_info():
            qml.Hadamard(0)
            qml.Hadamard(1)
            return qml.mutual_info(wires0=[0], wires1=[1])

        r_s = qml.QNode(circuit_mutual_info, dev_snowflurry)
        r_p = qml.QNode(circuit_mutual_info, dev_pennylane)
        self.assertEqual(type(r_s), type(r_p))

        def circuit_classical_shadow():
            qml.Hadamard(0)
            qml.Hadamard(1)
            return qml.classical_shadow(wires=[0, 1])

        r_s = qml.QNode(circuit_classical_shadow, dev_snowflurry)
        r_p = qml.QNode(circuit_classical_shadow, dev_pennylane)
        self.assertEqual(type(r_s), type(r_p))

        def circuit_shadow_expval():
            qml.Hadamard(0)
            qml.Hadamard(1)
            return qml.shadow_expval(
                qml.Hamiltonian(
                    [1.0, 1.0],
                    [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)],
                )
            )

        r_s = qml.QNode(circuit_shadow_expval, dev_snowflurry)
        r_p = qml.QNode(circuit_shadow_expval, dev_pennylane)
        self.assertEqual(type(r_s), type(r_p))

    def test_gate_PauliX(self):
        dev_snowflurry = qml.device("snowflurry.qubit", wires=1)
        tape = qml.tape.QuantumScript([qml.PauliX(0)], [qml.counts(wires=0)], shots=50)
        self.assertTrue(
            dev_snowflurry.execute(tape)["1"] == 50, "PauliX gate, statistical error"
        )

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


class Test_TestSnowflurryPennylaneBatch(unittest.TestCase):
    def test_batch_execute(self):
        dev = qml.device("snowflurry.qubit")
        tape = qml.tape.QuantumScript(
            [qml.Hadamard(0)], [qml.counts(wires=0)], shots=50
        )
        dev.execute(tape)


if __name__ == "__main__":
    # julia.install()
    unittest.main()
    dev_def = qml.device("snowflurry.qubit", wires=1, shots=50)

    @qml.qnode(dev_def)
    def testfunc():
        qml.PauliX(0)
        return qml.state()

    dev1 = qml.device("default.qubit", wires=1, shots=50)

    @qml.qnode(dev1)
    def testfunc2():
        qml.PauliX(0)
        return qml.state()

    # tape = qml.tape.QuantumTape([qml.Hadamard(0)],[qml.expval(qml.PauliZ(0))])
    # tape = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.counts(wires=0)])
    # tape = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.counts(wires=0)], shots=1)
    # tape = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.expval(qml.PauliZ(0))])
    # print(f"results : {dev1.execute(tape)}")
    # print(f"results : {dev_def.execute(tape)}")
    # tape = qml.tape.QuantumScript([qml.Hadamard(0),qml.Hadamard(1)], [qml.counts(wires=[0,1])], shots=50)
    # new_dev = qml.device("snowflurry.qubit", wires=1)
    # results = new_dev.execute(tape)
    # print(f"results : {dev1.execute(tape)}")
    # print(f"results : {results}")
    # tape = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.state()])
    # print(f"results : {dev1.execute(tape)}")
    # print(f"results : {dev_def.execute(tape)}")
    @qml.qnode(dev1)
    def circ1():
        qml.RX(0.5, 0)
        return qml.probs()

    @qml.qnode(dev_def)
    def circ2():
        qml.RX(0.5, 0)
        return qml.probs()

    print(f"results : {circ1()}")
    print(f"results : {circ2()}")
