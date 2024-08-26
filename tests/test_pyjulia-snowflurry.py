import pennylane as qml
import unittest
from pennylane import numpy as np
from pennylane.tape import QuantumScript
from juliacall import newmodule

# TODO : this namespace should be imported from the plugin
Snowflurry = newmodule("Snowflurry")
Snowflurry.seval("using Snowflurry")


class TestMeasurements(unittest.TestCase):

    def setUp(self):
        self.dev_snowflurry = qml.device("snowflurry.qubit", wires=1)
        self.dev_pennylane = qml.device("default.qubit", wires=1)

    def test_circuit_basic_measure(self):
        """Test the basic Measurements functions from PennyLane with the Snowflurry device."""

        print("Testing basic measurements")

        def circuit_state():
            qml.Hadamard(0)
            return qml.state()

        r_s = qml.QNode(circuit_state, self.dev_snowflurry)
        r_p = qml.QNode(circuit_state, self.dev_pennylane)
        self.assertEqual(type(r_s), type(r_p))

        @qml.qnode(self.dev_snowflurry)
        def circuit_expval():
            qml.Hadamard(0)
            return qml.expval(qml.PauliZ(0))

        r_s = qml.QNode(circuit_expval, self.dev_snowflurry)
        r_p = qml.QNode(circuit_expval, self.dev_pennylane)
        self.assertEqual(type(r_s), type(r_p))

        def circuit_counts():
            qml.Hadamard(0)
            return qml.counts(qml.PauliY(0))

        r_s = qml.QNode(circuit_counts, self.dev_snowflurry)
        r_p = qml.QNode(circuit_counts, self.dev_pennylane)
        self.assertEqual(type(r_s), type(r_p))
        self.assertEqual(type(r_s), type(r_p))

        def circuit_probs():
            qml.Hadamard(0)
            return qml.probs(wires=[0])

        r_s = qml.QNode(circuit_probs, self.dev_snowflurry)
        r_p = qml.QNode(circuit_probs, self.dev_pennylane)
        self.assertEqual(type(r_s), type(r_p))

    def test_circuit_advanced_measure(self):
        print("Testing advanced measurements")
        self.dev_snowflurry = qml.device("snowflurry.qubit", wires=2)
        self.dev_pennylane = qml.device("default.qubit", wires=2)

        def circuit_density_matrix():
            qml.Hadamard(0)
            return qml.density_matrix([0])

        r_s = qml.QNode(circuit_density_matrix, self.dev_snowflurry)
        r_p = qml.QNode(circuit_density_matrix, self.dev_pennylane)
        self.assertEqual(type(r_s), type(r_p))

        def circuit_var():
            qml.Hadamard(0)
            return qml.var(qml.PauliY(0))

        r_s = qml.QNode(circuit_var, self.dev_snowflurry)
        r_p = qml.QNode(circuit_var, self.dev_pennylane)

        def circuit_purity():
            qml.Hadamard(0)
            qml.Hadamard(1)
            return qml.purity(wires=[0, 1])

        r_s = qml.QNode(circuit_purity, self.dev_snowflurry)
        r_p = qml.QNode(circuit_purity, self.dev_pennylane)
        self.assertEqual(type(r_s), type(r_p))

        def circuit_vn_entropy():
            qml.Hadamard(0)
            return qml.vn_entropy(wires=[0])

        r_s = qml.QNode(circuit_vn_entropy, self.dev_snowflurry)
        r_p = qml.QNode(circuit_vn_entropy, self.dev_pennylane)
        self.assertEqual(type(r_s), type(r_p))

        def circuit_mutual_info():
            qml.Hadamard(0)
            qml.Hadamard(1)
            return qml.mutual_info(wires0=[0], wires1=[1])

        r_s = qml.QNode(circuit_mutual_info, self.dev_snowflurry)
        r_p = qml.QNode(circuit_mutual_info, self.dev_pennylane)
        self.assertEqual(type(r_s), type(r_p))

        def circuit_classical_shadow():
            qml.Hadamard(0)
            qml.Hadamard(1)
            return qml.classical_shadow(wires=[0, 1])

        r_s = qml.QNode(circuit_classical_shadow, self.dev_snowflurry)
        r_p = qml.QNode(circuit_classical_shadow, self.dev_pennylane)
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

        r_s = qml.QNode(circuit_shadow_expval, self.dev_snowflurry)
        r_p = qml.QNode(circuit_shadow_expval, self.dev_pennylane)
        self.assertEqual(type(r_s), type(r_p))


class TestPyJulia(unittest.TestCase):
    def test_basic_julia(self):
        print("Testing basic Julia integration")
        c = Snowflurry.QuantumCircuit(qubit_count=3)  # TODO : import Snowflurry
        dev_def = qml.device("snowflurry.qubit", wires=3)
        self.assertEqual(True, True)  # TODO : complete this test


class TestNativeOperations(unittest.TestCase):

    def setUp(self):
        self.dev_snowflurry = qml.device("snowflurry.qubit", wires=1)
        self.dev_pennylane = qml.device("default.qubit", wires=1)

    def test_execute_gate_hadamard(self):
        print("Testing Hadamard gate")

        tape_pennylane = QuantumScript(
            [qml.Hadamard(0)], [qml.counts(wires=0)], shots=1
        )
        tape_snowflurry = QuantumScript([qml.Hadamard(0)], [qml.counts(wires=0)])

        self.assertEqual(
            type(self.dev_pennylane.execute(tape_pennylane)),
            type(self.dev_snowflurry.execute(tape_snowflurry)),
            "Hadamard gate, singular shot error",
        )

        tape = QuantumScript([qml.Hadamard(0)], [qml.counts(wires=0)], shots=50)
        results = self.dev_snowflurry.execute(tape)

        self.assertEqual(
            type(self.dev_pennylane.execute(tape)),
            type(results),
            "Hadamard gate, multiple shots errors",
        )

        self.assertTrue(
            abs(results["0"] - results["1"]) < 10, "Hadamard gate, statistical error"
        )
        tape = qml.tape.QuantumTape([qml.Hadamard(0)], [qml.expval(qml.PauliZ(0))])

        self.assertEqual(
            self.dev_pennylane.execute(tape), self.dev_snowflurry.execute(tape)
        )

        tape = QuantumScript([qml.Hadamard(0)], [qml.expval(qml.PauliZ(0))])
        self.assertEqual(
            self.dev_pennylane.execute(tape), self.dev_snowflurry.execute(tape)
        )

    def test_gate_PauliX(self):
        print("Testing PauliX gate")
        tape = QuantumScript([qml.PauliX(0)], [qml.counts(wires=0)], shots=50)
        self.assertTrue(
            self.dev_snowflurry.execute(tape)["1"] == 50,
            "PauliX gate, statistical error",
        )

        @qml.qnode(self.dev_snowflurry)
        def snowflurry_circuit():
            qml.PauliX(0)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(self.dev_pennylane)
        def pennylane_circuit():
            qml.PauliX(0)
            return qml.expval(qml.PauliZ(0))

        self.assertEqual(snowflurry_circuit(), pennylane_circuit())


if __name__ == "__main__":

    unittest.main()
