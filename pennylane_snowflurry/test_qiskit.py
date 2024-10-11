from qiskit.quantum_info import Statevector

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi

def circuit1():
	qreg_q = QuantumRegister(9, 'q')
	creg_c = ClassicalRegister(4, 'c')
	circuit = QuantumCircuit(qreg_q, creg_c)

	circuit.h(qreg_q[0])
	circuit.h(qreg_q[1])
	circuit.h(qreg_q[2])
	circuit.h(qreg_q[3])
	circuit.h(qreg_q[4])
	circuit.z(qreg_q[4])
	circuit.cx(qreg_q[0], qreg_q[4])
	circuit.h(qreg_q[0])
	circuit.h(qreg_q[1])
	circuit.h(qreg_q[2])
	circuit.h(qreg_q[3])
	circuit.measure(qreg_q[0], creg_c[0])
	circuit.measure(qreg_q[1], creg_c[1])
	circuit.measure(qreg_q[2], creg_c[2])
	circuit.measure(qreg_q[3], creg_c[3])
	return circuit

def circuit2():
	from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
	from numpy import pi

	qreg_q = QuantumRegister(9, 'q')
	creg_c = ClassicalRegister(4, 'c')
	circuit = QuantumCircuit(qreg_q, creg_c)
	circuit.s(qreg_q[1])
	circuit.s(qreg_q[5])
	circuit.sx(qreg_q[1])
	circuit.s(qreg_q[1])
	circuit.x(qreg_q[5])
	circuit.s(qreg_q[5])
	circuit.cz(qreg_q[1], qreg_q[5])
	circuit.s(qreg_q[1])
	circuit.s(qreg_q[5])
	circuit.sx(qreg_q[1])
	circuit.s(qreg_q[1])
	circuit.sx(qreg_q[5])
	circuit.s(qreg_q[5])
	circuit.measure(qreg_q[1], creg_c[0])
	circuit.measure(qreg_q[2], creg_c[1])
	circuit.measure(qreg_q[3], creg_c[2])
	circuit.measure(qreg_q[4], creg_c[3])
	
	return circuit

def circuit3():
	from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
	from numpy import pi

	qreg_q = QuantumRegister(9, 'q')
	creg_c = ClassicalRegister(4, 'c')
	circuit = QuantumCircuit(qreg_q, creg_c)

	circuit.sx(qreg_q[0])
	circuit.x(qreg_q[4])
	circuit.cz(qreg_q[0], qreg_q[4])
	circuit.z(qreg_q[0])
	circuit.z(qreg_q[4])
	circuit.sx(qreg_q[4])
	circuit.sx(qreg_q[0])
	circuit.measure(qreg_q[0], creg_c[0])
	circuit.measure(qreg_q[1], creg_c[1])
	circuit.measure(qreg_q[7], creg_c[2])
	circuit.measure(qreg_q[8], creg_c[3])
	
	return circuit

not_transpiled = circuit1()
sf_transpiled = circuit2()
cq_transpiled = circuit3()

# Transpile for simulator
simulator = AerSimulator()
circ = transpile(not_transpiled, simulator)

# Run and get counts
result = simulator.run(circ).result()
counts = result.get_counts(circ)
print("BV 5 qubits non-transpiled" + str(counts))

circ = transpile(sf_transpiled, simulator)

# Run and get counts
result = simulator.run(circ).result()
counts = result.get_counts(circ)
print("BV 5 qubits sf transpiled" + str(counts))

circ = transpile(cq_transpiled, simulator)

# Run and get counts
result = simulator.run(circ).result()
counts = result.get_counts(circ)
print("BV 5 qubits cq transpiled" + str(counts))
