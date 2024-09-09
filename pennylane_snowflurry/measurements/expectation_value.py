from .measurement_strategy import MeasurementStrategy
import pennylane as qml
from juliacall import convert
import numpy as np


class ExpectationValue(MeasurementStrategy):

    def __init__(self):
        super().__init__()

    def measure(self, converter, mp, shots):
        # FIXME : this measurement does work when the number of qubits measured is not equal to the number of qubits
        #  in the circuit
        # Requires some processing to work with larger matrices
        converter.remove_readouts()
        self.Snowflurry.result_state = self.Snowflurry.simulate(self.Snowflurry.sf_circuit)
        if mp.obs is not None and mp.obs.has_matrix:
            print(mp.obs)
            observable_matrix = qml.matrix(mp.obs)
            print(observable_matrix)
            expected_value = self.Snowflurry.expected_value(
                self.Snowflurry.DenseOperator(convert(self.Snowflurry.Matrix, observable_matrix)),
                self.Snowflurry.result_state
            )
            return np.real(expected_value)
