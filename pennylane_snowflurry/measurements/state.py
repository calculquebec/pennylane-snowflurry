from .measurement_strategy import MeasurementStrategy

import numpy as np


class State(MeasurementStrategy):

    def __init__(self):
        super().__init__()

    def measure(self, converter, mp, shots):
        converter.remove_readouts()
        self.Snowflurry.result_state = self.Snowflurry.simulate(self.Snowflurry.sf_circuit)
        # Convert the final state from pyjulia to a NumPy array
        final_state_np = np.array([element for element in self.Snowflurry.result_state])
        return final_state_np
