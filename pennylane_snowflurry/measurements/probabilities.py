from .measurement_strategy import MeasurementStrategy
from juliacall import convert


class Probabilities(MeasurementStrategy):

    def __init__(self):
        super().__init__()

    def measure(self, converter, mp, shots):
        converter.remove_readouts()
        wires_list = mp.wires.tolist()
        if len(wires_list) == 0:
            return self.Snowflurry.get_measurement_probabilities(self.Snowflurry.sf_circuit)
        else:
            return self.Snowflurry.get_measurement_probabilities(
                self.Snowflurry.sf_circuit,
                convert(self.Snowflurry.Vector, [i + 1 for i in wires_list])
            )