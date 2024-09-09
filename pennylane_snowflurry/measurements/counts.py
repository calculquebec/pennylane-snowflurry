from .measurement_strategy import MeasurementStrategy
from collections import Counter


class Counts(MeasurementStrategy):

    def __init__(self):
        super().__init__()

    def measure(self, converter, mp, shots):
        if self.Snowflurry.currentClient is None:
            converter.remove_readouts()
            converter.apply_readouts(mp.obs)
            shots_results = self.Snowflurry.simulate_shots(self.Snowflurry.sf_circuit, shots)
            result = dict(Counter(shots_results))
            return result
        else:  # if we have a client, we use the real machine
            converter.apply_readouts(mp.obs)
            qpu = self.Snowflurry.AnyonYamaskaQPU(
                self.Snowflurry.currentClient, self.Snowflurry.seval("project_id")
            )
            shots_results, time = self.Snowflurry.transpile_and_run_job(
                qpu, self.Snowflurry.sf_circuit, shots
            )
            result = dict(Counter(shots_results))
            return result
