from .measurement_strategy import MeasurementStrategy
import numpy as np


class Sample(MeasurementStrategy):

    def __init__(self):
        super().__init__()

    def measure(self, converter, mp, shots):
        if self.Snowflurry.currentClient is None:
            converter.remove_readouts()
            converter.apply_readouts(mp.obs)
            shots_results = self.Snowflurry.simulate_shots(self.Snowflurry.sf_circuit, shots)
            return np.asarray(shots_results).astype(int)
        else:
            converter.apply_readouts(mp.obs)
            qpu = self.Snowflurry.AnyonYamaskaQPU(
                self.Snowflurry.currentClient, self.Snowflurry.seval("project_id")
            )
            shots_results, time = self.Snowflurry.transpile_and_run_job(
                qpu, self.Snowflurry.sf_circuit, shots,
            )
            return np.repeat(
                [int(key) for key in shots_results.keys()],
                [value for value in shots_results.values()],
            )