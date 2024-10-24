from pennylane.tape import QuantumTape
from pennylane.operation import Operation
from pennylane.measurements import MeasurementProcess
from pennylane_snowflurry.api_utility import ApiUtility
from dotenv import dotenv_values
import requests
import json

class ApiAdapter:
    """
    a wrapper around Thunderhead. Provide a host, user, access token and realm, and you can :
    - create job with circuit dict, circuit name, project id, machine name and shots count
    - get benchmark by machine name
    - get machine id by name

    Args : 
    """
    def __init__(self):
        config = dotenv_values(".env")

        self.host = config["HOST"]
        self.machine_name = config["MACHINE_NAME"]
        self.user = config["USER"]
        self.access_token = config["ACCESS_TOKEN"]
        self.project_id = config["PROJECT_ID"]
        self.realm = config["REALM"]

        self.headers = ApiUtility.headers(self.user, self.access_token, self.realm)
    
    def get_machine_id_by_name(self):
        route = self.host + ApiUtility.routes.machines + ApiUtility.routes.machineName + "=" + self.machine_name
        return requests.get(route, headers=self.headers)
    
    def get_qubits_and_couplers(self) -> dict[str, any] | None:
        res = self.get_benchmark()
        if res.status_code != 200:
            return None
        return json.loads(res.text)[ApiUtility.keys.resultsPerDevice]

    def get_benchmark(self):
        res = self.get_machine_id_by_name()
        if res.status_code != 200:
            return None
        result = json.loads(res.text)
        machine_id = result[ApiUtility.keys.items][0][ApiUtility.keys.id]
    
        route = self.host + ApiUtility.routes.machines + "/" + machine_id + ApiUtility.routes.benchmarking
        return requests.get(route, headers=self.headers)
    
    def create_job(self, circuit : dict[str, any], 
                   shot_count : int,
                   circuit_name: str = "default") -> requests.Response:
        body = ApiUtility.job_body(circuit, circuit_name, self.project_id, self.machine_name, shot_count)
        return requests.post(self.host + ApiUtility.routes.jobs, data=json.dumps(body), headers=self.headers)

    def list_jobs(self) -> requests.Response:
        return requests.get(self.host + ApiUtility.routes.jobs, headers=self.headers)

    def job_by_id(self, id : str) -> requests.Response:
        return requests.get(self.host + ApiUtility.routes.jobs + f"/{id}", headers=self.headers)
