import pennylane.tape as tape
import json
import time
from api_adapter import ApiAdapter

class JobException(Exception):
    def __init__(self, message : str):
        self.message = message
    
    def __str__(self): self.message
    
class Job:
    host : str
    user : str
    access_token : str
    realm : str
    
    def __init__(self, 
                 host = "", 
                 user = "", 
                 access_token = "", 
                 realm = "", ):
        self.adapter = ApiAdapter(host, user, access_token, realm)
        
    def verbosePrint(content, verbose : bool = False):
        if verbose:
            print(content)

    def run(self, circuit : tape.QuantumScript, circuit_name = "", project_id = "", machine_name = "", verbose = False):
        
        r = self.adapter.create_job(circuit, circuit_name=circuit_name, project_id=project_id, machine_name=machine_name)
        
        if(r.status_code == 200):
            current_status = ""
            job_id = json.loads(r.text)["job"]["id"]
            Job.verbosePrint("sent job with id : " + job_id, verbose)
            while(True):
                time.sleep(0.5)
                r = self.adapter.job_by_id(job_id)
                if r.status_code != 200: continue

                content = json.loads(r.text)
                status = content["job"]["status"]["type"]
                if(current_status != status):
                    current_status = status
                    Job.verbosePrint(current_status, verbose)

                if(status != "SUCCEEDED"): continue

                return content["result"]["histogram"]

        else:
            raise JobException(r.text)
    
    

