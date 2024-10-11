if __name__ == "__main__":
    import pennylane as qml
    from test_circuits import bernstein_varizani
    from pennylane_snowflurry.monarq_device import MonarqDevice
    from pennylane_snowflurry.test_device import TestDevice

    class const:
        host = "https://manager.anyonlabs.com"
        user = "stage"
        access_token = "FjjIKjmDMoAMzSO4v2Bu62a+8vD39zib"
        realm = "calculqc"
        machine_name = "yamaska"
        project_id = ""
        circuit_name = "test_circuit"

    num_wires = 5
    dev : qml.Device = TestDevice(num_wires, 100, const.host, const.user, const.access_token)
    
    print(qml.draw(bernstein_varizani)(11))
    result = qml.QNode(bernstein_varizani, dev)(11)
    print(list([float(i) for i in result]))

