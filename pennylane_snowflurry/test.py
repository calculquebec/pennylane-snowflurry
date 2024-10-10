if __name__ == "__main__":
    import pennylane as qml
    from test_circuits import bernstein_varizani
    from pennylane_snowflurry.test_device import TestDevice

    num_wires = 5
    dev : qml.Device = TestDevice()
    
    print(qml.draw(bernstein_varizani)(11))
    result = qml.QNode(bernstein_varizani, dev)(11)
    print(list([float(i) for i in result]))

