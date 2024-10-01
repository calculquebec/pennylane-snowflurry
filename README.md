# pennylane-snowflurry

The PennyLane-Snowflurry plugin provides a PennyLane device that allows the use of Anyon Systems' Snowflurry quantum computing platform with PennyLane.

[Pennylane](https://pennylane.ai/) is a cross-platform Python library for quantum machine learning, automatic differentiation, and optimization of hybrid quantum-classical computations.

[Snowflurry](https://snowflurry.org/) is a quantum computing framework developed in Julia by Anyon Systems and aims to provide access to quantum hardware and simulators.

PennyLane-Snowflurry makes use of dependencies like [PythonCall and JuliaCall](https://github.com/JuliaPy/PythonCall.jl) to allow interfacing between Python and Julia, and thus between PennyLane and Snowflurry.

## Project structure

As shown in the diagram below, this plugin is used in Pennylane as a [device](https://pennylane.ai/plugins/) named `snowflurry.qubit`. This device is defined by the class `SnowflurryQubitDevice`. It converts a PennyLane circuit into a Snowflurry circuit, thanks to packages like JuliaCall that allow the communication between Python and Julia environments. The Snowflurry circuit can then be used with the available backends, either a simulator or real quantum hardware. The results are then converted back into PennyLane's format and returned to the user.

![interaction_diagram](https://raw.githubusercontent.com/calculquebec/pennylane-snowflurry/main/doc/interaction_diagram_extended.png)

## Local installation

Since this plugin interfaces between Python and Julia, it requires both languages to be installed on your machine. As Python is widely used amongst the quantum computing community, we assume you already have it installed with a package manager like pip.

Note that to use [Calcul Québec's services](https://docs.alliancecan.ca/wiki/Les_services_quantiques/en), you may not need to install the plugin locally as our users might have access to a pre-configured environment.

### Plugin installation

Pennylane-snowflurry can be installed using pip:

```sh
pip install pennylane-snowflurry
```

Alternatively, you can clone this repo and install the plugin with the following command from the root of the repo:

```sh
pip install -e .
```

Pennylane and other Python dependencies will be installed automatically during the installation process.

The plugin will also take care of installing Julia and the required Julia packages, such as Snowflurry and PythonCall during the first run. Some notes on that matter are provided below.

If you wish to disable this behaviour, you can edit the julia_env.py file and set the value of the variable `IS_USER_CONFIGURED` to `TRUE`:

```py
IS_USER_CONFIGURED = True
```

## Julia

As of version 0.3.0, **there is no need to install Julia manually**, since the plugin will download and install the required version automatically upon first use. This Julia environment is bound to the plugin.

However, if you wish to manage your Julia environment, you can download it from the [official website](https://julialang.org/downloads/). It is highly recommended to install using the installer file, as it will ask to add Julia to the system's environment variables.

**To ensure this correct configuration, during the installation process, the checkbox `Add Julia to PATH` must be checked.**

## PennyLane and Snowflurry

Those packages are installed automatically during the plugin installation process and are necessary for the plugin to work. Here are the links to their respective documentation:

For PennyLane, please refer to the [PennyLane documentation](https://pennylane.ai/install/).

For Snowflurry, please refer to the [Snowflurry documentation](https://snowflurry.org).

## Usage

### Running files

The plugin can be used both in python scripts and Jupyter notebooks. To run a script, you can use the following command:

```sh
python base_circuit.py
```

### How to call the device

Once installed, you can write your PennyLane circuits as usual, but you'll need to specify the device as `snowflurry.qubit` and provide the Snowflurry backend you want to use if you have access to a quantum computer.

```py
dev_def = qml.device("snowflurry.qubit", wires=1, shots=50)
```

Example if you have an API key from Anyon Systems:

```py
dev_def = qml.device("snowflurry.qubit", wires=1, shots=50, host="example.anyonsys.com", user="test_user",access_token="not_a_real_access_token", realm="realm_name")
```

## State of the project and known issues

This plugin is still very early in its development and aims to provide a basic interface between PennyLane and Snowflurry, which are both also under active development. As such, it is expected that there will be issues and limitations.

### Future plans

- Add a test suite to ensure the plugin works as expected.
- Integrate a compiler to optimize the circuits.
- Add device that allows for communication with MonarQ directly through its API.