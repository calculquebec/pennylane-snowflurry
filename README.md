# pennylane-snowflurry

The PennyLane-Snowflurry plugin provides a PennyLane device that allows the use of Anyon Systems' Snowflurry quantum computing platform with PennyLane.

[Penylane](https://pennylane.ai/) is a cross-platform Python library for quantum machine learning, automatic differentiation, and optimization of hybrid quantum-classical computations.

[Snowflurry](https://snowflurry.org/) is a quantum computing framework developed in Julia by Anyon Systems and aims to provide access to quantum hardware and simulators.

PennyLane-Snowflurry makes use of dependencies such as PyJulia and PyCall to allow interfacing between Python and Julia, and thus between PennyLane and Snowflurry.

## Project structure

As shown in the diagram below, this plugin is used in Pennylane as a [device](https://pennylane.ai/plugins/) named `snowflurry.qubit`. This device is defined by the class `SnowflurryQubitDevice`. It converts a PennyLane circuit into a Snowflurry circuit, thanks to packages like PyJulia that allow the communication between Python and Julia environments. The Snowflurry circuit can then be used with the available backends, either a simulator or real quantum hardware. The results are then converted back into PennyLane's format and returned to the user.

![interaction_diagram](./doc/interaction_diagram_extended.png)

## Local installation

Since this plugin interfaces between Python and Julia, it requires both languages to be installed on your machine. As Python is widely used amongst the quantum computing community, we assume you already have it installed. The rest of this section will guide you through the installation of Julia and the plugin.

### Julia

If you don't have Julia installed, you can download it from the [official website](https://julialang.org/downloads/).

### PennyLane and Snowflurry

Before installing this plugin, makes sure you have a working Pennylane and Snowflurry installation.

For PennyLane, please refer to the [PennyLane documentation](https://pennylane.ai/install/).

For Snowflurry, please refer to the [Snowflurry documentation](https://snowflurry.org).

### Plugin installation

This plugin is available on PyPI, so you can install it with pip:

```sh
pip install pennylane-snowflurry
```

Alternatively, you can clone this repo and install the plugin with the following command from the root of the repo:

```sh
pip install -e .
```

### PyJulia and PyCall

PyJulia and PyCall are used to communicate between Python and Julia. At this point, PyJulia should already be installed with the plugin as it is listed in the dependencies, but you'll need to install PyCall in your Julia environment. To do so open a python terminal and execute the following commands:

```py
import julia
julia.install()
```

Alternatively, you could also install PyCall from the Julia REPL, but the previous method makes sure to build the package for your current python environment.

```julia
using Pkg
Pkg.add("PyCall")
```

### Running files and tests

For now, the plugin is only tested on python scripts and doesn't work in a Juptyer notebook. To run a file, you can use the following command:

```sh
python-jl -m tests.test_pyjulia-snowflurry -v
```

## Usage

Once installed, you can write your PennyLane circuits as usual, but you'll need to specify the device as `snowflurry.qubit` and provide the Snowflurry backend you want to use if you have access to a quantum computer.

```py
dev_def = qml.device("snowflurry.qubit", wires=1, shots=50)
```

Example if you have an API key from Anyon Systems:

```py
dev_def = qml.device("snowflurry.qubit", wires=1, shots=50, host="example.anyonsys.com", user="test_user",access_token="not_a_real_access_token")
```

## State of the project and known issues

This plugin is still very early in its development and aims to provide a basic interface between PennyLane and Snowflurry, which are both also under active development. As such, it is expected that there will be issues and limitations.

It is necessary to use the `python-jl` command to execute files, as it essentially launches Python from within Julia. This is necessary for the use of Julia packages such as Snowflurry.

For the sake of simplicity, a workaround it being researched to allow the use of the plugin in a Jupyter notebook and to allow debugging in an IDE.
