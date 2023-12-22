# pennylane-snowflurry

To install the snowflurry plugin, execute 

`pip install pennylane-snowflurry`

once done, you should be able to use the new device like so:

```py
dev_def = qml.device("snowflurry.qubit", wires=1, shots=50)
```

Example if you have access to anyon's quantum computer:

```py
dev_def = qml.device("snowflurry.qubit", wires=1, shots=50, host="example.anyonsys.com", user="test_user",access_token="not_a_real_access_token")
```

## State of the project

This plugin is still very early in its development and not thoroughly tested. expect issues.


## Project structure

![puml diagram](/doc/interaction_diagram.png)

# Local installation

first, clone this repo 

then

`pip install -r requirements.txt`


`pip install git+https://github.com/PennyLaneAI/pennylane.git@master`

the following command will install the pennylane plugin on your computer:

`pip install -e .`

this plugin require julia and as of now, the setup.py doesn't install julia automatically for you.
to install it, open the python interpreter and type in

```py
import julia
julia.install()
```

once done, to execute any file in this repo, you can do :

`python -m tests.test_pennylaneConverter`

or

`python -m pennylane_snowflurry.snowflurry_device`




## Running the tests

`python -m tests.test_pyjulia-snowflurry -v`