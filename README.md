# pennylane-snowflurry

to install snowflurry, simply execute 

`pip install pennylane-snowflurry`

# project structure

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




# Running the tests

`python -m tests.test_pyjulia-snowflurry -v`