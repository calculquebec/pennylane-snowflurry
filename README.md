# pennylane-snowflurry


# Local installation
clone repo on your computer

pip install git+https://github.com/PennyLaneAI/pennylane.git@master

the following command will install the pennylane plugin on your computer
pip install -e .

once done, to execute any file in this repo, do somthing like

python -m tests.test_pennylaneConverter

or

python -m pennylane_snowflurry.snowflurry_device




# Running the tests

python -m tests.test_pyjulia-snowflurry -v