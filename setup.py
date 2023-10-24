# Copyright 2018 Carsten Blank

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#https://docs.pennylane.ai/en/stable/development/plugins.html#installing-plugin
#https://github.com/PennyLaneAI/pennylane-qiskit/blob/517db668804f92851dabd88a70fd323a47a2e3dc/setup.py#L44
#!/usr/bin/env python3
from setuptools import setup

with open("pennylane_snowflurry/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    #"pennylane>=0.27",
    "julia",
    "numpy",
]

info = {
    'name': 'Pennylane-snowflurry',
    'version': version,
    'maintainer': 'Calcul Québec',
    'maintainer_email': 'support@tech.alliancecan.ca',
    'url': 'https://github.com/calculquebec/pennylane-snowflurry',
    'license': 'Apache License 2.0',
    'packages': [
        'pennylane_snowflurry'
    ],
    'entry_points': {
        'pennylane.plugins': [
            'snowflurry.qubit = pennylane_snowflurry:SnowflurryQubitDevice'
            ],
        'pennylane.io': [
            ],
        },
    'description': 'PennyLane plugin for interfacing with Anyon\'s quantum computers',
    'long_description': open('README.md').read(),
    'provides': ["pennylane_snowflurry"],
    'install_requires': requirements,
    # 'extras_require': extra_requirements,
    'command_options': {
        'build_sphinx': {
            'version': ('setup.py', version),
            'release': ('setup.py', version)}}
}

classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3 :: Only',
    "Topic :: Scientific/Engineering :: Physics"
]

setup(classifiers=classifiers, **(info))