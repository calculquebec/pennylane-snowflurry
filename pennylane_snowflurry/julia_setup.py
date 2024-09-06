import juliapkg
from juliapkg import PkgSpec
import json
import os

# Required packages for the Julia environment
REQUIRED_PACKAGES = [
    PkgSpec(
        name="Snowflurry", uuid="7bd9edc1-4fdc-40a1-a0f6-da58fb4f45ec", version="0.4"
    ),
]

IS_USER_CONFIGURED = False


class JuliaEnv:
    """
    This class is used to resolve dependencies for the Julia environment. It will install Julia and the required
    packages if they are not already installed or up to date.

    It makes use of the juliapkg Python package to manage installations and dependencies. Upon initialization,
    juliapkg will create a directory in which metadata regarding dependencies will be stored. A JSON file saved in
    this directory is used by juliapkg to resolve dependencies.

    Simply calling the update method will install the required packages by manipulating the JSON file and forcing
    juliapkg to resolve dependencies.

    It is possible to deactivate the automatic update by setting the IS_USER_CONFIGURED variable to True.

    Attributes:
        julia_env_path: str: The path to the Julia environment metadata directory created by juliapkg
        json_pkg_list: dict: Content of the JSON file used by juliapkg to resolve dependencies
        json_path: str: The path to the juliapkg.json file used by juliapkg to resolve dependencies
        required_packages: list: The required packages defined in the REQUIRED_PACKAGES variable in julia_setup.py

    """

    def __init__(self):
        self.julia_env_path = juliapkg.project()
        self.json_path = self.julia_env_path + "/pyjuliapkg/juliapkg.json"
        self.json_pkg_list = self.get_json_pkg_list()
        self.required_packages = REQUIRED_PACKAGES

    def update(self):
        """
        This function updates the Julia environment by manipulating a JSON file and forcing juliapkg to resolve
        the environment's dependencies.
        Setting the IS_USER_CONFIGURED variable to True will deactivate the update.
        """
        if IS_USER_CONFIGURED:
            return

        for required_pkg in self.required_packages:
            if required_pkg.name in self.json_pkg_list:
                if self.parse_version(required_pkg):
                    continue
            self.new_json_pkg_list()
            self.write_json()
            juliapkg.resolve(force=True)
            break

    def get_json_pkg_list(self) -> dict:
        """
        This function returns a dictionary of packages in the configuration file juliapkg.json.
        The key is the package name and the value is a dictionary containing the package's UUID and version.

        Returns: dict

        """
        if not os.path.exists(self.json_path):
            return {}
        with open(self.json_path, "r") as f:
            juliapkg_data = json.load(f)

        return juliapkg_data.get("packages", [])

    def parse_version(self, required_pkg) -> bool:
        """
        This function compares the version of the required package with the version in the JSON file.

        Args:
            required_pkg: PkgSpec: The required package
        """
        try:
            for package_name, package_info in self.json_pkg_list.items():
                if str(package_name) == required_pkg.name:
                    if package_info.get("version", []) == required_pkg.version:
                        return True
                    return False
        except:
            LookupError("Package not found")

    def write_json(self):
        """
        This function writes the updated list of packages to the JSON file
        It also creates the JSON file if it does not exist.
        """
        if "packages" not in self.json_pkg_list:
            self.json_pkg_list = {"packages": self.json_pkg_list}
        with open(self.json_path, "w") as f:
            json.dump(self.json_pkg_list, f)

    def new_json_pkg_list(self):
        """
        This function updates the list of packages with the required packages
        """

        for required_pkg in self.required_packages:
            self.json_pkg_list[required_pkg.name] = {
                "uuid": required_pkg.uuid,
                "version": required_pkg.version,
            }
