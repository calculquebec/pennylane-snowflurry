import juliapkg
from juliapkg import PkgSpec
import json


# Required packages for the Julia environment
REQUIRED_PACKAGES = [
    PkgSpec(
        name="Snowflurry", uuid="7bd9edc1-4fdc-40a1-a0f6-da58fb4f45ec", version="0.3"
    )
]


class JuliaEnv:
    """
    This class is used to manage the Julia environment in the specific context of the Snowflurry plugin
    using the rudimentary juliapkg package manager functions.

    This is integrated in the plugin's __init__.py file to ensure that the required packages are installed
    before execution."""

    def __init__(self):
        self.pkg_path = juliapkg.project()  # Get the path to the Julia environment
        self.pkg_list = self.get_pkg_list()  # Get the list of installed packages
        self.required_packages = REQUIRED_PACKAGES  # Required packages

    def update(self):
        """
        This function updates the Julia environment with the required packages if they are
        not already installed or up to date.
        """
        for pkg in self.required_packages:
            if pkg.name in self.pkg_list:
                if self.parse_version(pkg):
                    continue

            juliapkg.add(pkg.name, pkg.uuid, version=pkg.version)

    def get_pkg_list(self):
        """
        This function returns the list of packages that are currently installed in the Julia environment
        """
        with open(self.pkg_path + "/pyjuliapkg/juliapkg.json", "r") as f:
            juliapkg_data = json.load(f)

        return juliapkg_data.get("packages", [])

    def parse_version(self, pkg) -> bool:
        """
        This function lookups the pkg passed in argument within the list of installed packages
        and compares the version to the one passed in argument. If the package is found and the version
        is the same, it returns True, otherwise it returns False.
        """
        try:
            for package in self.pkg_list:
                if package["name"] == pkg.name:
                    if package["version"] == pkg.version:
                        return True
                    return False
        except:
            LookupError("Package not found")
