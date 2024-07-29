from .julia_setup import JuliaEnv
from configparser import ConfigParser
from os import path as os_path

CONFIG_FILE_PATH = os_path.abspath(os_path.dirname(__file__) + "\\..\\env_config.ini")

# Load the configuration file
config = ConfigParser()
config.read(CONFIG_FILE_PATH)


if not config.getboolean("JULIA", "is_user_configured"):
    JuliaEnv().update()

from .pennylane_converter import PennylaneConverter
from .snowflurry_device import SnowflurryQubitDevice
