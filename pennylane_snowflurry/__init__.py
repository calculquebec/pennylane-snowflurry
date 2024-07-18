from .pennylane_converter import PennylaneConverter
from .snowflurry_device import SnowflurryQubitDevice
from .julia_setup import JuliaEnv
from configparser import RawConfigParser

# Load the configuration file
config = RawConfigParser()
config.read("env_config.ini")

# Check if the Julia environment should be updated
# by user or by default

if not config.getboolean("JULIA", "is_user_configured"):
    JuliaEnv().update()
