#!/usr/bin/python3.10
########################################################################################
# __utils__.py - The utility module for HEAT-Desalination simulation and optimisation. #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 14/10/2022                                                             #
# License: MIT, Open-source                                                            #
# For more information, contact: benedict.winchester@gmail.com                         #
########################################################################################

"""
__utils__.py - The utility module for the HEATDeslination program.

"""

import dataclasses
import logging
import os

from logging import Logger
from typing import Any, Dict, List, Union

import yaml

__all__ = (
    "AVERAGE_IRRADIANCE_DAY",
    "InputFileError",
    "LATITUDE",
    "LOGGER_DIRECTORY",
    "LONGITUDE",
    "MAXIMUM_IRRADIANCE_DAY",
    "NAME",
    "read_yaml",
    "reduced_temperature",
    "Scenario",
)

# AMBIENT_TEMPERATURE:
#   Keyword for the ambient temperature.
AMBIENT_TEMPERATURE: str = "ambient_temperature"

# AUTO_GENERATED_FILES_DIRECTORY:
#   Name of the directory into which auto-generated files should be saved.
AUTO_GENERATED_FILES_DIRECTORY: str = "auto_generated"

# AVERAGE_IRRADIANCE_DAY:
#   Keyword for saving the average weather profiles for the location.
AVERAGE_IRRADIANCE_DAY: str = "average_weather_conditions"

# LATITUDE:
#   Keyword for latitude.
LATITUDE: str = "latitude"

# LOGGER_DIRECTORY:
#   Directory for storing logs.
LOGGER_DIRECTORY: str = "logs"

# LONGITUDE:
#   Keyword for longitude.
LONGITUDE: str = "longitude"

# MAXIMUM_IRRADIANCE_DAY:
#   Keyword for saving the weather conditions for the day of maximum irradiance.
MAXIMUM_IRRADIANCE_DAY: str = "maximum_irradiance_weather_conditions"

# MINIMUM_IRRADIANCE_DAY:
#   Keyword for saving the weather conditions for the day of minimum irradiance.
MINIMUM_IRRADIANCE_DAY: str = "minimum_irradiance_weather_conditions"

# NAME:
#   Keyword for parsing the name of the object.
NAME: str = "name"

# OPTIMUM_TILT_ANGLE:
#   Keyword for the optimum tilt angle of the panel.
OPTIMUM_TILT_ANGLE: str = "optimum_tilt_angle"

# SOLAR_ELEVATION:
#   Keyword for the solar elevation.
SOLAR_ELEVATION: str = "solar_elevation"

# SOLAR_IRRADIANCE:
#   Keyword for the solar irradiance.
SOLAR_IRRADIANCE: str = "irradiance"

# WIND_SPEED:
#   Keyword for the wind speed.
WIND_SPEED: str = "wind_speed"

# ZERO_CELCIUS_OFFSET:
#   Keyword for the offset of Kelvin to Celcius.
ZERO_CELCIUS_OFFSET: float = 273.15


def get_logger(logger_name: str, verbose: bool = False) -> logging.Logger:
    """
    Set-up and return a logger.
    Inputs:
        - logger_name:
            The name for the logger, which is also used to denote the filename with a
            "<logger_name>.log" format.
        - verbose:
            Whether the log level should be verbose (True) or standard (False).
    Outputs:
        - The logger for the component.
    """

    # Create a logger and logging directory.
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    os.makedirs(LOGGER_DIRECTORY, exist_ok=True)

    # Create a formatter.
    formatter = logging.Formatter(
        "%(asctime)s: %(name)s: %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %I:%M:%S %p",
    )

    # Create a console handler.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(formatter)

    # Delete the existing log if there is one already.
    if os.path.isfile(os.path.join(LOGGER_DIRECTORY, f"{logger_name}.log")):
        os.remove(os.path.join(LOGGER_DIRECTORY, f"{logger_name}.log"))

    # Create a file handler.
    file_handler = logging.FileHandler(
        os.path.join(LOGGER_DIRECTORY, f"{logger_name}.log")
    )
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger.
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


class InputFileError(Exception):
    """Raised when there is an error in an input file."""

    def __init__(self, input_file: str, msg: str) -> None:
        """
        Instantiate a :class:`InputFileError` instance.
        Inputs:
            - input_file:
                The name of the input file which contained the invalid data.
            - msg:
                The error message to append.
        """

        super().__init__(
            f"Error parsing input file '{input_file}', invalid data in file: {msg}"
        )


def read_yaml(
    filepath: str, logger: Logger
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Reads a YAML file and returns the contents.
    """

    # Process the new-location data.
    try:
        with open(filepath, "r") as filedata:
            file_contents: Union[Dict[str, Any], List[Dict[str, Any]]] = yaml.safe_load(
                filedata
            )
    except FileNotFoundError:
        logger.error(
            "The file specified, %s, could not be found. "
            "Ensure that you run the new-locations script from the workspace root.",
            filepath,
        )
        raise
    return file_contents


def reduced_temperature(
    ambient_temperature: float, average_temperature: float, solar_irradiance: float
) -> float:
    """
    Computes the reduced temperature of the collector.

    NOTE: The ambient temperature and average temperature need to be measured in the
    same units, whether it's Kelvin or Celcius, but it does not matter which of these
    two is used.

    Inputs:
        - ambient_temperature:
            The ambient temperature surrounding the collector.
        - average_temperature:
            The average temperature of the collector.
        - solar_irradiance:
            The solar irradiance, measured in Watts per meter squared.

    Outputs:
        The reduced temperature of the collector in Kelvin meter squared per Watt.

    """

    return (average_temperature - ambient_temperature) / solar_irradiance


@dataclasses.dataclass
class Scenario:
    """
    Represents the scenario being modelled.

    .. attribute:: plant
        The name of the desalination plant being modelled.

    .. attribute:: pv
        Whether PV panels are being included.

    .. attribute:: pv_panel_name
        The name of the PV panel being considered.

    .. attribute:: pv_t
        Whether PV-T panels are being included.

    .. attribute:: pv_t_panel_name
        The name of the PV-T panel being considered.

    .. attribute:: solar_thermal
        Whether solar-thermal panels are being included.

    .. attribute:: solar_thermal_panel_name
        The name of the solar-thermal panel being considered.

    """

    plant: str
    _pv: Union[bool, str]
    _pv_t: Union[bool, str]
    _solar_thermal: Union[bool, str]

    @property
    def pv(self) -> bool:  # pylint: disable=invalid-name
        """
        Whether PV panels are being included.

        Outputs:
            Whether PV panels are being included (True) or not (False) in the modelling.

        """

        return not isinstance(self._pv, bool)

    @property
    def pv_panel_name(self) -> str:
        """
        Returns the name of the PV panel being modelled.

        Outputs:
            The name of the PV panel being modelled.

        Raises:
            Exception:
                Raised if the PV panel name is requested but the PV panel is not being
                included in the modeling.

        """

        if isinstance(self._pv, str):
            return self._pv

        raise Exception(
            "PV panel name requested but PV panels are not activated in the scenario."
        )

    @property
    def pv_t(self) -> bool:
        """
        Whether PV-T panels are being included.

        Outputs:
            Whether PV-T panels are being included (True) or not (False) in the
            modelling.

        """

        return not isinstance(self._pv_t, bool)

    @property
    def pv_t_panel_name(self) -> str:
        """
        Returns the name of the PV-T panel being modelled.

        Outputs:
            The name of the PV-T panel being modelled.

        Raises:
            Exception:
                Raised if the PV-T panel name is requested but the PV-T panel is not
                being included in the modeling.

        """

        if isinstance(self._pv_t, str):
            return self._pv_t

        raise Exception(
            "PV-T panel name requested but PV-T panels are not activated in the "
            "scenario."
        )

    @property
    def solar_thermal(self) -> bool:
        """
        Whether solar-thermal panels are being included.

        Outputs:
            Whether solar-thermal panels are being included (True) or not (False) in the
            modelling.

        """

        return not isinstance(self._solar_thermal, bool)

    @property
    def solar_thermal_panel_name(self) -> str:
        """
        Returns the name of the solar-thermal panel being modelled.

        Outputs:
            The name of the solar-thermal panel being modelled.

        Raises:
            Exception:
                Raised if the solar-thermal panel name is requested but the
                solar-thermal panel is not being included in the modeling.

        """

        if isinstance(self._pv_t, str):
            return self._pv_t

        raise Exception(
            "Solar-thermal panel name requested but solar-thermal panels are not "
            "activated in the scenario."
        )
