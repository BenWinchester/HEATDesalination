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

__all__ = ("InputFileError", "NAME", "reduced_temperature")

# NAME:
#   Keyword for parsing the name of the object.
NAME: str = "name"

# ZERO_CELCIUS_OFFSET:
#   Keyword for the offset of Kelvin to Celcius.
ZERO_CELCIUS_OFFSET: float = 273.15


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
