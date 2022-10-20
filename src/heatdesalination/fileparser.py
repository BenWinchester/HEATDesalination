#!/usr/bin/python3.10
########################################################################################
# fileparser.py - The file-parsing module                                              #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 14/10/2022                                                             #
# License: MIT, Open-source                                                            #
# For more information, contact: benedict.winchester@gmail.com                         #
########################################################################################

"""
fileparser.py - The file parser module for the HEATDeslination program.

"""

import os

from logging import Logger
from typing import Dict

import json

from .__utils__ import (
    AMBIENT_TEMPERATURE,
    AUTO_GENERATED_FILES_DIRECTORY,
    NAME,
    SOLAR_IRRADIANCE,
    ProfileType,
    read_yaml,
    Scenario,
    TIMEZONE
)
from .plant import DesalinationPlant

__all__ = ("parse_input_files",)

# DESLINATION_PLANT_INPUTS:
#   The name of the desalination plant inputs file.
DESALINATION_PLANT_INPUTS: str = "plants.yaml"

# DESALINATION_PLANTS:
#   Keyword for desalination plants.
DESALINATION_PLANTS: str = "desalination_plants"

# HEAT_EXCHANGER_EFFICIENCY:
#   Keyword for parsing the heat capacity of the heat exchangers.
HEAT_EXCHANGER_EFFICIENCY: str = "heat_exchanger_efficiency"

# HTF_HEAT_CAPACITY:
#   Keyword for parsing the htf heat capacity.
HTF_HEAT_CAPACITY: str = "htf_heat_capacity"

# INPUTS_DIRECTORY:
#   The name of the inputs directory.
INPUTS_DIRECTORY: str = "inputs"

# PLANT:
#   Keyword for parsing the desalination plant name.
PLANT: str = "plant"

# PV:
#   Keyword for parsing the PV panel name.
PV: str = "pv"

# PV_T:
#   Keyword for parsing the PV-T panel name.
PV_T: str = "pv_t"

# SCENARIO_INPUTS:
#   Keyword for scenario inputs file.
SCENARIO_INPUTS: str = "scenarios.yaml"

# SCENARIOS:
#   Keyword for scenario information.
SCENARIOS: str = "scenarios"

# SOLAR_THERMAL:
#   Keyword for parsing the solar-thermal collector name.
SOLAR_THERMAL: str = "solar_thermal"


def parse_input_files(
    location: str, logger: Logger, scenario_name: str, start_hour: int
):
    """
    Parses the various input files.

    Inputs:
        - location:
            The name of the location being used for which weather profiles should be
            used.
        - logger:
            The :class:`logging.Logger` to use.
        - scenario_name:
            The name of the scenario to use.
        - start_hour:
            The start hour for the plant's operation.

    Outputs:
        - desalination_plant:
            The :class:`DesalinationPlant` to use for the modelling.
        - pv_panels:
            A `list` of :class:`PVPanel` instances available for modelling.
        - pv_t:
            A `list` of :class:`HybridPVTPanel` instances available for modelling.
        - scenario:
            The :class:`Scenario` to use for the mode
        - solar_thermal:
            A `list` of :class:`SolarThermalPanel` instances available for modelling.
        - weather_data:
            The weather data for the modelling.

    """

    # Parse the scenario.
    scenario_inputs = read_yaml(os.path.join(INPUTS_DIRECTORY, SCENARIO_INPUTS), logger)
    scenarios = [
        Scenario(
            entry[HEAT_EXCHANGER_EFFICIENCY],
            entry[HTF_HEAT_CAPACITY],
            entry[NAME],
            entry[PLANT],
            entry[PV],
            entry[PV_T],
            entry[SOLAR_THERMAL],
        )
        for entry in scenario_inputs[SCENARIOS]
    ]
    try:
        scenario = [entry for entry in scenarios if entry.name == scenario_name][0]
    except IndexError:
        logger.error("Could not find scenario '%s' in input file.", scenario_name)
        raise

    # Parse the desalination plant inputs.
    desalination_inputs = read_yaml(
        os.path.join(INPUTS_DIRECTORY, DESALINATION_PLANT_INPUTS), logger
    )
    desalination_plants = [
        DesalinationPlant.from_dict(entry, logger, start_hour)
        for entry in desalination_inputs[DESALINATION_PLANTS]
    ]
    try:
        desalination_plant = [
            entry for entry in desalination_plants if entry.name == scenario.plant
        ][0]
    except IndexError:
        logger.error("Could not find plant '%s' in input file.", scenario.plant)
        raise

    # Parse the solar panels.

    # Parse the weather data.
    with open(
        os.path.join(AUTO_GENERATED_FILES_DIRECTORY, f"{location}.json"),
        "r",
        encoding="UTF-8",
    ) as f:
        weather_data = json.load(f)

    ambient_temperatures: Dict[ProfileType, Dict[int, float]] = {}
    solar_irradiances: Dict[ProfileType, Dict[int, float]] = {}
    time_difference: int = weather_data[TIMEZONE]

    import pdb

    pdb.set_trace()

    for profile_type in ProfileType:
        ambient_temperatures[profile_type] = {
            (int(key) + time_difference) % 24: value
            for key, value in weather_data[profile_type.value][
                AMBIENT_TEMPERATURE
            ].items()
        }
        solar_irradiances[profile_type] = {
            (int(key) + time_difference) % 24: value
            for key, value in weather_data[profile_type.value][SOLAR_IRRADIANCE].items()
        }

    # Return the information.
    return (
        ambient_temperatures,
        buffer_tank,
        desalination_plant,
        hybrid_pvt_panel,
        pv_panel,
        scenario,
        solar_irradiances,
        solar_thermal_collector,
    )
