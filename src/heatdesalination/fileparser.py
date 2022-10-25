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
from typing import Dict, Tuple

import json

from .storage.storage_utils import Battery, HotWaterTank

from .solar import (
    COLLECTOR_FROM_TYPE,
    HybridPVTPanel,
    PVPanel,
    SolarPanelType,
    SolarThermalPanel,
)

from .__utils__ import (
    AMBIENT_TEMPERATURE,
    AUTO_GENERATED_FILES_DIRECTORY,
    NAME,
    SOLAR_IRRADIANCE,
    ProfileType,
    read_yaml,
    Scenario,
    TIMEZONE,
)
from .plant import DesalinationPlant

__all__ = ("parse_input_files",)

# BATTERIES:
#   Keyword for batteries.
BATTERIES: str = "batteries"

# BATTERY:
#   Keyword for battery.
BATTERY: str = "battery"

# DESLINATION_PLANT_INPUTS:
#   The name of the desalination plant inputs file.
DESALINATION_PLANT_INPUTS: str = "plants.yaml"

# DESALINATION_PLANTS:
#   Keyword for desalination plants.
DESALINATION_PLANTS: str = "desalination_plants"

# HEAT_EXCHANGER_EFFICIENCY:
#   Keyword for parsing the heat capacity of the heat exchangers.
HEAT_EXCHANGER_EFFICIENCY: str = "heat_exchanger_efficiency"

# HOT_WATER_TANK:
#   Keyword for hot-water tank.
HOT_WATER_TANK: str = "hot_water_tank"

# HOT_WATER_TANKS:
#   Keyword for hot-water tanks.
HOT_WATER_TANKS: str = "hot_water_tanks"

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

# SOLAR_COLLECTORS:
#   Keyword for solar collectors.
SOLAR_COLLECTORS: str = "solar_collectors"

# SOLAR_INPUTS:
#   The name of the solar-collectors input file.
SOLAR_INPUTS: str = "solar.yaml"

# SOLAR_THERMAL:
#   Keyword for parsing the solar-thermal collector name.
SOLAR_THERMAL: str = "solar_thermal"

# STORAGE_INPUTS:
#   Keyword for storage inputs file.
STORAGE_INPUTS: str = "storage.yaml"

# TYPE:
#   Keyword for the type of solar collector.
TYPE: str = "type"


def parse_input_files(
    location: str, logger: Logger, scenario_name: str, start_hour: int
) -> Tuple[
    Dict[ProfileType, Dict[int, float]],
    Battery,
    HotWaterTank,
    DesalinationPlant,
    HybridPVTPanel | None,
    PVPanel | None,
    Scenario,
    Dict[ProfileType, Dict[int, float]],
    SolarThermalPanel | None,
]:
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
        - ambient_temperatures:
            The ambient temperature keyed by profile type for:
            - the day with maximum irradiance,
            - the day with minimum irradiance,
            - an average over all days.
        - battery:
            The :class:`Battery` to use for the modelling.
        - desalination_plant:
            The :class:`DesalinationPlant` to use for the modelling.
        - hybrid_pvt_panel:
            The :class:`HybridPVTPanel` to use for the modelling.
        - pv_panel:
            The :class:`PVPanel` to use for the modelling.
        - scenario:
            The :class:`Scenario` to use for the mode
        - solar_irradiances:
            The solar irradiance keyed by profile type for:
            - the day with maximum irradiance,
            - the day with minimum irradiance,
            - an average over all days.
        - solar_thermal_collector:
            The :class:`SolarThermalCollector` to use for the modelling.

    """

    # Parse the scenario.
    scenario_inputs = read_yaml(os.path.join(INPUTS_DIRECTORY, SCENARIO_INPUTS), logger)
    scenarios = [
        Scenario(
            entry[BATTERY],
            entry[HEAT_EXCHANGER_EFFICIENCY],
            entry[HOT_WATER_TANK],
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
    solar_inputs = read_yaml(os.path.join(INPUTS_DIRECTORY, SOLAR_INPUTS), logger)
    try:
        solar_collectors = [
            COLLECTOR_FROM_TYPE[SolarPanelType(entry[TYPE])].from_dict(logger, entry)
            for entry in solar_inputs[SOLAR_COLLECTORS]
        ]
    except KeyError as exception:
        logger.error(
            "Could not parse all solar-panel inputs, potentially missing `type`: %s",
            str(exception),
        )
        raise
    except ValueError as exception:
        logger.error("Invalid panel type: %s", str(exception))
        raise

    # Determine the PV, PV-T and solar-thermal panels selected based on the scenario.
    if scenario.pv:
        try:
            pv_panel: PVPanel | None = [
                entry
                for entry in solar_collectors
                if entry.name == scenario.pv_panel_name
            ][0]
        except IndexError:
            logger.error(
                "Could not find PV panel '%s' in input file.", scenario.pv_panel_name
            )
            raise
    else:
        pv_panel = None

    if scenario.pv_t:
        try:
            hybrid_pvt_panel: HybridPVTPanel | None = [
                entry
                for entry in solar_collectors
                if entry.name == scenario.pv_t_panel_name
            ][0]
        except IndexError:
            logger.error(
                "Could not find PV-T panel '%s' in input file.",
                scenario.pv_t_panel_name,
            )
            raise
    else:
        hybrid_pvt_panel = None

    if scenario.solar_thermal:
        try:
            solar_thermal_collector: SolarThermalPanel | None = [
                entry
                for entry in solar_collectors
                if entry.name == scenario.solar_thermal_panel_name
            ][0]
        except IndexError:
            logger.error(
                "Could not find solar-thermal collector '%s' in input file.",
                scenario.solar_thermal,
            )
            raise
    else:
        solar_thermal_collector = None

    # Parse the batteries and buffer tank.
    storage_inputs = read_yaml(os.path.join(INPUTS_DIRECTORY, STORAGE_INPUTS), logger)
    batteries = [Battery.from_dict(entry) for entry in storage_inputs[BATTERIES]]
    try:
        battery: Battery = [
            entry for entry in batteries if entry.name == scenario.battery
        ][0]
    except IndexError:
        logger.error(
            "Could not find battery '%s' in input file.",
            scenario.solar_thermal,
        )
        raise
    tanks = [HotWaterTank.from_dict(entry) for entry in storage_inputs[HOT_WATER_TANKS]]
    try:
        buffer_tank: HotWaterTank = [
            entry for entry in tanks if entry.name == scenario.hot_water_tank
        ][0]
    except IndexError:
        logger.error(
            "Could not find hot-water tank '%s' in input file.",
            scenario.solar_thermal,
        )
        raise
    # Parse the weather data.
    with open(
        os.path.join(AUTO_GENERATED_FILES_DIRECTORY, f"{location}.json"),
        "r",
        encoding="UTF-8",
    ) as weather_data_file:
        weather_data = json.load(weather_data_file)

    ambient_temperatures: Dict[ProfileType, Dict[int, float]] = {}
    solar_irradiances: Dict[ProfileType, Dict[int, float]] = {}
    time_difference: int = weather_data[TIMEZONE]

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
        battery,
        buffer_tank,
        desalination_plant,
        hybrid_pvt_panel,
        pv_panel,
        scenario,
        solar_irradiances,
        solar_thermal_collector,
    )
