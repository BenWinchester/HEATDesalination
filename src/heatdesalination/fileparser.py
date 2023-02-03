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
from typing import Tuple

import json

from .__utils__ import (
    AMBIENT_TEMPERATURE,
    AUTO_GENERATED_FILES_DIRECTORY,
    GridCostScheme,
    NAME,
    OptimisationParameters,
    ProfileType,
    read_yaml,
    Scenario,
    SOLAR_IRRADIANCE,
    WIND_SPEED,
    ZERO_CELCIUS_OFFSET,
)
from .heat_pump import HeatPump
from .plant import DesalinationPlant
from .solar import (
    COLLECTOR_FROM_TYPE,
    HybridPVTPanel,
    PVPanel,
    SolarPanelType,
    SolarThermalPanel,
)
from .storage.storage_utils import Battery, HotWaterTank

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

# FRACTIONAL_BATTERY_COST_CHANGE:
#   Keyword for the fractional change in the price of the batteries installed.
FRACTIONAL_BATTERY_COST_CHANGE: str = "fractional_battery_cost_change"

# FRACTIONAL_GRID_COST_CHANGE:
#   Keyword for the fractional change in the price of grid electricity.
FRACTIONAL_GRID_COST_CHANGE: str = "fractional_grid_cost_change"

# FRACTIONAL_HEAT_PUMP_COST_CHANGE:
#   Keyword for the fractional change in the cost of the heat pump(s) installed.
FRACTIONAL_HEAT_PUMP_COST_CHANGE: str = "fractional_heat_pump_cost_change"

# FRACTIONAL_HW_TANK_COST_CHANGE:
#   Keyword for the fractional change in the cost of the hot-water tanks installed.
FRACTIONAL_HW_TANK_COST_CHANGE: str = "fractional_hw_tank_cost_change"

# FRACTIONAL_INVERTER_COST_CHANGE:
#   Keyword for the fractional change in the price of the inverter(s) installed.
FRACTIONAL_INVERTER_COST_CHANGE: str = "fractional_inverter_cost_change"

# FRACTIONAL_PV_COST_CHANGE:
#   Keyword for the fractional change in the price of grid electricity.
FRACTIONAL_PV_COST_CHANGE: str = "fractional_pv_cost_change"

# FRACTIONAL_PV_T_COST_CHANGE:
#   Keyword for the fractional change in the price of grid electricity.
FRACTIONAL_PV_T_COST_CHANGE: str = "fractional_pvt_cost_change"

# FRACTIONAL_ST_COST_CHANGE:
#   Keyword for the fractional change in the price of grid electricity.
FRACTIONAL_ST_COST_CHANGE: str = "fractional_st_cost_change"

# GRID_COST_SCHEME:
#   Keyword for the name of the pricing scheme for the cost of grid (alternative/unmet)
# electricity.
GRID_COST_SCHEME: str = "grid_cost_scheme"

# HEAT_EXCHANGER_EFFICIENCY:
#   Keyword for parsing the heat capacity of the heat exchangers.
HEAT_EXCHANGER_EFFICIENCY: str = "heat_exchanger_efficiency"

# HEAT_PUMP:
#   Keyword for parsing the name of the installed heat pump.
HEAT_PUMP: str = "heat_pump"

# HEAT_PUMPS:
#   Keyword for parsing the heat-pump information.
HEAT_PUMPS: str = "heat_pumps"

# HEAT_PUMP_INPUTS:
#   The name of the heat-pumps input file.
HEAT_PUMP_INPUTS: str = "heat_pumps.yaml"

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

# INVERTER_COST:
#   Keyword for the cost of the inverter installed.
INVERTER_COST: str = "inverter_cost"

# INVERTER_LIFETIME:
#   Keyword for the lifetime of the inverter installed.
INVERTER_LIFETIME: str = "inverter_lifetime"

# OPTIMISATION_INPUTS:
#   The name of the optimisation inputs file.
OPTIMISATION_INPUTS: str = "optimisations.yaml"

# OPTIMISATIONS:
#   Keyword for the optimisations.
OPTIMISATIONS: str = "optimisations"

# PLANT:
#   Keyword for parsing the desalination plant name.
PLANT: str = "plant"

# PV:
#   Keyword for parsing the PV panel name.
PV: str = "pv"

# PV_DEGRADATION_RATE:
#   Keyword for parsing the annual degradation rate of the PV panels.
PV_DEGRADATION_RATE: str = "pv_degradation_rate"

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
    location: str, logger: Logger, scenario_name: str, start_hour: int | None
) -> Tuple[
    dict[ProfileType, dict[int, float]],
    Battery,
    HotWaterTank,
    DesalinationPlant,
    HeatPump,
    HybridPVTPanel | None,
    list[OptimisationParameters],
    PVPanel | None,
    Scenario,
    dict[ProfileType, dict[int, float]],
    SolarThermalPanel | None,
    dict[ProfileType, dict[int, float]],
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
            The start hour for the plant's operation or `None` if running an
            optimisation and the start-hour is optimisable.

    Outputs:
        - ambient_temperatures:
            The ambient temperature measured in degrees Kelvin keyed by profile type.
        - battery:
            The :class:`Battery` to use for the modelling.
        - desalination_plant:
            The :class:`DesalinationPlant` to use for the modelling.
        - heat_pump:
            The :class:`HeatPump` to use for the modelling.
        - hybrid_pv_t_panel:
            The :class:`HybridPVTPanel` to use for the modelling.
        - optimisations:
            The `list` of :class:`OptimisationParameters` instances describing the
            optimisations that should be carried out.
        - pv_panel:
            The :class:`PVPanel` to use for the modelling.
        - scenario:
            The :class:`Scenario` to use for the mode
        - solar_irradiances:
            The solar irradiance keyed by profile type.
        - solar_thermal_collector:
            The :class:`SolarThermalCollector` to use for the modelling.
        - wind_speeds:
            The wind speeds, keyed by profile type.

    """

    # Parse the scenario.
    scenario_inputs = read_yaml(os.path.join(INPUTS_DIRECTORY, SCENARIO_INPUTS), logger)
    scenarios = [
        Scenario(
            entry[BATTERY],
            GridCostScheme(entry[GRID_COST_SCHEME]),
            entry[HEAT_EXCHANGER_EFFICIENCY],
            entry[HEAT_PUMP],
            entry[HOT_WATER_TANK],
            entry[HTF_HEAT_CAPACITY],
            entry[INVERTER_COST],
            entry[INVERTER_LIFETIME],
            entry[NAME],
            entry[PLANT],
            entry[PV_DEGRADATION_RATE],
            entry[PV],
            entry[PV_T],
            entry[SOLAR_THERMAL],
            entry.get(FRACTIONAL_BATTERY_COST_CHANGE, 0),
            entry.get(FRACTIONAL_GRID_COST_CHANGE, 0),
            entry.get(FRACTIONAL_HEAT_PUMP_COST_CHANGE, 0),
            entry.get(FRACTIONAL_HW_TANK_COST_CHANGE, 0),
            entry.get(FRACTIONAL_INVERTER_COST_CHANGE, 0),
            entry.get(FRACTIONAL_PV_COST_CHANGE, 0),
            entry.get(FRACTIONAL_PV_T_COST_CHANGE, 0),
            entry.get(FRACTIONAL_ST_COST_CHANGE, 0),
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

    # Parse the heat-pump inputs.
    heat_pump_inputs = read_yaml(
        os.path.join(INPUTS_DIRECTORY, HEAT_PUMP_INPUTS), logger
    )
    heat_pumps = [HeatPump(**entry) for entry in heat_pump_inputs[HEAT_PUMPS]]
    try:
        heat_pump = [entry for entry in heat_pumps if entry.name == scenario.heat_pump][
            0
        ]
    except IndexError:
        logger.error(
            "Could not find heat pump '%s' in input file. Valid pumps: %s",
            scenario.heat_pump,
            ", ".join(pump.name for pump in heat_pumps),
        )
        raise

    # Parse the optimisation inputs
    optimisation_inputs = read_yaml(
        os.path.join(INPUTS_DIRECTORY, OPTIMISATION_INPUTS), logger
    )
    try:
        optimisations: list[OptimisationParameters] = [
            OptimisationParameters.from_dict(logger, entry)
            for entry in optimisation_inputs[OPTIMISATIONS]
        ]
    except KeyError:
        logger.error("Missing information in optimisation inputs file.")
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
            hybrid_pv_t_panel: HybridPVTPanel | None = [
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
        hybrid_pv_t_panel = None

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

    ambient_temperatures: dict[ProfileType, dict[int, float]] = {}
    solar_irradiances: dict[ProfileType, dict[int, float]] = {}
    wind_speeds: dict[ProfileType, dict[int, float]] = {}

    # Extend profiles to 25 hours.
    for profile_type in ProfileType:
        ambient_temperatures[profile_type] = {
            int(key) % 24: value + ZERO_CELCIUS_OFFSET
            for key, value in weather_data[profile_type.value][
                AMBIENT_TEMPERATURE
            ].items()
        }
        solar_irradiances[profile_type] = {
            int(key) % 24: value
            for key, value in weather_data[profile_type.value][SOLAR_IRRADIANCE].items()
        }
        wind_speeds[profile_type] = {
            int(key) % 24: value
            for key, value in weather_data[profile_type.value][WIND_SPEED].items()
        }

    # Return the information.
    return (
        ambient_temperatures,
        battery,
        buffer_tank,
        desalination_plant,
        heat_pump,
        hybrid_pv_t_panel,
        optimisations,
        pv_panel,
        scenario,
        solar_irradiances,
        solar_thermal_collector,
        wind_speeds,
    )
