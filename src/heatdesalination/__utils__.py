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

import argparse
import dataclasses
import enum
import logging
import os

from collections import defaultdict
from logging import Logger
from typing import Any, DefaultDict, Dict, List, NamedTuple, Tuple

import json
import yaml

import numpy as np
import pandas as pd

__all__ = (
    "AMBIENT_TEMPERATURE",
    "AREA",
    "CLI_TO_PROFILE_TYPE",
    "COST",
    "CostableComponent",
    "DONE",
    "FAILED",
    "FlowRateError",
    "HPCSimulation",
    "InputFileError",
    "LATITUDE",
    "LOGGER_DIRECTORY",
    "LONGITUDE",
    "NAME",
    "parse_hpc_args_and_runs",
    "ProfileDegradationType",
    "ProfileType",
    "read_yaml",
    "reduced_temperature",
    "ResourceType",
    "Scenario",
    "Solution",
    "TIMEZONE",
)

# AMBIENT_TEMPERATURE:
#   Keyword for the ambient temperature.
AMBIENT_TEMPERATURE: str = "ambient_temperature"

# AREA:
#   Keyword for the area of the panel.
AREA: str = "area"

# AUTO_GENERATED_FILES_DIRECTORY:
#   Name of the directory into which auto-generated files should be saved.
AUTO_GENERATED_FILES_DIRECTORY: str = "auto_generated"

# BOUNDS:
#   Keyword for the bounds on the optimisation.
BOUNDS: str = "bounds"

# CONSTRAINTS:
#   Keyword for the constraints on the optimisation.
CONSTRAINTS: str = "constraints"

# COST:
#   Keyword for the cost of a component.
COST: str = "cost"

# CRITERION:
#   Keyword for the criterion on the optimisation.
CRITERION: str = "criterion"

# DAYS_PER_YEAR:
#   The number of days per year.
DAYS_PER_YEAR: float = 365.25

# DONE:
#   Keyword for "done".
DONE: str = "[  DONE  ]"

# FAILED:
#   Keyword for "failed".
FAILED: str = "[ FAILED ]"

# FIXED:
#   Keyword for the fixed capacity of an optimisable component.
FIXED: str = "fixed"

# HEAT_CAPACITY_OF_WATER:
#   The heat capacity of water, measured in Joules per kilogram Kelvin.
HEAT_CAPACITY_OF_WATER: float = 4182

# INITIAL_GUESS:
#   Keyword for the initial guess.
INITIAL_GUESS: str = "initial_guess"

# LATITUDE:
#   Keyword for latitude.
LATITUDE: str = "latitude"

# LOGGER_DIRECTORY:
#   Directory for storing logs.
LOGGER_DIRECTORY: str = "logs"

# LONGITUDE:
#   Keyword for longitude.
LONGITUDE: str = "longitude"

# MAX:
#   Keyword for max value for optimisation.
MAX: str = "max"

# MIN:
#   Keyword for min value for optimisation.
MIN: str = "min"

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

# TIMEZONE:
#   Keyword for parsing timezone.
TIMEZONE: str = "timezone"

# WALLTIME:
#   The keyword for walltime information.
WALLTIME: str = "walltime"

# WIND_SPEED:
#   Keyword for the wind speed.
WIND_SPEED: str = "wind_speed"

# ZERO_CELCIUS_OFFSET:
#   Keyword for the offset of Kelvin to Celcius.
ZERO_CELCIUS_OFFSET: float = 273.15


class CostableComponent:
    """
    Represents a costable component.

    .. attribute:: cost
        The cost of the component, per unit component.

    """

    def __init__(self, cost: float) -> None:
        """
        Instantiate the costable component.

        Inputs:
            - cost:
                The cost of the component, per unit component.

        """

        self.cost = cost


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

    logger.addHandler(console_handler)

    # Delete the existing log if there is one already.
    # if os.path.isfile(
    #     (logger_filepath := os.path.join(LOGGER_DIRECTORY, f"{logger_name}.log"))
    # ):
    #     try:
    #         os.remove(logger_filepath)
    #     except FileNotFoundError:
    #         pass

    # Create a file handler.
    # if not os.path.isfile(
    #     (logger_filename := os.path.join(LOGGER_DIRECTORY, f"{logger_name}.log"))
    # ):
    #     file_handler = logging.FileHandler(
    #         os.path.join(LOGGER_DIRECTORY, f"{logger_name}.log")
    #     )
    #     file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    #     file_handler.setFormatter(formatter)
    #
    #     # Add the file handler to the logger.
    #     logger.addHandler(file_handler)

    return logger


class FlowRateError(Exception):
    """Raised when there is a mismatch in the flow rates.."""

    def __init__(self, collector_name: str, msg: str) -> None:
        """
        Instantiate a :class:`FlowRateError` instance.

        Inputs:
            - collector_name:
                The name of the collector for which a flow-rate mismatch has occurred.
            - msg:
                The error message to append.

        """

        super().__init__(f"Flow-rate mismatch for collector '{collector_name}': {msg}")


@dataclasses.dataclass
class HPCSimulation:
    """
    Contains information about a run to carry out on the HPC.

    - location:
        The name of the location to consider.

    - output:
        The name of the output file to use.

    - simulations:
        The name of the input simulations file to use.

    - walltime:
        The walltime to use if specified.

    """

    location: str
    output: str
    simulation: str
    walltime: int | None = None


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


class OptimisableComponent(enum.Enum):
    """
    Keeps track of components that can be optimised.

    - BATTERY_CAPACITY:
        The capacity of the electrical storage system in kWh.

    - BUFFER_TANK_CAPACITY:
        The capacity in kg of the buffer tank.

    - MASS_FLOW_RATE:
        The HTF mass flow rate.

    - PV:
        The PV-panel capacity.

    - PV_T:
        The PV-T system size.

    - START_HOUR:
        The start hour for the plant.

    - SOLAR_THERMAL:
        The solar-thermal system size.

    - STORAGE:
        The battery capacity.

    """

    BATTERY_CAPACITY: str = "battery_capacity"
    BUFFER_TANK_CAPACITY: str = "buffer_tank_capacity"
    MASS_FLOW_RATE: str = "mass_flow_rate"
    PV: str = "pv"
    PV_T: str = "pv_t"
    START_HOUR: str = "start_hour"
    SOLAR_THERMAL: str = "st"


class OptimisationMode(enum.Enum):
    """
    The mode of optimisation being carried out.

    - MAXIMISE:
        Maximise the target criterion.

    - MINIMISE:
        Minimise the target criterion.

    """

    MAXIMISE: str = "maximise"
    MINIMISE: str = "minimise"


@dataclasses.dataclass
class OptimisationParameters:
    """
    Contains optimisation parameters.

    .. attribute:: bounds
        Bounds associated with the optimisation, stored as a mapping.

    .. attribute:: constraints
        Whether to use constraints on the optimisation (True) or not (False).

    .. attribute:: maximise
        Whether to maximise the target_criterion (True) or minimise it (False).

    .. attribute:: minimise
        Whether to minimise the target criterion (True) or maximise it (False).

    .. attribute:: optimisable_component_to_index
        Mapping between component and the index in the vector.

    .. attribute:: target_criterion
        The target criterion.

    """

    bounds: Dict[OptimisableComponent, Dict[str, float | None]]
    constraints: bool
    optimisable_component_to_index: Dict[OptimisableComponent, int]
    _criterion: Dict[str, OptimisationMode]

    @property
    def asdict(self) -> Dict[str, Any]:
        """
        Return a dictionary representing the class.

        Outputs:
            A dictionary representing the class.

        """

        return {
            BOUNDS: {key.value: value for key, value in self.bounds.items()},
            CONSTRAINTS: self.constraints,
        }

    @property
    def fixed_battery_capacity_value(self) -> float | None:
        """
        The initial guess for the `battery_capacity` if provided, else `None`.

        Outputs:
            - An initial guess for the battery capacity or `None` if not provided.

        """

        return self.bounds[OptimisableComponent.BATTERY_CAPACITY].get(FIXED, None)

    @property
    def fixed_buffer_tank_capacity_value(self) -> float | None:
        """
        The initial guess for the `buffer_tank_capacity` if provided, else `None`.

        Outputs:
            - An initial guess for the buffer tank value or `None` if not provided.

        """

        return self.bounds[OptimisableComponent.BUFFER_TANK_CAPACITY].get(FIXED, None)

    @property
    def fixed_mass_flow_rate_value(self) -> float | None:
        """
        The initial guess for the `mass_flow_rate` if provided, else `None`.

        Outputs:
            - An initial guess for the mass-flow rate or `None` if not provided.

        """

        return self.bounds[OptimisableComponent.MASS_FLOW_RATE].get(FIXED, None)

    @property
    def fixed_pv_value(self) -> float | None:
        """
        The initial guess for the `pv` if provided, else `None`.

        Outputs:
            - An initial guess for number of PV collectors installed or `None` if not
              provided.

        """

        return self.bounds[OptimisableComponent.PV].get(FIXED, None)

    @property
    def fixed_pv_t_value(self) -> float | None:
        """
        The initial guess for the `pv_t` if provided, else `None`.

        Outputs:
            - An initial guess for number of PV-T collectors installed or `None` if not
              provided.

        """

        return self.bounds[OptimisableComponent.PV_T].get(FIXED, None)

    @property
    def fixed_st_value(self) -> float | None:
        """
        The initial guess for the `st` if provided, else `None`.

        Outputs:
            - An initial guess for number of solar-thermal collectors installed or
              `None` if not provided.

        """

        return self.bounds[OptimisableComponent.SOLAR_THERMAL].get(FIXED, None)

    @property
    def fixed_start_hour_value(self) -> float | None:
        """
        The initial guess for the `start_hour` if provided, else `None`.

        Outputs:
            - An initial guess for the start hour of the plant operation or `None` if
              not provided.

        """

        return self.bounds[OptimisableComponent.START_HOUR].get(FIXED, None)

    def get_initial_guess_vector_and_bounds(
        self,
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Fetch the initial guess vector and bounds for the various parameters.

        Each entry which is optimisable should have a `min`, `max` and `initial_guess`
        entry. These are then returned for use within the optimisation.

        Outputs:
            - The initial guess vector,
            - A list of the bounds as (min, max) tuples.

        """

        # Instantiate variables.
        initial_guess_vector = []
        bounds = []

        # Append variables in the required order.
        if self.fixed_battery_capacity_value is None:
            initial_guess_vector.append(
                self.bounds[(component := OptimisableComponent.BATTERY_CAPACITY)][
                    INITIAL_GUESS
                ]
            )
            bounds.append((self.bounds[component][MIN], self.bounds[component][MAX]))
        if self.fixed_buffer_tank_capacity_value is None:
            initial_guess_vector.append(
                self.bounds[(component := OptimisableComponent.BUFFER_TANK_CAPACITY)][
                    INITIAL_GUESS
                ]
            )
            bounds.append((self.bounds[component][MIN], self.bounds[component][MAX]))
        if self.fixed_mass_flow_rate_value is None:
            initial_guess_vector.append(
                self.bounds[(component := OptimisableComponent.MASS_FLOW_RATE)][
                    INITIAL_GUESS
                ]
            )
            bounds.append((self.bounds[component][MIN], self.bounds[component][MAX]))
        if self.fixed_pv_value is None:
            initial_guess_vector.append(
                self.bounds[(component := OptimisableComponent.PV)][INITIAL_GUESS]
            )
            bounds.append((self.bounds[component][MIN], self.bounds[component][MAX]))
        if self.fixed_pv_t_value is None:
            initial_guess_vector.append(
                self.bounds[(component := OptimisableComponent.PV_T)][INITIAL_GUESS]
            )
            bounds.append((self.bounds[component][MIN], self.bounds[component][MAX]))
        if self.fixed_st_value is None:
            initial_guess_vector.append(
                self.bounds[(component := OptimisableComponent.SOLAR_THERMAL)][
                    INITIAL_GUESS
                ]
            )
            bounds.append((self.bounds[component][MIN], self.bounds[component][MAX]))
        if self.fixed_start_hour_value is None:
            initial_guess_vector.append(
                self.bounds[(component := OptimisableComponent.START_HOUR)][
                    INITIAL_GUESS
                ]
            )
            bounds.append((self.bounds[component][MIN], self.bounds[component][MAX]))

        return initial_guess_vector, bounds

    @property
    def maximise(self) -> bool:
        """
        Returns whether to maximise the target criterion.

        """

        if len(self._criterion) > 1:
            raise InputFileError(
                "optimisations.yaml",
                "Only one optimisation criterion can be specified.",
            )

        return list(self._criterion.values())[0] == OptimisationMode.MAXIMISE

    @property
    def minimise(self) -> bool:
        """
        Returns whether to minimise the target criterion.

        """

        return not self.maximise

    @property
    def target_criterion(self) -> str:
        """
        Return the target criterion name.

        Outputs:
            The target criterion name.

        """

        if len(self._criterion) > 1:
            raise InputFileError(
                "optimisations.yaml",
                "Only one optimisation criterion can be specified.",
            )

        return str(list(self._criterion.keys())[0])

    @classmethod
    def from_dict(cls, logger: Logger, optimisation_inputs: Dict[str, Any]) -> Any:
        """
        Instantiate a :class:`OptimisationParameters` instance based on the input data.

        Inputs:
            - logger:
                The logger to use for the run.
            - optimisation_inputs:
                The optimisation inputs for this entry.

        Outputs:
            A :class:`OptimisationParameters` instance.

        """

        try:
            bounds = {
                OptimisableComponent(key): value
                for key, value in optimisation_inputs[BOUNDS].items()
            }
        except KeyError:
            logger.error("Missing bounds information under keyword '%s'.", BOUNDS)
            raise InputFileError(
                "optimisation inputs", "Missing bounds information."
            ) from None
        except ValueError:
            logger.error(
                "Invalid component bounds specified, check optimisation inputs."
            )
            raise

        try:
            criterion = {
                key: OptimisationMode(value)
                for key, value in optimisation_inputs[CRITERION].items()
            }
        except KeyError:
            logger.error("Missing criterion information under keyword '%s'.", CRITERION)
            raise InputFileError(
                "optimisation inputs", "Missing criterion information."
            ) from None
        except ValueError:
            logger.error(
                "Invalid optimisation mode specified, check optimisation inputs."
            )
            raise

        # Determine the indicies in the vector of the optimisable components.
        index = 0
        optimisable_component_to_index: Dict[OptimisableComponent, int] = {}
        for component in OptimisableComponent:
            if bounds[component].get(FIXED, None) is not None:
                continue
            optimisable_component_to_index[component] = index
            index += 1

        return cls(
            bounds,
            optimisation_inputs[CONSTRAINTS],
            optimisable_component_to_index,
            criterion,
        )


class ProfileDegradationType(enum.Enum):
    """
    Denotes whether a profile is degraded.

    - DEGRADED:
        Denotes that the profile has been degraded.

    - UNDEGRADED:
        Denotes that the profile has not been degraded.

    """

    DEGRADED: str = "degraded"
    UNDEGRADED: str = "undegraded"


class ProfileType(enum.Enum):
    """
    Denotes which profile type is being considered.

    - AVERAGE:
        Denotes that the average profiles are being considered.

    - MAXIMUM:
        Denotes that the maximum profils are being considered.

    - MINIMUM:
        Denotes that the minimum profiles are being considered.

    """

    AVERAGE: str = "average_weather_conditions"
    LOWER_ERROR_BAR: str = "lower_error_bar_weather_conditions"
    LOWER_STANDARD_DEVIATION: str = "lower_standard_deviation_weather_conditions"
    MAXIMUM: str = "maximum_irradiance_weather_conditions"
    MINIMUM: str = "minimum_irradiance_weather_conditions"
    UPPER_ERROR_BAR: str = "upper_error_bar_weather_conditions"
    UPPER_STANDARD_DEVIATION: str = "upper_standard_deviation_weather_conditions"


CLI_TO_PROFILE_TYPE: Dict[str, ProfileType] = {
    "avr": ProfileType.AVERAGE,
    "ler": ProfileType.LOWER_ERROR_BAR,
    "lsd": ProfileType.LOWER_STANDARD_DEVIATION,
    "max": ProfileType.MAXIMUM,
    "min": ProfileType.MINIMUM,
    "uer": ProfileType.UPPER_ERROR_BAR,
    "usd": ProfileType.UPPER_STANDARD_DEVIATION,
}


def read_yaml(
    filepath: str, logger: Logger
) -> Dict[str, bool | float | int | str] | List[Dict[str, bool | float | int | str]]:
    """
    Reads a YAML file and returns the contents.
    """

    # Process the new-location data.
    try:
        with open(filepath, "r", encoding="UTF-8") as filedata:
            file_contents: Dict[str, bool | float | int | str] | List[
                Dict[str, bool | float | int | str]
            ] = yaml.safe_load(filedata)
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
) -> float | None:
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

    # The reduced temperature cannot be computed when there is no solar irradiance
    if solar_irradiance == 0:
        return None

    return (average_temperature - ambient_temperature) / solar_irradiance


class ResourceType(enum.Enum):
    """
    Specifies the type of load being investigated.

    - CLEAN_WATER:
        Represents water which has not been heated or which is contained in a tank for
        which the temperature does not need to be considered.

    - ELECTRICITY:
        Represents an electric load or resource.

    - HOT_WATER:
        Represents water which has been heated.

    """

    CLEAN_WATER = "clean_water"
    ELECTRICITY = "electricity"
    HOT_WATER = "hot_water"


@dataclasses.dataclass
class Scenario:
    """
    Represents the scenario being modelled.

    .. attribute:: battery
        The name of the battery.

    .. attribute:: grid_cost
        The cost of grid electricity (i.e., alternative or unmet electricity) in USD per
        kWh/

    .. attribute:: heat_exchanger_efficiency
        The efficiency of the heat exchanger.

    .. attribute:: heat_pump_efficiency
        The system efficiency of the heat pump installed, expressed as a fraction of the
        Carnot efficiency.

    .. attribute:: hot_water_tank
        The name of the hot-water tank.

    .. attribute:: htf_heat_capacity
        The heat capacity of the HTF.

    .. attribute:: name
        The name of the scenario.

    .. attribute:: plant
        The name of the desalination plant being modelled.

    .. attribute:: pv
        Whether PV panels are being included.

    .. attribute:: pv_degradation_rate
        The annual degradation rate of the PV panels.

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

    .. attribute:: discount_rate
        The discount rate for grid electricity if provided, else 0.

    """

    battery: str
    grid_cost: float
    heat_exchanger_efficiency: float
    heat_pump_efficiency: float
    hot_water_tank: str
    htf_heat_capacity: float
    name: str
    plant: str
    pv_degradation_rate: float
    _pv: bool | str
    _pv_t: bool | str
    _solar_thermal: bool | str
    discount_rate: float = 0

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

        if isinstance(self._solar_thermal, str):
            return self._solar_thermal

        raise Exception(
            "Solar-thermal panel name requested but solar-thermal panels are not "
            "activated in the scenario."
        )


class Solution(NamedTuple):
    """
    Represents a steady-state solution for the system.

    .. attribute:: ambient_temperatures
        The ambient temperature at each time step.

    .. attribute:: auxiliary_heating_demands:
        The auxiliary heating demand in thermal kW needed by the plant in addition to
        any PV-T- or solar-thermal-driven heating.

    .. attribute:: auxiliary_heating_electricity_demands:
        The electricity requirements from the plant due to auxiliary heating.

    .. attribute:: base_electricity_demands:
        The electricity requirements from the plant due to its base electrical load.

    .. attribute:: collector_input_temperatures
        The input temperature to the collector system at each time step.

    .. attribute:: collector_system_output_temperatures
        The output temperature from the solar collectors at each time step.

    .. attribute:: electricity_demands
        The electricity demands placed on the system in kWh at each time step.

    .. attribute:: hot_water_demand_temperature
        The temperature of the hot-water demand at each time step.

    .. attribute:: hot_water_demand_volume
        The volume of the hot-water demand at each time step.

    .. attribute:: pv_electrical_efficiencies
        The electrical efficiencies of the PV collectors at each time step for each year
        of the simulation.

    .. attribute:: pv_electrical_output_power
        The electrcial output power of the PV collectors at each time step for each year
        of the simulation.

    .. attribute:: pv_system_electrical_output_power
        The electrcial output power from all of the installed PV collectors at each time
        step, measured in kWh, for each year of the simulation.

    .. attribute:: pv_t_electrical_efficiencies
        The electrical efficiencies of the PV-T collectors at each time step for each
        year of the simulation.

    .. attribute:: pv_t_electrical_output_power
        The electrical output power of the PV-T collectors at each time step for each
        year of the simulation.

    .. attribute:: pv_t_htf_output_temperatures
        The output temperature from the PV-T collectors at each time step.

    .. attribute:: pv_t_reduced_temperatures
        The reduced temperature of the PV-T collectors at each time step.

    .. attribute:: pv_t_system_electrical_output_power
        The electrcial output power from all of the installed PV-T collectors at each
        time step, measured in kWh, for each year of the simulation.

    .. attribute:: pv_t_thermal_efficiencies
        The thermal efficiency of the PV-T collectors at each time step.

    .. attribute:: solar_thermal_htf_output_temperatures
        The output temperature from the solar-thermal collectors at each time step
        if present.

    .. attribute:: solar_thermal_reduced_temperatures
        The reduced temperature of the solar-thermal collectors at each time step.

    .. attribute:: solar_thermal_thermal_efficiencies
        The thermal efficiency of the solar-thermal collectors at each time step.

    .. attribute:: tank_temperatures
        The temperature of the hot-water tank at each time step.

    .. attribute:: total_collector_electrical_output_power
        The output power of the system at each time step.

    .. attribute:: battery_electricity_suppy_profile
        The amount of energy supplied by the batteries at each hour of the day for each
        year of the simulation.

    .. attribute:: battery_lifetime_degradation
        The total degradation of the batteries over the lifetime of the system.

    .. attribute:: battery_replacements
        The number of battery replacements required over the lifetime of the simulation.

    .. attribute:: battery_storage_profile
        The energy stored in the batteries at each hour of the day for each year of the
        simulation.

    .. attribute:: dumped_solar
        The dumped solar power at each hour of the day.

    .. attribute:: grid_electricity_supply_profile
        The electricity supplied by the grid to the system at each hour of the day for
        each year of the simulation.

    .. attribute:: solar_power_supplied
        The amount of energy supplied by the solar collectors at each hour of the day
        for each year of the simulation.

    """

    ambient_temperatures: Dict[int, float]
    auxiliary_heating_demands: Dict[int, float]
    auxiliary_heating_electricity_demands: Dict[int, float]
    base_electricity_demands: Dict[int, float]
    collector_input_temperatures: Dict[int, float]
    collector_system_output_temperatures: Dict[int, float]
    electricity_demands: Dict[int, float]
    hot_water_demand_temperature: Dict[int, float | None]
    hot_water_demand_volume: Dict[int, float | None]
    pv_electrical_efficiencies: Dict[ProfileDegradationType, Dict[int, float | None]]
    pv_electrical_output_power: Dict[ProfileDegradationType, Dict[int, float | None]]
    pv_system_electrical_output_power: Dict[
        ProfileDegradationType, Dict[int, float | None]
    ]
    pv_t_electrical_efficiencies: Dict[ProfileDegradationType, Dict[int, float | None]]
    pv_t_electrical_output_power: Dict[ProfileDegradationType, Dict[int, float | None]]
    pv_t_htf_output_temperatures: Dict[int, float]
    pv_t_reduced_temperatures: Dict[int, float | None]
    pv_t_system_electrical_output_power: Dict[
        ProfileDegradationType, Dict[int, float | None]
    ]
    pv_t_thermal_efficiencies: Dict[int, float | None]
    solar_thermal_htf_output_temperatures: Dict[int, float]
    solar_thermal_reduced_temperatures: Dict[int, float | None]
    solar_thermal_thermal_efficiencies: Dict[int, float | None]
    tank_temperatures: Dict[int, float]
    battery_electricity_suppy_profile: Dict[int, float | None] | None = None
    battery_lifetime_degradation: int | None = None
    battery_power_input_profile: Dict[int, float] = None
    battery_replacements: int | None = None
    battery_storage_profile: Dict[int, float | None] | None = None
    dumped_solar: Dict[int, float] | None = None
    grid_electricity_supply_profile: Dict[int, float | None] | None = None
    solar_power_supplied: Dict[int, float] | None = None
    output_power_map: Dict[ProfileDegradationType, Dict[int, float]] | None = None

    @property
    def renewable_heating_fraction(self) -> Dict[int, float]:
        """
        Calculate and return the renewable heating fraction.

        `None` is used if the value cannot be defined, e.g., there is no tank demand
        temperature.

        Outputs:
            A mapping containing the renewable heating fraction at each time step.

        """

        return {
            hour: (
                (
                    (self.tank_temperatures[hour] - self.ambient_temperatures[hour])
                    / (
                        self.hot_water_demand_temperature[hour]
                        - self.ambient_temperatures[hour]
                    )
                )
                if demand_temperature is not None
                else None
            )
            for hour, demand_temperature in self.hot_water_demand_temperature.items()
        }

    @property
    def total_collector_electrical_output_power(
        self,
    ) -> Dict[ProfileDegradationType, Dict[int, float]]:
        """
        The total electrical output power at each time step.

        Outputs:
            A mapping containing the degraded and undegraded profiles for total power
            generation.

        """

        if self.output_power_map is not None:
            return self.output_power_map

        output_power_map: DefaultDict[
            ProfileDegradationType, Dict[int, float]
        ] = defaultdict(lambda: defaultdict(float))

        pv = self.pv_system_electrical_output_power is not None
        pvt = self.pv_t_system_electrical_output_power is not None

        # Loop through the profiles.
        for profile in ProfileDegradationType:

            # Add the output for each hour for which it is not None.
            for hour in range(24):
                # Add the PV output power.
                if (
                    pv
                    and (
                        pv_output := self.pv_system_electrical_output_power[
                            profile.value
                        ][hour]
                    )
                    is not None
                ):
                    output_power_map[profile.value][hour] += pv_output

                # Output the PV-T output power.
                if (
                    pvt
                    and (
                        pvt_output := self.pv_t_system_electrical_output_power[
                            profile.value
                        ][hour]
                    )
                    is not None
                ):
                    output_power_map[profile.value][hour] += pvt_output

        self._replace(output_power_map=output_power_map)

        return output_power_map

    @property
    def as_dataframe(self) -> pd.DataFrame:
        """
        Return a :class:`pandas.DataFrame` containing the solution information.

        Outputs:
            A dataframe containing the information associated with the solution.

        """

        # Construct a dictionary based on the available output information.
        output_information_dict = {
            "Ambient temperature / degC": {
                key: value - ZERO_CELCIUS_OFFSET
                for key, value in self.ambient_temperatures.items()
            },
            "Auxiliary heating demand / kWh(th)": self.auxiliary_heating_demands,
            "Battery power inflow profile / kWh": self.battery_power_input_profile,
            "Battery storage profile / kWh": self.battery_storage_profile,
            "Collector system input temperature / degC": {
                key: value - ZERO_CELCIUS_OFFSET
                for key, value in self.collector_input_temperatures.items()
            },
            "Collector system output temperature / degC": {
                key: value - ZERO_CELCIUS_OFFSET
                for key, value in self.collector_system_output_temperatures.items()
            },
            "Dumped electricity / kWh": self.dumped_solar,
            "Electricity demand / kWh": self.electricity_demands,
            "Electrical auxiliary heating demand / kWh(el)": self.auxiliary_heating_electricity_demands,
            "Base electricity dewmand / kWh": self.base_electricity_demands,
            "Electricity demand met through the grid / kWh": self.grid_electricity_supply_profile,
            "Electricity demand met through solar collectors / kWh": self.solar_power_supplied,
            "Electricity demand met through storage / kWh": self.battery_electricity_suppy_profile,
            "Hot-water demand temperature / degC": {
                key: (value - ZERO_CELCIUS_OFFSET) if value is not None else None
                for key, value in self.hot_water_demand_temperature.items()
            },
            "Hot-water demand volume / kg/s": self.hot_water_demand_volume,
            "PV-T collector output temperature / degC": {
                key: (value - ZERO_CELCIUS_OFFSET if value is not None else None)
                for key, value in self.pv_t_htf_output_temperatures.items()
            },
            "Renewable heating fraction": self.renewable_heating_fraction,
            "Solar-thermal collector output temperature / degC": {
                key: (value - ZERO_CELCIUS_OFFSET if value is not None else None)
                for key, value in self.solar_thermal_htf_output_temperatures.items()
            },
            "Tank temperature / degC": {
                key: value - ZERO_CELCIUS_OFFSET
                for key, value in self.tank_temperatures.items()
            },
            "Total degraded collector electrical output power / kW": self.total_collector_electrical_output_power[
                ProfileDegradationType.DEGRADED.value
            ],
            "Total undegraded collector electrical output power / kW": self.total_collector_electrical_output_power[
                ProfileDegradationType.UNDEGRADED.value
            ],
        }

        # Update with PV information if applicable.
        if self.pv_electrical_efficiencies is not None:
            output_information_dict.update(
                {
                    "Undegraded PV electric efficiencies": self.pv_electrical_efficiencies[
                        ProfileDegradationType.UNDEGRADED.value
                    ],
                    "Degraded PV electric efficiencies": self.pv_electrical_efficiencies[
                        ProfileDegradationType.DEGRADED.value
                    ],
                    "Undegraded PV electric output power / kW": self.pv_electrical_output_power[
                        ProfileDegradationType.UNDEGRADED.value
                    ],
                    "Degraded PV electric output power / kW": self.pv_electrical_output_power[
                        ProfileDegradationType.DEGRADED.value
                    ],
                    "Undegraded Total PV electric power produced / kW": self.pv_system_electrical_output_power[
                        ProfileDegradationType.UNDEGRADED.value
                    ],
                    "Degraded Total PV electric power produced / kW": self.pv_system_electrical_output_power[
                        ProfileDegradationType.DEGRADED.value
                    ],
                }
            )

        # Update with PV-T information if applicable.
        if self.pv_t_electrical_efficiencies is not None:
            output_information_dict.update(
                {
                    "Undegraded PV-T electric efficiencies": self.pv_t_electrical_efficiencies[
                        ProfileDegradationType.UNDEGRADED.value
                    ],
                    "Degraded PV-T electric efficiencies": self.pv_t_electrical_efficiencies[
                        ProfileDegradationType.DEGRADED.value
                    ],
                    "Undegraded PV-T electric output power / kW": self.pv_t_electrical_output_power[
                        ProfileDegradationType.UNDEGRADED.value
                    ],
                    "Degraded PV-T electric output power / kW": self.pv_t_electrical_output_power[
                        ProfileDegradationType.DEGRADED.value
                    ],
                    "Undegraded Total PV-T electric power produced / kW": self.pv_t_system_electrical_output_power[
                        ProfileDegradationType.UNDEGRADED.value
                    ],
                    "Degraded Total PV-T electric power produced / kW": self.pv_t_system_electrical_output_power[
                        ProfileDegradationType.DEGRADED.value
                    ],
                    "PV-T output temperature / degC": {
                        key: (
                            value - ZERO_CELCIUS_OFFSET if value is not None else None
                        )
                        for key, value in self.pv_t_htf_output_temperatures.items()
                    },
                    "PV-T reduced temperature / degC/W/m^2": self.pv_t_reduced_temperatures,
                    "PV-T thermal efficiency": self.pv_t_thermal_efficiencies,
                }
            )

        # Update with solar-thermal information if applicable.
        if self.solar_thermal_htf_output_temperatures is not None:
            output_information_dict.update(
                {
                    "Solar-thermal output temperature / degC": {
                        key: (
                            value - ZERO_CELCIUS_OFFSET if value is not None else None
                        )
                        for key, value in self.solar_thermal_htf_output_temperatures.items()
                    },
                    "Solar-thermal reduced temperature / degC/W/m^2": self.solar_thermal_reduced_temperatures,
                    "Solar-thermal thermal efficiency": self.solar_thermal_thermal_efficiencies,
                }
            )

        return pd.DataFrame.from_dict(output_information_dict).sort_index()
