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
import enum
import logging
import os

from logging import Logger
from typing import Any, Dict, List, NamedTuple, Tuple

import yaml

import numpy as np
import pandas as pd

__all__ = (
    "AMBIENT_TEMPERATURE",
    "AREA",
    "COST",
    "CostableComponent",
    "FlowRateError",
    "InputFileError",
    "LATITUDE",
    "LOGGER_DIRECTORY",
    "LONGITUDE",
    "NAME",
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
    STORAGE: str = "storage"


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
        Constraints associated with the optimisation, stored as a mappin.

    .. attribute:: maximise
        Whether to maximise the target_criterion (True) or minimise it (False).

    .. attribute:: minimise
        Whether to minimise the target criterion (True) or maximise it (False).

    .. attribute:: target_criterion
        The target criterion.

    """

    bounds: Dict[OptimisableComponent, Dict[str, float | None]]
    constraints: Dict[str, Dict[str, float | None]]
    _criterion: Dict[str, OptimisationMode]

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

    @property
    def fixed_storage_value(self) -> float | None:
        """
        The initial guess for the `storage` if provided, else `None`.

        Outputs:
            - An initial guess for number of batteries installed or `None` if not
              provided.

        """

        return self.bounds[OptimisableComponent.STORAGE].get(FIXED, None)

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
        if self.fixed_storage_value is None:
            initial_guess_vector.append(
                self.bounds[(component := OptimisableComponent.STORAGE)][INITIAL_GUESS]
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

        return cls(bounds, optimisation_inputs[CONSTRAINTS], criterion)


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
    MAXIMUM: str = "maximum_irradiance_weather_conditions"
    MINIMUM: str = "minimum_irradiance_weather_conditions"


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

    battery: str
    heat_exchanger_efficiency: float
    heat_pump_efficiency: float
    hot_water_tank: str
    htf_heat_capacity: float
    name: str
    plant: str
    _pv: bool | str
    _pv_t: bool | str
    _solar_thermal: bool | str

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
        The electrical efficiencies of the PV collectors at each time step.

    .. attribute:: pv_electrical_output_power
        The electrcial output power of the PV collectors at each time step.

    .. attribute:: pv_system_electrical_output_power
        The electrcial output power from all of the installed PV collectors at each time
        step, measured in kWh.

    .. attribute:: pv_t_electrical_efficiencies
        The electrical efficiencies of the PV-T collectors at each time step.

    .. attribute:: pv_t_electrical_output_power
        The electrical output power of the PV-T collectors at each time step.

    .. attribute:: pv_t_htf_output_temperatures
        The output temperature from the PV-T collectors at each time step.

    .. attribute:: pv_t_reduced_temperatures
        The reduced temperature of the PV-T collectors at each time step.

    .. attribute:: pv_t_system_electrical_output_power
        The electrcial output power from all of the installed PV-T collectors at each
        time step, measured in kWh.

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

    """

    ambient_temperatures: Dict[int, float]
    collector_input_temperatures: Dict[int, float]
    collector_system_output_temperatures: Dict[int, float]
    electricity_demands: Dict[int, float]
    hot_water_demand_temperature: Dict[int, float | None]
    hot_water_demand_volume: Dict[int, float | None]
    pv_electrical_efficiencies: Dict[int, float | None]
    pv_electrical_output_power: Dict[int, float | None]
    pv_system_electrical_output_power: Dict[int, float | None]
    pv_t_electrical_efficiencies: Dict[int, float | None]
    pv_t_electrical_output_power: Dict[int, float | None]
    pv_t_htf_output_temperatures: Dict[int, float]
    pv_t_reduced_temperatures: Dict[int, float | None]
    pv_t_system_electrical_output_power: Dict[int, float | None]
    pv_t_thermal_efficiencies: Dict[int, float | None]
    solar_thermal_htf_output_temperatures: Dict[int, float]
    solar_thermal_reduced_temperatures: Dict[int, float | None]
    solar_thermal_thermal_efficiencies: Dict[int, float | None]
    tank_temperatures: Dict[int, float]

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

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a :class:`pandas.DataFrame` containing the solution information.

        Outputs:
            A dataframe containing the information associated with the solution.

        """

        return pd.DataFrame.from_dict(
            {
                "Ambient temperature / degC": {
                    key: value - ZERO_CELCIUS_OFFSET
                    for key, value in self.ambient_temperatures.items()
                },
                "Collector system input temperature / degC": {
                    key: value - ZERO_CELCIUS_OFFSET
                    for key, value in self.collector_input_temperatures.items()
                },
                "Collector system output temperature / degC": {
                    key: value - ZERO_CELCIUS_OFFSET
                    for key, value in self.collector_system_output_temperatures.items()
                },
                "Electricity demand / kWh": self.electricity_demands,
                "Hot-water demand temperature / degC": self.hot_water_demand_temperature,
                "Hot-water demand volume / kg/s": self.hot_water_demand_volume,
                "PV-T collector output temperature / degC": {
                    key: value - ZERO_CELCIUS_OFFSET
                    for key, value in self.pv_t_htf_output_temperatures.items()
                },
                "Renewable heating fraction": self.renewable_heating_fraction,
                "Solar-thermal collector output temperature / degC": {
                    key: value - ZERO_CELCIUS_OFFSET
                    for key, value in self.solar_thermal_htf_output_temperatures.items()
                },
                "Tank temperature / degC": {
                    key: value - ZERO_CELCIUS_OFFSET
                    for key, value in self.tank_temperatures.items()
                },
                "PV electric efficiencies": self.pv_electrical_efficiencies,
                "PV electric output power / kW": self.pv_electrical_output_power,
                "Total PV electric power produced / kW": self.pv_system_electrical_output_power,
                "PV-T electric efficiencies": self.pv_t_electrical_efficiencies,
                "PV-T electric output power / kW": self.pv_t_electrical_output_power,
                "PV-T reduced temperature / degC/W/m^2": self.pv_t_reduced_temperatures,
                "Total PV-T electric power produced / kW": self.pv_t_system_electrical_output_power,
                "PV-T thermal efficiency": self.pv_t_thermal_efficiencies,
                "Solar-thermal reduced temperature / degC/W/m^2": self.solar_thermal_reduced_temperatures,
                "Solar-thermal thermal efficiency": self.solar_thermal_thermal_efficiencies,
            }
        ).sort_index()
