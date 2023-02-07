#!/usr/bin/python3.10
########################################################################################
# plant.py - The desalination plant module for the HEATDesalination program.           #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 14/10/2022                                                             #
# License: MIT, Open-source                                                            #
# For more information, contact: benedict.winchester@gmail.com                         #
########################################################################################

"""
plant.py - The desalination-plant module for the HEATDeslination program.

The solar module is responsible for simulations involving the desalination plant(s) in
the system.

"""


import dataclasses

from logging import Logger
from typing import Any

from .__utils__ import NAME, ZERO_CELCIUS_OFFSET

# BRINE:
#   Keyword for the brine outputs from the plant.
BRINE: str = "brine"

# CLEAN_WATER:
#   Keyword for the clean-water outputs from the plant.
CLEAN_WATER: str = "clean_water"

# ELECTRICITY:
#   Keyword for the electricity requirements.
ELECTRICITY: str = "electricity"

# HOT_WATER:
#   Keyword for hot-water requirements.
HOT_WATER: str = "hot_water"

# OPERATING_HOURS:
#   Keyword for the operating hours of the plant.
OPERATING_HOURS: str = "operating_hours"

# OUTPUTS:
#   Keyword for plant outputs.
OUTPUTS: str = "outputs"

# PLANT_DISABLED:
#   Keyword for when the plant is disabled, i.e., not operating.
PLANT_DISABLED: str = "plant_disabled"

# PLANT_OPERATING:
#   Keyword for when the plant is operating.
PLANT_OPERATING: str = "plant_operating"

# REQUIREMENTS:
#   Keyword for plant requirements.
REQUIREMENTS: str = "requirements"

# TEMPERATURE:
#   Keyword for the temperature requirements of the plant.
TEMPERATURE: str = "temperature"

# VOLUME:
#   Keyword for the volume hot-water requirements.
VOLUME: str = "volume"


@dataclasses.dataclass
class PlantOutputs:
    """
    Represents the requirements of the plant.

    .. attribute:: brine
        The brine outputted by the plant in kg/s.

    .. attribute:: clean_water
        The volume of clean water produced in kg/s.

    .. attribute:: hot_water_return_temperature
        The return temperature of the hot water leaving the plant and returning to the
        HTF buffer tank, measured in degrees Celsius.

    """

    brine: float
    clean_water: float
    hot_water_return_temperature: float | None


@dataclasses.dataclass
class PlantRequirements:
    """
    Represents the requirements of the plant.

    .. attribute:: electricity
        The electricity required by the plant, measured in kWh/h.

    .. attribute:: hot_water_temperature
        The temperature of hot water required, measured in degrees Kelvin, or `None` if
        no hot water is required.

    .. attribute:: hot_water_volume
        The volume of hot water required, measured in kilograms per second.

    """

    # Private attributes:
    #   .. attribute:: _hot_water_temperature
    #       The hot-water demand temperature in degrees Celcius.

    electricity: float
    _hot_water_temperature: float | None
    hot_water_volume: float

    @property
    def hot_water_temperature(self) -> float | None:
        """
        The hot-water demand temperature, measured in degrees Kelvin.

        Outputs:
            The temperature in degrees Kelvin.

        """

        if self._hot_water_temperature is None:
            return self._hot_water_temperature

        return self._hot_water_temperature + ZERO_CELCIUS_OFFSET


class DesalinationPlant:
    """
    Represents a desalination plant.

    .. attribute:: end_hour
        The time at which the plant stops operating.

    .. attribute:: name
        The name of the desalination plant.

    .. attribute:: operating
        A map between the time of day and whether the plant is in operation.

    .. attribute:: operating_hours
        The number of hours a day that the plant is operating.

    .. attribute:: plant_outputs
        The outputs of the plant when in operation (True) and not in operation (False).

    .. attribute:: plant_requirements
        The requirements of the plant when in operation (True) and not in operation
        (False).

    .. attribute:: start_hour
        The time at which the plant starts operating.

    """

    def __init__(
        self,
        name: str,
        operating_hours: int,
        plant_outputs: dict[bool, PlantOutputs],
        plant_requirements: dict[bool, PlantRequirements],
        start_hour: int,
    ) -> None:
        """
        Instante the desalination plant instance.

        Inputs:
            - name:
                The name of the desalination plant.
            - operating_hours:
                The number of hours a day that the plant is in operation.
            - plant_outputs:
                The outputs from the desalination plant.
            - plant_reqiurements:
                The requirements for the desalination plant.
            - start_hour:
                The start time for operation of the plant.

        """

        self._end_hour: int | None = (
            (start_hour + operating_hours) % 24 if start_hour is not None else None
        )
        self.name: str = name
        self.operating_hours = operating_hours
        self.plant_outputs: dict[bool, PlantOutputs] = plant_outputs
        self.plant_requirements: dict[bool, PlantRequirements] = plant_requirements
        self.start_hour: int = start_hour
        self._operating: dict[int, bool] | None = None

    def __repr__(self) -> str:
        """
        Return the default representation of the plant.

        Outputs:
            The default representation of the desalination plant.

        """

        return f"DesalinationPlant(name={self.name})"

    @classmethod
    def from_dict(
        cls, input_data: dict[str, Any], logger: Logger, start_hour: int
    ) -> Any:
        """
        Create a :class:`DesalinationPlant` instance based on the inputs.

        Inputs:
            - input_data:
                The input data for the desalination plant.
            - logger:
                The logger being used for the run.
            - start_hour:
                The start time for operation of the plant.

        Outputs:
            The desalination plant based on the input information.

        """

        try:
            plant_outputs = {
                True: PlantOutputs(**input_data[PLANT_OPERATING][OUTPUTS]),
                False: PlantOutputs(**input_data[PLANT_DISABLED][OUTPUTS]),
            }
        except KeyError as exception:
            logger.error(
                "Missing plant-output information in input file: %s", exception
            )
            raise

        try:
            plant_requirements = {
                True: PlantRequirements(
                    input_data[PLANT_OPERATING][REQUIREMENTS][ELECTRICITY],
                    input_data[PLANT_OPERATING][REQUIREMENTS][HOT_WATER][TEMPERATURE],
                    input_data[PLANT_OPERATING][REQUIREMENTS][HOT_WATER][VOLUME],
                ),
                False: PlantRequirements(
                    input_data[PLANT_DISABLED][REQUIREMENTS][ELECTRICITY],
                    input_data[PLANT_DISABLED][REQUIREMENTS][HOT_WATER][TEMPERATURE],
                    input_data[PLANT_DISABLED][REQUIREMENTS][HOT_WATER][VOLUME],
                ),
            }
        except KeyError as exception:
            logger.error(
                "Missing plant-output information in input file: %s", exception
            )
            raise

        return cls(
            input_data[NAME],
            input_data[OPERATING_HOURS],
            plant_outputs,
            plant_requirements,
            start_hour,
        )

    def operating(self, hour: int) -> bool:
        """
        Returns whether the plant is operating at this time of day.

        On the first instance of calling the function, a map between the time of day and
        whether the plant is operating is constructed based on the start and end times
        for the operation of the plant. On subsequent calls, this map is accessed.

        Inputs:
            - hour:
                The hour of the day.

        Outputs:
            Whether the plant is operating (True) or not (False).

        Raises:
            - Exception:
                Raise if the start and end hours are the same.

        """

        if self._operating is not None:
            return self._operating[hour]

        # Save the end-hour information.
        self._end_hour = self.start_hour + self.operating_hours

        # If the map does not exist, compute and save it.
        if self.start_hour < self._end_hour:
            self._operating = {
                hour: self.start_hour <= hour < self._end_hour for hour in range(24)
            }
        elif self.start_hour > self._end_hour:
            self._operating = {
                hour: hour < self._end_hour or hour >= self.start_hour
                for hour in range(24)
            }
        else:
            raise Exception(
                "Error in desalination plant hours: start hour cannot equal end hour."
            )

        return self._operating[hour]

    def outputs(self, hour: int) -> PlantOutputs:
        """
        Returns the outputs from the plant based on the input hour.

        Inputs:
            - hour:
                The hour of the day.

        Outputs:
            The outputs from the plant.

        """

        return self.plant_outputs[self.operating(hour % 24)]

    def requirements(self, hour: int) -> PlantRequirements:
        """
        Returns the requirements for the plant based on the input hour.

        Inputs:
            - hour:
                The hour of the day.

        Outputs:
            The requirements for the plant.

        """

        return self.plant_requirements[self.operating(hour % 24)]

    def reset_operating_hours(self) -> None:
        """
        Resets the internal operating hours variable.

        """

        self._operating = None
