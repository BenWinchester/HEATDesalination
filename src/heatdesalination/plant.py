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


@dataclasses.dataclass
class PlantRequirements:
    """
    Represents the requirements of the plant.

    .. attribute:: electricity
        The electricity required by the plant.

    .. attribute:: hot_water_volume
        The volume of hot water required.

    .. attribute:: hot_water_temperature
        The temperature of hot water required.

    """


class DesalinationPlant:
    """
    Represents a desalination plant.

    .. attribute:: end_time
        The time at which the plant stops operating.

    .. attribute:: operating
        A map between the time of day and whether the plant is in operation.

    .. attribute:: plant_requirements
        The requirements of the plant when in operation (True) and not in operation
        (False).

    .. attribute:: start_time
        The time at which the plant starts operating.

    """

    def __init__(self) -> None:
        """
        Instante the desalination plant instance.

        """

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

        """
