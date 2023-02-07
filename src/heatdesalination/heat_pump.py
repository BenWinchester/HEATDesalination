#!/usr/bin/python3.10
########################################################################################
# heat_pump.py - The heat-pump module for the HEATDesalination program.                #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 03/11/2022                                                             #
# License: MIT, Open-source                                                            #
# For more information, contact: benedict.winchester@gmail.com                         #
########################################################################################

"""
heat_pump.py - The heat-pump module for the HEATDeslination program.

The heat-pump module contains functionality for modelling a heat pump's performance.

"""

import dataclasses

from typing import list, Tuple

from scipy import interpolate

__all__ = ("calculate_heat_pump_electricity_consumption",)


@dataclasses.dataclass
class HeatPump:
    """
    Represents a heat pump.

    .. attribute:: cop_data
        The data points for interpolation for COP.

    .. attribute:: efficiency
        The system efficiency of the heat pump installed, expressed as a fraction of the
        Carnot efficiency.

    .. attribute:: name
        The name of the heat pump.

    .. attribute:: specific_cost_data
        The data points for interpolation for specific cost in USD per kW.

    """

    # Private attributes:
    # .. attribute:: _interpolator
    #   Used for storing the scipy interpolator created.
    #

    cop_data: list[float]
    efficiency: float
    name: str
    specific_costs_data: str
    _interpolator: interpolate.PchipInterpolator | None = None

    def get_cost(self, cop: float, thermal_power: float) -> float:
        """
        Calculate the cost of the heat pump given its COP and thermal power.

        Inputs:
            - cop:
                The COP of the heat pump.
            - thermal_power:
                The thermal power rating of the heat pump, i.e., its maximum thermal
                power output in kWh_th.

        Outputs:
            The cost of the heat pump in USD.

        """

        # Set the interpolator if not already calculated.
        if self._interpolator is None:
            self._interpolator = interpolate.PchipInterpolator(
                self.cop_data, self.specific_costs_data
            )

        # Determine the cost
        return self._interpolator(cop) * thermal_power


def _coefficient_of_performance(
    condensation_temperature: float,
    evaporation_temperature: float,
    system_efficiency: float,
) -> float:
    """
    Calculate the coefficient of performance of the heat pump.

    The coefficient of performance of the heat pump can be calculated based on the
    difference between the temperatures of the condensation and evaporation resevoirs:

        COP = system_efficiency * T_condensation / (T_condensation - T_evaporation)

    in such a way that, as the two temperatures appraoch one another, the COP increases
    as there is less of a temperature range to span. The system efficiency is the
    fraction of this ideal Carnot efficiency which can be achieved.

    Inputs:
        - condensation_temperature:
            The temperature at which condensation within the heat pump occurs. This is
            the temperature at which heat is transferred from the heat pump to the
            surroundings and is hence the desired temperature for the environment,
            measured in degrees Kelvin.
        - evaporation_temperature:
            The temperature at which evaporation within the heat pump occurs. This is
            the temperature at which heat is absorbed from the environment in order to
            evaporate the heat-transfer fluid (refrigerant) within the heat pump,
            measured in degrees Kelvin.
        - system_efficiency:
            The efficiency of the heat pump system given as a fraction of its efficiency
            against the Carnot efficiency.

    Outputs:
        The coefficient of performance of the heat pump.

    """

    return (
        system_efficiency
        * condensation_temperature
        / (condensation_temperature - evaporation_temperature)
    )


def calculate_heat_pump_electricity_consumption_and_cost(
    condensation_temperature: float,
    evaporation_temperature: float,
    heat_demand: float,
    heat_pump: HeatPump,
) -> Tuple[float, float]:
    """
    Calculate the electricity comsumption and cost of the heat pump.

    The coefficient of performance of a heat pump gives the ratio between the heat
    demand which can be achieved and the electricity input which is required to achieve
    it:

        COP = q_th / q_el,

    where q_th and q_el give the heat and electricity powers/energies respectively.
    Hence, this equation can be reversed to give:

        q_el = q_th / COP.

    Inputs:
        - condensation_temperature:
            The temperature at which condensation within the heat pump occurs. This is
            the temperature at which heat is transferred from the heat pump to the
            surroundings and is hence the desired temperature for the environment,
            measured in degrees Kelvin.
        - evaporation_temperature:
            The temperature at which evaporation within the heat pump occurs. This is
            the temperature at which heat is absorbed from the environment in order to
            evaporate the heat-transfer fluid (refrigerant) within the heat pump,
            measured in degrees Kelvin.
        - heat_demand:
            The heat demand flux, measured in kiloWatts.
        - heat_pump:
            The heat pump currently being considered.

    Outputs:
        - The cost of the heat pump in USD,
        - The electricity consumption, measured in kiloWatts.

    """

    power_consumption = heat_demand / (
        cop := _coefficient_of_performance(
            condensation_temperature, evaporation_temperature, heat_pump.efficiency
        )
    )
    cost = heat_pump.get_cost(cop, heat_demand)

    return (cost, power_consumption)
