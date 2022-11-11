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

__all__ = ("calculate_heat_pump_electricity_consumption",)


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


def calculate_heat_pump_electricity_consumption(
    condensation_temperature: float,
    evaporation_temperature: float,
    heat_demand: float,
    system_efficiency: float,
) -> float:
    """
    Calculate the electricity comsumption of the heat pump.

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
        - system_efficiency:
            The efficiency of the heat pump system given as a fraction of its efficiency
            against the Carnot efficiency.

    Outputs:
        The electricity consumption, measured in kiloWatts.

    """

    return heat_demand / _coefficient_of_performance(
        condensation_temperature, evaporation_temperature, system_efficiency
    )
