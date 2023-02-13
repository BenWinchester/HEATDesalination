#!/usr/bin/python3.10
########################################################################################
# water_pump.py - The water-pump module for the HEATDesalination program.              #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 01/02/2022                                                             #
# License: MIT, Open-source                                                            #
# For more information, contact: benedict.winchester@gmail.com                         #
########################################################################################

"""
water_pump.py - The water-pump module for the HEATDeslination program.

The water pump(s) are responsible for pushing heat-transfer fluid (HTF) through the PV-T
and solar-thermal collectors. In doing so, they consume some amount of power.

"""

import math

from .__utils__ import CostableComponent

__all__ = (
    "num_water_pumps",
    "WaterPump",
)


class WaterPump(CostableComponent):
    """
    Represents a water pump within the system.

    .. attribute:: efficiency
        The efficiency of the pump (unused attribute).

    .. attribute:: nominal_flow_rate
        The nominal flow rate of the pump in kilograms per second.

    .. attribute:: nominal_power
        The nominal power consumption of the pump, i.e., the power consumed per unit
        pump, measured in kW/pump.

    """

    def __init__(
        self,
        cost: float,
        efficiency: float,
        name: str,
        nominal_flow_rate: float,
        nominal_power: float,
    ) -> None:
        """
        Instantiate a water pump.

        Inputs:
            - cost:
                The cost of the :class:`WaterPump` instance.
            - efficiency:
                The efficiency of the water pump.
            - nominal_flow_rate:
                The nominal flow rate of the pump, measured in kg/s.
            - nominal_power:
                The nominal power consumption of the pump, measured in kW/pump.

        """

        super().__init__(cost, name)

        self.efficiency = efficiency
        self.nominal_flow_rate = nominal_flow_rate
        self.nominal_power = nominal_power

    def __str__(self) -> str:
        """Return a nice-looking `str` representing the water pump."""

        return (
            "WaterPump("
            + f"name={self.name}, "
            + f"cost={self.cost} USD/pump, "
            + f"nominal_flow_rate={self.nominal_flow_rate} kg/s, "
            + f"nominal_power={self.nominal_power} kW"
            ")"
        )

    def __repr__(self) -> str:
        """Return a nice-looking representation of the water pump."""

        return self.__str__()

    def electricity_demand(self, flow_rate: float) -> float:
        """
        Calculate the electricity demand of the pump based on the flow rate.

        The assumption in this calculation is that the power consumption of the pump
        scales linearly with the flow rate of HTF through the pump.

        Inputs:
            - flow_rate:
                The flow rate of HTF through the system, measured in kg/s.

        Outputs:
            The electricity demand of the pump, measured in kW.

        """

        return self.nominal_power * (flow_rate / self.nominal_flow_rate)


def num_water_pumps(htf_mass_flow_rate: float, water_pump: WaterPump) -> int:
    """
    Return the capacity of water pump(s) installed, i.e., the number of installed pumps.

    Inputs:
        - htf_mass_flow_rate:
            The mass flow rate of HTF through the collectors, measured in kg/s.
        - water_pump:
            The water pump being considered.

    """

    return math.ceil(htf_mass_flow_rate / water_pump.nominal_flow_rate)
