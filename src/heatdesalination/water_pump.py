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

from .__utils__ import CostableComponent

__all__ = ("WaterPump",)


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
                The cost of the :class:`HotWaterTank` instance.
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
