#!/usr/bin/python3
########################################################################################
# test_matrix.py - Tests for the matrix module.                                        #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 24/10/2022                                                             #
# License: Open source                                                                 #
########################################################################################
"""
test_matrix.py - Tests for the matrix module.

"""

import unittest

from unittest import mock

from heatdesalination.__utils__ import HEAT_CAPACITY_OF_WATER

from ...matrix import (
    _collectors_input_temperature,
    _solar_system_output_temperatures,
    _tank_temperature,
    solve_matrix,
)


class TestCollectorsInputTemperature(unittest.TestCase):
    """Tests the `_collectors_input_temperature` helper function."""

    def test_mainline(self) -> None:
        """
        Tests the mainline case.

        The continuity equation being tested is the energy exchange through the heat
        exchanger:

        c_tank * eff_exchanger * (T_collector,out - T_tank)
        = c_htf * (T_collector,out - T_collector,in)

        """

        collector_system_input_temperature = _collectors_input_temperature(
            (collector_system_output_temperature := 65),
            (heat_exchanger_efficiency := 0.4),
            (htf_heat_capacity := HEAT_CAPACITY_OF_WATER),
            (tank_temperature := 75),
            (tank_water_heat_capacity := HEAT_CAPACITY_OF_WATER),
        )

        self.assertEqual(
            tank_water_heat_capacity
            * heat_exchanger_efficiency
            * (collector_system_output_temperature - tank_temperature),
            htf_heat_capacity
            * (
                collector_system_output_temperature - collector_system_input_temperature
            ),
        )


class TestSolarSystemOutputTemperatures(unittest.TestCase):
    """Tests the `_solar_system_output_temperatures` helper function."""


class TestTankTemperature(unittest.TestCase):
    """Tests the `_tank_temperature` helper function."""

    def setUp(self) -> None:
        """Sets up functionality in common across test cases."""

        super().setUp()


class TestSolveMatrix(unittest.TestCase):
    """Tests the `solve_matrix` function."""

    def setUp(self) -> None:
        """Sets up functionality in common across test cases."""

        super().setUp()
