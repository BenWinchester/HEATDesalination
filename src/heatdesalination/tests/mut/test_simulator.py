#!/usr/bin/python3
# type: ignore
########################################################################################
# test_simulator.py - Tests for the simulation module.                                 #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 24/10/2022                                                             #
# License: Open source                                                                 #
########################################################################################
"""
test_simulator.py - Tests for the simulation module.

"""

from typing import Tuple
import unittest

from unittest import mock

from heatdesalination.__utils__ import ZERO_CELCIUS_OFFSET

from ...simulator import (
    _collector_mass_flow_rate,
    _tank_ambient_temperature,
    _tank_replacement_temperature,
    run_simulation,
)


class TestCollectorMassFlowRate(unittest.TestCase):
    """Tests the `_collector_mass_flow_rate` helper function."""

    def test_mainline(self) -> None:
        """
        Tests the mainline case.

        Tests that positive values work as expected: the HTF mass flow rate divided by
        the number of collectors should give the flow rate through each collector.

        """

        # Test positive values work as expected.
        for htf_flow_rate, system_size in zip([100, 40, 30], [10, 33, 15]):
            self.assertEqual(
                _collector_mass_flow_rate(htf_flow_rate, system_size),
                htf_flow_rate / system_size,
            )


class TestTankAmbientTemperature(unittest.TestCase):
    """Tests the `_tank_ambient_temperature` helper function."""

    def test_mainline(self) -> None:
        """
        Tests the mainline case.

        The temperature of the air surrounding the tank should be the same as the
        ambient.

        """

        for ambient_temperature in range(-25, 30, 25):
            self.assertEqual(
                ambient_temperature, _tank_ambient_temperature(ambient_temperature)
            )


class TestTankReplacementTempearture(unittest.TestCase):
    """Tests the `_tank_replacement_temperature` helper function."""

    def test_mainline(self) -> None:
        """
        Tests the mainline case.

        The tank replacement temperature should be a constant.

        """

        for hour in range(24):
            self.assertEqual(
                _tank_replacement_temperature(hour), ZERO_CELCIUS_OFFSET + 20
            )


class TestRunSimulation(unittest.TestCase):
    """Tests the `run_simulation` function."""
