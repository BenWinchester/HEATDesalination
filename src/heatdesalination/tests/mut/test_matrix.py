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

from ..matrix import (
    _collectors_input_temperature,
    _solar_system_output_temperatures,
    _tank_temperature,
    solve_matrix,
)


class TestCollectorsInputTemperature(unittest.TestCase):
    """Tests the `_collectors_input_temperature` helper function."""


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
