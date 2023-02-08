#!/usr/bin/python3
# type: ignore
########################################################################################
# test_optimiser.py - Tests for the optimisation module.                               #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 24/10/2022                                                             #
# License: MIT                                                                         #
########################################################################################
"""
test_optimiser.py - Tests for the optimisation module.

"""

import dataclasses
import random
import unittest

from unittest import mock

from src.heatdesalination.__utils__ import Scenario
from src.heatdesalination.optimiser import _total_cost


@dataclasses.dataclass
class MockCostableComponent:
    """
    Mocks the costable component with a `cost` attribute.

    .. attribute:: cost
        The cost per component unit.

    """

    cost: float

    def __hash__(self) -> int:
        """Return a uniquely identifiable hash."""

        return hash(random.randint(0, 1000))


@dataclasses.dataclass
class MockSolution:
    """
    Mocks the grid electricity supply profile attribute of the solution.

    .. attribute:: grid_electricity_supply_profile
        The grid electricity supplied per hour.

    """

    grid_electricity_supply_profile: dict[int, float]


class TestCollectorMassFlowRate(unittest.TestCase):
    """Tests the `_collector_mass_flow_rate` helper function."""

    def setUp(self) -> None:
        """Setups mocks in common across test cases."""

        self.no_grid_scenario = Scenario(
            (default_name := "default"),
            0,
            0,
            0,
            default_name,
            0,
            default_name,
            default_name,
            0,
            default_name,
            default_name,
            default_name,
        )

        # Cost per kWh of grid electricity.
        self.grid_cost = 100
        self.grid_scenario = Scenario(
            (default_name := "default"),
            self.grid_cost,
            0,
            0,
            default_name,
            0,
            default_name,
            default_name,
            0,
            default_name,
            default_name,
            default_name,
        )

        # Mock solution to the simulation.
        self.solution = MockSolution({hour: hour**2 for hour in range(24)})
        self.no_grid_solution = MockSolution({hour: 0 for hour in range(24)})

        # Lifetime of the system in years.
        self.system_lifetime: int = 35

    def test_component_costs(self) -> None:
        """
        Tests that the calculation works correctly for various component costs.

        """

        # Test with a pair of components.
        total_cost = _total_cost(
            {MockCostableComponent(100): 100, MockCostableComponent(3): 9},
            self.no_grid_scenario,
            self.solution,
            self.system_lifetime,
        )
        self.assertEqual(total_cost, 100 * 100 + 3 * 9)

        # Test with a single component.
        total_cost = _total_cost(
            {MockCostableComponent(100): 100},
            self.no_grid_scenario,
            self.solution,
            self.system_lifetime,
        )
        self.assertEqual(total_cost, 100 * 100)

        # Test with no components.
        total_cost = _total_cost(
            {},
            self.no_grid_scenario,
            self.solution,
            self.system_lifetime,
        )
        self.assertEqual(total_cost, 0)

    def test_grid_costs(self) -> None:
        """
        Tests that the calculation works correctly for various grid costs.

        """

        # Test with a non-zero grid cost and non-zero grid hours.
        total_cost = _total_cost(
            {},
            self.grid_scenario,
            self.solution,
            self.system_lifetime,
        )
        self.assertEqual(
            total_cost,
            self.grid_scenario.grid_cost
            * sum(self.solution.grid_electricity_supply_profile.values())
            * 365.25
            * self.system_lifetime,
        )

        # Test with zero grid cost and non-zero grid hours.
        total_cost = _total_cost(
            {},
            self.no_grid_scenario,
            self.solution,
            self.system_lifetime,
        )
        self.assertEqual(total_cost, 0)

        # Test with a non-zero grid cost and zero grid hours.

        total_cost = _total_cost(
            {},
            self.grid_scenario,
            self.no_grid_solution,
            self.system_lifetime,
        )
        self.assertEqual(total_cost, 0)

    def test_grid_and_component_cost(self) -> None:
        """Tests when there are grid and component costs."""

        total_cost = _total_cost(
            {MockCostableComponent(100): 100, MockCostableComponent(3): 9},
            self.grid_scenario,
            self.solution,
            self.system_lifetime,
        )
        self.assertEqual(
            total_cost,
            self.grid_scenario.grid_cost
            * sum(self.solution.grid_electricity_supply_profile.values())
            * 365.25
            * self.system_lifetime
            + 100 * 100
            + 3 * 9,
        )


class TestTotalElectricitySuppliedFunction(unittest.TestCase):
    """
    Tests of the helper function for the total cost of the system.

    """


class TestLCUECriterion(unittest.TestCase):
    """
    Tests of the helper function for the total cost of the system.

    """


class TestRenewableElectricityFractionCriterion(unittest.TestCase):
    """
    Tests of the helper function for the total cost of the system.

    """


class TestRenewableHeatingFractionCriterion(unittest.TestCase):
    """
    Tests of the helper function for the total cost of the system.

    """


class TestAuxiliaryHeatingFractionCriterion(unittest.TestCase):
    """
    Tests of the helper function for the total cost of the system.

    """


class TestTotalCostCriterion(unittest.TestCase):
    """
    Tests of the helper function for the total cost of the system.

    """
