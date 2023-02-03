#!/usr/bin/python3
# type: ignore
########################################################################################
# test_simulator.py - Tests for the simulation module.                                 #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 24/10/2022                                                             #
# License: MIT                                                                         #
########################################################################################
"""
test_simulator.py - Tests for the simulation module.

"""

import unittest

from unittest import mock

from heatdesalination.__utils__ import (
    HEAT_CAPACITY_OF_WATER,
    ZERO_CELCIUS_OFFSET,
    Scenario,
)
from heatdesalination.solar import PVPanel

from ...simulator import (
    _collector_mass_flow_rate,
    _tank_ambient_temperature,
    _tank_replacement_temperature,
    determine_steady_state_simulation,
)


class TestCollectorMassFlowRate(unittest.TestCase):
    """Tests the `_collector_mass_flow_rate` helper function."""

    @unittest.skip
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

    @unittest.skip
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

    @unittest.skip
    def test_mainline(self) -> None:
        """
        Tests the mainline case.

        The tank replacement temperature should be a constant.

        """

        for hour in range(24):
            self.assertEqual(
                _tank_replacement_temperature(hour), ZERO_CELCIUS_OFFSET + 40
            )


class TestRunSimulation(unittest.TestCase):
    """Tests the `run_simulation` function."""

    def setUp(self) -> None:
        """Set up mocks in common across test cases."""

        self.ambient_temperatures: dict[int, float] = {hour: 300 for hour in range(24)}
        self.battery = mock.MagicMock()
        self.battery_capacity = mock.MagicMock()
        self.buffer_tank = mock.MagicMock()
        self.desalination_plant = mock.MagicMock()
        self.htf_mass_flow_rate = mock.Mock()
        self.hybrid_pv_t_panel = mock.MagicMock()
        self.logger = mock.MagicMock()
        self.pv_system_size = mock.MagicMock()
        self.pv_t_system_size = 10
        self.solar_irradiances: dict[int, float] = {hour: 15 for hour in range(24)}
        self.solar_thermal_collector = mock.MagicMock()
        self.solar_thermal_system_size = 10
        self.system_lifetime = 25

        # Setup the PV panel
        pv_input_data = {
            "name": "dusol_ds60275_m",
            "type": "pv",
            "area": 1.638,
            "land_use": 1.638,
            "reference_efficiency": 0.1692,
            "reference_temperature": 25.0,
            "thermal_coefficient": 0.0044,
            "pv_unit": 0.275,
            "cost": 100,
        }

        self.pv_panel: PVPanel = PVPanel.from_dict(self.logger, pv_input_data)
        self.pv_panel.calculate_performance = mock.MagicMock(
            side_effect=[(0, None, None, None)] * 8
            + [(0.3, None, None, None)] * 8
            + [(0, None, None, None)] * 8
        )

        # Setup the scenario
        self.scenario = Scenario(
            (default_name := "default"),
            0,
            0.4,
            0.4,
            default_name,
            default_name,
            HEAT_CAPACITY_OF_WATER,
            "plant",
            0.01,
            default_name,
            default_name,
            default_name,
        )

    @unittest.skip
    def test_mainline(self) -> None:
        """
        Tests the mainline case.

        The run-simulation function primarily acts as a for-loop, and so all that really
        needs to be checked is that the other functions are called as expected.

        """

        # Setup return values and side effects.
        solve_matrix_outputs = (
            [(10, 10, 0, 10, 0, 0, 10, 0, 0, 40)] * 8
            + [(20, 40, 0.125, 30, 0.02, 0.45, 40, 0.01, 0.8, 80)] * 8
            + [(10, 10, 0, 10, 0, 0, 10, 0, 0, 40)] * 8
        )

        # Mock the various functions and methods.
        with mock.patch(
            "heatdesalination.simulator.solve_matrix", side_effect=solve_matrix_outputs
        ) as mock_solve_matrix, mock.patch(
            "heatdesalination.simulator._collector_mass_flow_rate",
            side_effect=[5, 10] * 10,
        ) as mock_collector_mass_flow_rate, mock.patch(
            "heatdesalination.simulator._tank_ambient_temperature", return_value=15
        ) as mock_tank_ambient_temperature, mock.patch(
            "heatdesalination.simulator._tank_replacement_temperature", return_value=10
        ) as mock_tank_replacement_temperature, mock.patch(
            "heatdesalination.simulator.electric_output",
            side_effect=[0.3] * 24 + [0.2] * 24,
        ):
            outputs = determine_steady_state_simulation(
                self.ambient_temperatures,
                self.battery,
                self.battery_capacity,
                self.buffer_tank,
                self.desalination_plant,
                self.htf_mass_flow_rate,
                self.hybrid_pv_t_panel,
                self.logger,
                self.pv_panel,
                self.pv_system_size,
                self.pv_t_system_size,
                self.scenario,
                self.solar_irradiances,
                self.solar_thermal_collector,
                self.solar_thermal_system_size,
                system_lifetime=self.system_lifetime,
                disable_tqdm=True,
            )

        # Check that all mocked functions were called as expected.
        self.assertEqual(len(mock_collector_mass_flow_rate.call_args_list), 2)
        self.assertEqual(len(mock_solve_matrix.call_args_list), 24)
        # The tank ambient temperature is called before iteration.
        self.assertEqual(len(mock_tank_ambient_temperature.call_args_list), 25)
        self.assertEqual(len(mock_tank_replacement_temperature.call_args_list), 24)

        import pdb

        pdb.set_trace()

        # Check that all the outputs are as expected.
        # Collector input temperatures:
        self.assertEqual(
            outputs[0],
            {hour: 10 for hour in range(0, 8)}
            | {hour: 20 for hour in range(8, 16)}
            | {hour: 10 for hour in range(16, 24)},
        )
        # Collector output temperatures:
        self.assertEqual(
            outputs[1],
            {hour: 10 for hour in range(0, 8)}
            | {hour: 40 for hour in range(8, 16)}
            | {hour: 10 for hour in range(16, 24)},
        )
        # PV efficiencies
        self.assertEqual(
            outputs[2],
            {hour: 0 for hour in range(0, 8)}
            | {hour: 0.3 for hour in range(8, 16)}
            | {hour: 0 for hour in range(16, 24)},
        )
        # PV output powers
        self.assertEqual(outputs[3], {hour: 0.2 for hour in range(24)})
        # PV-T electrical efficiencies
        self.assertEqual(
            outputs[4],
            {hour: 0 for hour in range(0, 8)}
            | {hour: 0.125 for hour in range(8, 16)}
            | {hour: 0 for hour in range(16, 24)},
        )
        # PV-T output powers
        self.assertEqual(outputs[5], {hour: 0.3 for hour in range(24)})
        # PV-T HTF output temperatures:
        self.assertEqual(
            outputs[6],
            {hour: 10 for hour in range(0, 8)}
            | {hour: 30 for hour in range(8, 16)}
            | {hour: 10 for hour in range(16, 24)},
        )
        # PV-T reduced temperatures:
        self.assertEqual(
            outputs[7],
            {hour: 0 for hour in range(0, 8)}
            | {hour: 0.02 for hour in range(8, 16)}
            | {hour: 0 for hour in range(16, 24)},
        )
        # PV-T thermal efficiencies:
        self.assertEqual(
            outputs[8],
            {hour: 0 for hour in range(0, 8)}
            | {hour: 0.45 for hour in range(8, 16)}
            | {hour: 0 for hour in range(16, 24)},
        )
        # Solar-thermal output temperatures:
        self.assertEqual(
            outputs[9],
            {hour: 10 for hour in range(0, 8)}
            | {hour: 40 for hour in range(8, 16)}
            | {hour: 10 for hour in range(16, 24)},
        )
        # Solar-thermal reduced temperatures:
        self.assertEqual(
            outputs[10],
            {hour: 0 for hour in range(0, 8)}
            | {hour: 0.01 for hour in range(8, 16)}
            | {hour: 0 for hour in range(16, 24)},
        )
        # Solar-thermal thermal efficiency:
        self.assertEqual(
            outputs[11],
            {hour: 0 for hour in range(0, 8)}
            | {hour: 0.8 for hour in range(8, 16)}
            | {hour: 0 for hour in range(16, 24)},
        )
        # Tank temperatures:
        self.assertEqual(
            outputs[12],
            {hour: 40 for hour in range(0, 8)}
            | {hour: 80 for hour in range(8, 16)}
            | {hour: 40 for hour in range(16, 24)},
        )
