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

from heatdesalination.__utils__ import HEAT_CAPACITY_OF_WATER, InputFileError, Scenario
from heatdesalination.solar import HybridPVTPanel, SolarThermalPanel

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

    def setUp(self) -> None:
        """Set up mocks in common across all test cases."""

        self.logger = mock.MagicMock()

        # Setup the PV-T panel mock.
        pvt_panel_inputs = {
            "name": "dualsun_spring_uninsulated",
            "type": "pv_t",
            "area": 1.876,
            "land_use": 1.876,
            "max_mass_flow_rate": 1000.0,
            "min_mass_flow_rate": 20.0,
            "nominal_mass_flow_rate": 60.0,
            "pv_module_characteristics": {
                "reference_efficiency": 0.213,
                "reference_temperature": 25.0,
                "thermal_coefficient": 0.0034,
            },
            "thermal_performance_curve": {
                "zeroth_order": 0.633,
                "first_order": -11.5,
                "second_order": -0.0,
            },
        }
        self.pvt_panel = HybridPVTPanel.from_dict(self.logger, pvt_panel_inputs)
        self.pvt_panel.calculate_performance = mock.MagicMock(
            return_value=(0.125, 50, 0.002, 0.75)
        )

        # Setup the solar-thermal panel mock.
        solar_thermal_inputs = {
            "area": 2.169965517,
            "land_use": 2.375310345,
            "max_mass_flow_rate": 332.7733333,
            "min_mass_flow_rate": 56.10666667,
            "name": "mean_fpc",
            "nominal_mass_flow_rate": 97.85714286,
            "thermal_performance_curve": {
                "first_order": -3.619626844,
                "second_order": -0.009328034801,
                "zeroth_order": 0.7334194095,
            },
            "type": "solar_thermal",
        }
        self.solar_thermal_panel = SolarThermalPanel.from_dict(
            self.logger, solar_thermal_inputs
        )
        self.solar_thermal_panel.calculate_performance = mock.MagicMock(
            return_value=(None, 60, 0.001, 0.85)
        )

        # Misc. input parameters.
        self.ambient_temperature: float = 40
        self.collector_system_input_temperature: float = 30
        self.heat_exchanger_efficiency: float = 0.4
        self.pvt_mass_flow_rate: float = 0.005
        self.solar_irradiance: float = 1000
        self.solar_thermal_mass_flow_rate: float = 0.001

    def _solar_system_output_temperatures_wrapper(
        self, *, pv_t: bool, solar_thermal: bool
    ) -> None:
        """
        Wrapper for the solar-system output-temperatures function.

        """

        scenario = Scenario(
            self.heat_exchanger_efficiency,
            HEAT_CAPACITY_OF_WATER,
            (default_name := "default"),
            "plant",
            default_name,
            default_name if pv_t else False,
            default_name if solar_thermal else False,
        )

        return _solar_system_output_temperatures(
            self.ambient_temperature,
            self.collector_system_input_temperature,
            self.pvt_panel,
            self.logger,
            self.pvt_mass_flow_rate,
            scenario,
            self.solar_irradiance,
            self.solar_thermal_panel,
            self.solar_thermal_mass_flow_rate,
        )

    def test_no_thermal_collectors(self) -> None:
        """Tests the PV-T only case."""

        with self.assertRaises(InputFileError):
            self._solar_system_output_temperatures_wrapper(
                pv_t=False, solar_thermal=False
            )

    def test_pvt_only(self) -> None:
        """Tests the PV-T only case."""

        (
            collector_system_output_temperature,
            pvt_electrical_efficiency,
            pvt_htf_output_temperature,
            pvt_reduced_temperature,
            pvt_thermal_efficiency,
            solar_thermal_htf_output_temperature,
            solar_thermal_reduced_temperature,
            solar_thermal_thermal_efficiency,
        ) = self._solar_system_output_temperatures_wrapper(
            pv_t=True, solar_thermal=False
        )

        # Check the correct numbers were returned.
        self.assertEqual(collector_system_output_temperature, 50)
        self.assertEqual(pvt_electrical_efficiency, 0.125)
        self.assertEqual(pvt_htf_output_temperature, 50)
        self.assertEqual(pvt_reduced_temperature, 0.002)
        self.assertEqual(pvt_thermal_efficiency, 0.75)
        self.assertEqual(solar_thermal_htf_output_temperature, None)
        self.assertEqual(solar_thermal_reduced_temperature, None)
        self.assertEqual(solar_thermal_thermal_efficiency, None)

        # Check that the correct panel methods were called.
        self.pvt_panel.calculate_performance.assert_called_once_with(
            self.ambient_temperature,
            HEAT_CAPACITY_OF_WATER,
            self.collector_system_input_temperature,
            self.logger,
            self.pvt_mass_flow_rate,
            self.solar_irradiance,
        )
        self.solar_thermal_panel.calculate_performance.assert_not_called()

    def test_pvt_and_solar_thermal(self) -> None:
        """Tests the PV-T only case."""

        (
            collector_system_output_temperature,
            pvt_electrical_efficiency,
            pvt_htf_output_temperature,
            pvt_reduced_temperature,
            pvt_thermal_efficiency,
            solar_thermal_htf_output_temperature,
            solar_thermal_reduced_temperature,
            solar_thermal_thermal_efficiency,
        ) = self._solar_system_output_temperatures_wrapper(
            pv_t=True, solar_thermal=True
        )

        # Check the correct numbers were returned.
        self.assertEqual(collector_system_output_temperature, 60)
        self.assertEqual(pvt_electrical_efficiency, 0.125)
        self.assertEqual(pvt_htf_output_temperature, 50)
        self.assertEqual(pvt_reduced_temperature, 0.002)
        self.assertEqual(pvt_thermal_efficiency, 0.75)
        self.assertEqual(solar_thermal_htf_output_temperature, 60)
        self.assertEqual(solar_thermal_reduced_temperature, 0.001)
        self.assertEqual(solar_thermal_thermal_efficiency, 0.85)

        # Check that the correct panel methods were called.
        self.pvt_panel.calculate_performance.assert_called_once_with(
            self.ambient_temperature,
            HEAT_CAPACITY_OF_WATER,
            self.collector_system_input_temperature,
            self.logger,
            self.pvt_mass_flow_rate,
            self.solar_irradiance,
        )
        self.solar_thermal_panel.calculate_performance.assert_called_once_with(
            self.ambient_temperature,
            HEAT_CAPACITY_OF_WATER,
            50,
            self.logger,
            self.solar_thermal_mass_flow_rate,
            self.solar_irradiance,
        )

    def test_solar_thermal_only(self) -> None:
        """Tests the PV-T only case."""

        (
            collector_system_output_temperature,
            pvt_electrical_efficiency,
            pvt_htf_output_temperature,
            pvt_reduced_temperature,
            pvt_thermal_efficiency,
            solar_thermal_htf_output_temperature,
            solar_thermal_reduced_temperature,
            solar_thermal_thermal_efficiency,
        ) = self._solar_system_output_temperatures_wrapper(
            pv_t=False, solar_thermal=True
        )

        # Check the correct numbers were returned.
        self.assertEqual(collector_system_output_temperature, 60)
        self.assertEqual(pvt_electrical_efficiency, None)
        self.assertEqual(pvt_htf_output_temperature, None)
        self.assertEqual(pvt_reduced_temperature, None)
        self.assertEqual(pvt_thermal_efficiency, None)
        self.assertEqual(solar_thermal_htf_output_temperature, 60)
        self.assertEqual(solar_thermal_reduced_temperature, 0.001)
        self.assertEqual(solar_thermal_thermal_efficiency, 0.85)

        # Check that the correct panel methods were called.
        self.pvt_panel.calculate_performance.assert_not_called()
        self.solar_thermal_panel.calculate_performance.assert_called_once_with(
            self.ambient_temperature,
            HEAT_CAPACITY_OF_WATER,
            self.collector_system_input_temperature,
            self.logger,
            self.solar_thermal_mass_flow_rate,
            self.solar_irradiance,
        )


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
