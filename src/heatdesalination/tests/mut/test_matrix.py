#!/usr/bin/python3
# type: ignore
########################################################################################
# test_matrix.py - Tests for the matrix module.                                        #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 24/10/2022                                                             #
# License: MIT                                                                         #
########################################################################################
"""
test_matrix.py - Tests for the matrix module.

"""

from typing import Tuple
import unittest

from unittest import mock

from heatdesalination.__utils__ import (
    HEAT_CAPACITY_OF_WATER,
    ZERO_CELCIUS_OFFSET,
    InputFileError,
    Scenario,
)
from heatdesalination.solar import HybridPVTPanel, SolarThermalPanel
from heatdesalination.storage.storage_utils import HotWaterTank

from ...matrix import (
    _collectors_input_temperature,
    _solar_system_output_temperatures,
    _tank_temperature,
    solve_matrix,
)


class TestCollectorsInputTemperature(unittest.TestCase):
    """Tests the `_collectors_input_temperature` helper function."""

    @unittest.skip
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
        pv_t_panel_inputs = {
            "name": "dualsun_spring_uninsulated",
            "type": "pv_t",
            "area": 1.876,
            "land_use": 1.876,
            "max_mass_flow_rate": 1000.0,
            "min_mass_flow_rate": 20.0,
            "nominal_mass_flow_rate": 60.0,
            "pv_module_characteristics": {
                "nominal_power": 0.400,
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
        self.pv_t_panel = HybridPVTPanel.from_dict(self.logger, pv_t_panel_inputs)
        self.pv_t_panel.calculate_performance = mock.MagicMock(
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
        self.pv_t_mass_flow_rate: float = 0.005
        self.solar_irradiance: float = 1000
        self.solar_thermal_mass_flow_rate: float = 0.001

    def _solar_system_output_temperatures_wrapper(
        self, *, pv_t: bool, solar_thermal: bool
    ) -> float:
        """
        Wrapper for the solar-system output-temperatures function.

        """

        scenario = Scenario(
            (default_name := "default"),
            0,
            self.heat_exchanger_efficiency,
            default_name,
            HEAT_CAPACITY_OF_WATER,
            default_name,
            "plant",
            0,
            default_name,
            default_name if pv_t else False,
            default_name if solar_thermal else False,
        )

        return _solar_system_output_temperatures(
            self.ambient_temperature,
            self.collector_system_input_temperature,
            self.pv_t_panel,
            self.logger,
            self.pv_t_mass_flow_rate,
            scenario,
            self.solar_irradiance,
            self.solar_thermal_panel,
            self.solar_thermal_mass_flow_rate,
        )

    @unittest.skip
    def test_no_thermal_collectors(self) -> None:
        """Tests the PV-T only case."""

        with self.assertRaises(InputFileError):
            self._solar_system_output_temperatures_wrapper(
                pv_t=False, solar_thermal=False
            )

    @unittest.skip
    def test_pv_t_only(self) -> None:
        """Tests the PV-T only case."""

        (
            collector_system_output_temperature,
            pv_t_electrical_efficiency,
            pv_t_htf_output_temperature,
            pv_t_reduced_temperature,
            pv_t_thermal_efficiency,
            solar_thermal_htf_output_temperature,
            solar_thermal_reduced_temperature,
            solar_thermal_thermal_efficiency,
        ) = self._solar_system_output_temperatures_wrapper(
            pv_t=True, solar_thermal=False
        )

        # Check the correct numbers were returned.
        self.assertEqual(collector_system_output_temperature, 50)
        self.assertEqual(pv_t_electrical_efficiency, 0.125)
        self.assertEqual(pv_t_htf_output_temperature, 50)
        self.assertEqual(pv_t_reduced_temperature, 0.002)
        self.assertEqual(pv_t_thermal_efficiency, 0.75)
        self.assertEqual(solar_thermal_htf_output_temperature, None)
        self.assertEqual(solar_thermal_reduced_temperature, None)
        self.assertEqual(solar_thermal_thermal_efficiency, None)

        # Check that the correct panel methods were called.
        self.pv_t_panel.calculate_performance.assert_called_once_with(
            self.ambient_temperature,
            self.logger,
            self.solar_irradiance,
            HEAT_CAPACITY_OF_WATER,
            self.collector_system_input_temperature,
            self.pv_t_mass_flow_rate,
        )
        self.solar_thermal_panel.calculate_performance.assert_not_called()

    @unittest.skip
    def test_pv_t_and_solar_thermal(self) -> None:
        """Tests the PV-T only case."""

        (
            collector_system_output_temperature,
            pv_t_electrical_efficiency,
            pv_t_htf_output_temperature,
            pv_t_reduced_temperature,
            pv_t_thermal_efficiency,
            solar_thermal_htf_output_temperature,
            solar_thermal_reduced_temperature,
            solar_thermal_thermal_efficiency,
        ) = self._solar_system_output_temperatures_wrapper(
            pv_t=True, solar_thermal=True
        )

        # Check the correct numbers were returned.
        self.assertEqual(collector_system_output_temperature, 60)
        self.assertEqual(pv_t_electrical_efficiency, 0.125)
        self.assertEqual(pv_t_htf_output_temperature, 50)
        self.assertEqual(pv_t_reduced_temperature, 0.002)
        self.assertEqual(pv_t_thermal_efficiency, 0.75)
        self.assertEqual(solar_thermal_htf_output_temperature, 60)
        self.assertEqual(solar_thermal_reduced_temperature, 0.001)
        self.assertEqual(solar_thermal_thermal_efficiency, 0.85)

        # Check that the correct panel methods were called.
        self.pv_t_panel.calculate_performance.assert_called_once_with(
            self.ambient_temperature,
            self.logger,
            self.solar_irradiance,
            HEAT_CAPACITY_OF_WATER,
            self.collector_system_input_temperature,
            self.pv_t_mass_flow_rate,
        )
        self.solar_thermal_panel.calculate_performance.assert_called_once_with(
            self.ambient_temperature,
            self.logger,
            self.solar_irradiance,
            HEAT_CAPACITY_OF_WATER,
            50,
            self.solar_thermal_mass_flow_rate,
        )

    @unittest.skip
    def test_solar_thermal_only(self) -> None:
        """Tests the PV-T only case."""

        (
            collector_system_output_temperature,
            pv_t_electrical_efficiency,
            pv_t_htf_output_temperature,
            pv_t_reduced_temperature,
            pv_t_thermal_efficiency,
            solar_thermal_htf_output_temperature,
            solar_thermal_reduced_temperature,
            solar_thermal_thermal_efficiency,
        ) = self._solar_system_output_temperatures_wrapper(
            pv_t=False, solar_thermal=True
        )

        # Check the correct numbers were returned.
        self.assertEqual(collector_system_output_temperature, 60)
        self.assertEqual(pv_t_electrical_efficiency, None)
        self.assertEqual(pv_t_htf_output_temperature, None)
        self.assertEqual(pv_t_reduced_temperature, None)
        self.assertEqual(pv_t_thermal_efficiency, None)
        self.assertEqual(solar_thermal_htf_output_temperature, 60)
        self.assertEqual(solar_thermal_reduced_temperature, 0.001)
        self.assertEqual(solar_thermal_thermal_efficiency, 0.85)

        # Check that the correct panel methods were called.
        self.pv_t_panel.calculate_performance.assert_not_called()
        self.solar_thermal_panel.calculate_performance.assert_called_once_with(
            self.ambient_temperature,
            self.logger,
            self.solar_irradiance,
            HEAT_CAPACITY_OF_WATER,
            self.collector_system_input_temperature,
            self.solar_thermal_mass_flow_rate,
        )


class TestTankTemperature(unittest.TestCase):
    """Tests the `_tank_temperature` helper function."""

    @unittest.skip
    def test_mainline(self) -> None:
        """
        Tests the mainline case.

        The equation for the temperature of the tank is

        """

        buffer_tank: HotWaterTank = HotWaterTank.from_dict(
            {
                "name": "hot_water_tank",
                "area": 1900,
                "capacity": 30000,
                "cycle_lifetime": 1500,
                "heat_loss_coefficient": 1.9,
                "leakage": 0.0,
                "maximum_charge": 1.0,
                "minimum_charge": 0.0,
                "resource_type": "hot_water",
            }
        )
        logger = mock.MagicMock()

        tank_temperature = _tank_temperature(
            buffer_tank,
            (collector_system_output_temperature := 85 + ZERO_CELCIUS_OFFSET),
            (heat_exchanger_efficiency := 0.4),
            (htf_heat_capacity := HEAT_CAPACITY_OF_WATER),
            (htf_mass_flow_rate := 4),
            (load_mass_flow_rate := 4),
            logger,
            (previous_tank_temperature := 75 + ZERO_CELCIUS_OFFSET),
            (tank_ambient_temperature := 15 + ZERO_CELCIUS_OFFSET),
            (tank_replacement_water_temperature := 10 + ZERO_CELCIUS_OFFSET),
            (tank_water_heat_capacity := HEAT_CAPACITY_OF_WATER),
            time_interval=(time_interval := 3600),
        )

        # Re-compute the LHS and RHS of the equation before rearranging.
        lhs = (
            buffer_tank.capacity
            * tank_water_heat_capacity
            * (tank_temperature - previous_tank_temperature)
            / time_interval
        )
        rhs = (
            htf_mass_flow_rate
            * htf_heat_capacity
            * heat_exchanger_efficiency
            * (collector_system_output_temperature - tank_temperature)
            + load_mass_flow_rate
            * htf_heat_capacity
            * (tank_replacement_water_temperature - tank_temperature)
            + buffer_tank.heat_transfer_coefficient
            * (tank_ambient_temperature - tank_temperature)
        )

        self.assertAlmostEqual(lhs, rhs)


class TestSolveMatrix(unittest.TestCase):
    """Tests the `solve_matrix` function."""

    def setUp(self) -> None:
        """Sets up functionality in common across test cases."""

        self.ambient_temperature: float = 15 + ZERO_CELCIUS_OFFSET
        self.buffer_tank = mock.MagicMock()
        self.htf_mass_flow_rate: float = 4 + ZERO_CELCIUS_OFFSET
        self.hybrid_pv_t_panel = mock.MagicMock()
        self.load_mass_flow_rate: float = 4 + ZERO_CELCIUS_OFFSET
        self.logger = mock.MagicMock()
        self.previous_tank_temperature: float = 75 + ZERO_CELCIUS_OFFSET
        self.pv_t_mass_flow_rate: float = 4 + ZERO_CELCIUS_OFFSET
        self.scenario = Scenario(
            (default := "default"),
            0,
            0,
            default,
            0,
            0,
            default,
            default,
            default,
            default,
        )
        self.solar_irradiance = mock.MagicMock()
        self.solar_thermal_collector = mock.MagicMock()
        self.solar_thermal_mass_flow_rate: float = 4 + ZERO_CELCIUS_OFFSET
        self.tank_ambient_temperature: float = 15 + ZERO_CELCIUS_OFFSET
        self.tank_replacement_water_temperature: float = 10 + ZERO_CELCIUS_OFFSET

        super().setUp()

    def _solve_matrix(
        self, *, no_pvt: bool = False, no_solar_thermal: bool = False
    ) -> Tuple[
        float,
        float,
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
        float,
    ]:
        """Wrapper for the solve matrix function."""

        return solve_matrix(
            self.ambient_temperature,
            self.buffer_tank,
            self.htf_mass_flow_rate,
            None if no_pvt else self.hybrid_pv_t_panel,
            self.load_mass_flow_rate,
            self.logger,
            self.previous_tank_temperature,
            self.pv_t_mass_flow_rate,
            self.scenario,
            self.solar_irradiance,
            None if no_solar_thermal else self.solar_thermal_collector,
            self.solar_thermal_mass_flow_rate,
            self.tank_ambient_temperature,
            self.tank_replacement_water_temperature,
        )

    @unittest.skip
    def test_missing_collectors(self) -> None:
        """Tests the case where there are missing collectors."""

        with self.assertRaises(InputFileError):
            self._solve_matrix(no_pvt=True)
        with self.assertRaises(InputFileError):
            self._solve_matrix(no_solar_thermal=True)

    @unittest.skip
    def test_mainline(self) -> None:
        """Tests the mainline case."""

        with mock.patch(
            "heatdesalination.matrix._solar_system_output_temperatures"
        ) as mock_solar_system_output_temperatures, mock.patch(
            "heatdesalination.matrix._tank_temperature",
            side_effect=[entry + ZERO_CELCIUS_OFFSET for entry in [80, 70, 66]],
        ) as mock_tank_temperature, mock.patch(
            "heatdesalination.matrix._collectors_input_temperature",
            side_effect=[entry + ZERO_CELCIUS_OFFSET for entry in [40, 30, 34]],
        ) as mock_collectors_input_temperature, mock.patch(
            "heatdesalination.matrix.TEMPERATURE_PRECISION", 5
        ):
            # Setup the return values for the mock solar-system output
            mock_solar_system_output_temperatures.side_effect = [
                (entry,) + (None,) * 7
                for entry in [entry + ZERO_CELCIUS_OFFSET for entry in [80, 75, 70]]
            ]

            _ = self._solve_matrix()

        # Check that the mocked functions were called as expected.
        for best_guess_input in [
            self.ambient_temperature,
            40 + ZERO_CELCIUS_OFFSET,
            30 + ZERO_CELCIUS_OFFSET,
        ]:
            self.assertEqual(
                best_guess_input,
                mock_solar_system_output_temperatures.call_args_list.pop(0)[0][1],
            )

        for colector_ouptut_temp in [
            entry + ZERO_CELCIUS_OFFSET for entry in [80, 75, 70]
        ]:
            self.assertEqual(
                colector_ouptut_temp, mock_tank_temperature.call_args_list.pop(0)[0][1]
            )

        for collector_input_temp, tank_temp in zip(
            [entry + ZERO_CELCIUS_OFFSET for entry in [80, 75, 70]],
            [entry + ZERO_CELCIUS_OFFSET for entry in [80, 70, 66]],
        ):
            call_args = mock_collectors_input_temperature.call_args_list.pop(0)[0]
            self.assertEqual(collector_input_temp, call_args[0])
            self.assertEqual(tank_temp, call_args[3])
