#!/usr/bin/python3
# type: ignore
########################################################################################
# test_solar.py - Tests for the solar module.                                          #
#                                                                                      #
# Author: Ben Winchester, Phil Sandwell                                                #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 21/10/2022                                                             #
# License: MIT                                                                         #
########################################################################################
"""
test_solar.py - Tests for the solar module.

"""

import unittest

from typing import Any
from unittest import mock

from ...__utils__ import (
    HEAT_CAPACITY_OF_WATER,
    ZERO_CELCIUS_OFFSET,
    reduced_temperature,
)

from ...solar import (
    REFERENCE_SOLAR_IRRADIANCE,
    HybridPVTPanel,
    PVPanel,
    PerformanceCurve,
    PVModuleCharacteristics,
    SolarThermalPanel,
    electric_output,
)


class TestPerformanceCurve(unittest.TestCase):
    """
    Tests the :class:`PerformanceCurve` class.

    The :class:`PerformanceCurve` instances expose three property methods which are
    tested here.

    """

    @unittest.skip
    def test_properties(self) -> None:
        """Tests that a :class:`PerformanceCurve` can be instantiated as expected."""

        zeroth: float = 0.0
        first: float = 1.0
        second: float = 2.0

        performance_curve = PerformanceCurve(zeroth, first, second)

        self.assertEqual(zeroth, performance_curve.eta_0)
        self.assertEqual(first, performance_curve.c_1)
        self.assertEqual(second, performance_curve.c_2)


class TestPVPanel(unittest.TestCase):
    """Tests the :class:`PVPanel` instance."""

    def setUp(self) -> None:
        """Sets up functionality in common across test cases."""

        self.input_data = {
            "name": "dusol_ds60275_m",
            "type": "pv",
            "area": 1.638,
            "land_use": 1.638,
            "reference_efficiency": 0.1692,
            "reference_temperature": 25.0,
            "thermal_coefficient": 0.0044,
            "pv_unit": 0.275,
        }

        self.ambient_temperature: float = ZERO_CELCIUS_OFFSET
        self.logger = mock.MagicMock()
        self.solar_irradiance: float = 1000

        super().setUp()

    @unittest.skip
    def test_instantiate(self) -> None:
        """Tests instantiation with the default PV unit."""

        PVPanel.from_dict(mock.MagicMock(), self.input_data)

    @unittest.skip("PV code in progress")
    @unittest.skip
    def test_calculate_performance(self) -> None:
        """Tests the calculate performance method."""

        pv_panel = PVPanel.from_dict(mock.MagicMock(), self.input_data)
        pv_panel.calculate_performance(
            self.ambient_temperature, self.logger, self.solar_irradiance
        )


class TestHybridPVTPanelPerformance(unittest.TestCase):
    """Tests the `calculate_performance` function of the hybrid PV-T panel."""

    def setUp(self) -> None:
        """Sets up functionality in common across test cases."""

        self.linear_input_data = {
            "name": "abora_h72sk",
            "type": "pv_t",
            "area": 1.9602,
            "land_use": 1.9602,
            "max_mass_flow_rate": 1000.0,
            "min_mass_flow_rate": 20.0,
            "nominal_mass_flow_rate": 60.0,
            "pv_module_characteristics": {
                "nominal_power": 0.400,
                "reference_efficiency": 0.178,
                "reference_temperature": 25.0,
                "thermal_coefficient": 0.0036,
            },
            "thermal_performance_curve": {
                "zeroth_order": 0.7,
                "first_order": -5.98,
                "second_order": -0.0,
            },
        }

        self.quadratic_input_data = {
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

        # Set up required mocks for instantiation.
        self.ambient_temperature = 40
        self.input_temperature = 30
        self.mass_flow_rate = 100
        self.logger = mock.Mock()
        self.test_logger = mock.Mock()
        self.solar_irradiance = 950

        super().setUp()

    def _calculate_performance(self, input_data: dict[str, Any]) -> None:
        """Wrapper for the calculate-performance method."""

        pv_t_panel: HybridPVTPanel = HybridPVTPanel.from_dict(self.logger, input_data)
        (
            electrical_efficiency,
            output_temperature,
            reduced_collector_temperature,
            thermal_efficiency,
        ) = pv_t_panel.calculate_performance(
            self.ambient_temperature + ZERO_CELCIUS_OFFSET,
            self.logger,
            self.solar_irradiance,
            HEAT_CAPACITY_OF_WATER,
            self.input_temperature + ZERO_CELCIUS_OFFSET,
            self.mass_flow_rate / 3600,
        )

        # Check the electrical efficiency.
        average_temperature = 0.5 * (self.input_temperature + output_temperature)
        if pv_t_panel.pv_module_characteristics is not None:
            electrical_efficiency_by_equation = (
                pv_t_panel.pv_module_characteristics.reference_efficiency
                * (
                    1
                    - pv_t_panel.pv_module_characteristics.thermal_coefficient
                    * (
                        average_temperature
                        - (
                            pv_t_panel.pv_module_characteristics.reference_temperature
                            - ZERO_CELCIUS_OFFSET
                        )
                    )
                )
            )
            self.assertEqual(electrical_efficiency, electrical_efficiency_by_equation)

        # Check the thermal efficiency.
        efficiency_by_equation = (
            pv_t_panel.thermal_performance_curve.eta_0
            + pv_t_panel.thermal_performance_curve.c_1
            * (average_temperature - self.ambient_temperature)
            / self.solar_irradiance
            + pv_t_panel.thermal_performance_curve.c_2
            * (average_temperature - self.ambient_temperature) ** 2
            / self.solar_irradiance
        )
        self.assertAlmostEqual(efficiency_by_equation, thermal_efficiency)

        reduced_collector_temperature_by_equation = (
            average_temperature - self.ambient_temperature
        ) / self.solar_irradiance
        self.assertAlmostEqual(
            reduced_collector_temperature_by_equation, reduced_collector_temperature
        )

        efficiency_by_output: float = (
            (self.mass_flow_rate / 3600)
            * HEAT_CAPACITY_OF_WATER
            * (output_temperature - self.input_temperature)
        ) / (pv_t_panel.area * self.solar_irradiance)

        self.assertAlmostEqual(
            round(efficiency_by_equation, 8), round(efficiency_by_output, 8)
        )

    @unittest.skip
    def test_calculate_performance_linear_curve(self) -> None:
        """Tests the calculate-performance method with a linear performance curve."""

        self._calculate_performance(self.linear_input_data)

    @unittest.skip
    def test_calculate_performance_quadratic_curve(self) -> None:
        """Tests the calculate-performance method with a quadratic performance curve."""

        self._calculate_performance(self.quadratic_input_data)


class TestSolarThermalPanelPerformance(unittest.TestCase):
    """Tests the `calculate_performance` function of the solar-thermal collector."""

    def setUp(self) -> None:
        """Sets up functionality in common across test cases."""

        self.input_data = {
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

        # Set up required mocks for instantiation.
        self.ambient_temperature = 40
        self.input_temperature = 30
        self.logger = mock.Mock()
        self.test_logger = mock.Mock()
        self.solar_irradiance = 1000

        super().setUp()

    @unittest.skip
    def test_calculate_performance(self) -> None:
        """
        Tests the mainline case.

        The output temperature of the solar-thermal collector is calculated and then
        used to compute the efficiency of the collector two ways:

            eta = eta_0
                + c_1 * (T_c - T_amb) / G
                + c_2 * (T_c - T_amb) ** 2 / G ,                        (1)

            eta = m_htf * c_htf * (T_out - T_in) / (A * G) .            (2)

        """

        solar_thermal_panel: SolarThermalPanel = SolarThermalPanel.from_dict(
            self.logger, self.input_data
        )

        (
            _,
            output_temperature,
            reduced_collector_temperature,
            thermal_efficiency,
        ) = solar_thermal_panel.calculate_performance(
            self.ambient_temperature + ZERO_CELCIUS_OFFSET,
            self.test_logger,
            self.solar_irradiance,
            HEAT_CAPACITY_OF_WATER,
            self.input_temperature + ZERO_CELCIUS_OFFSET,
            solar_thermal_panel.nominal_mass_flow_rate,
        )

        # Type-check the outputs
        self.assertIsInstance(output_temperature, float)
        self.assertIsInstance(reduced_collector_temperature, float)
        self.assertIsInstance(thermal_efficiency, float)

        # Compute the efficiency two ways and check that these are equal.
        collector_temperature = 0.5 * (self.input_temperature + output_temperature)
        efficiency_by_equation = (
            solar_thermal_panel.thermal_performance_curve.eta_0
            + solar_thermal_panel.thermal_performance_curve.c_1
            * (collector_temperature - self.ambient_temperature)
            / self.solar_irradiance
            + solar_thermal_panel.thermal_performance_curve.c_2
            * (collector_temperature - self.ambient_temperature) ** 2
            / self.solar_irradiance
        )
        self.assertEqual(efficiency_by_equation, thermal_efficiency)

        efficiency_by_output: float = (
            (solar_thermal_panel.nominal_mass_flow_rate)
            * HEAT_CAPACITY_OF_WATER
            * (output_temperature - self.input_temperature)
        ) / (solar_thermal_panel.area * self.solar_irradiance)

        self.assertEqual(
            round(efficiency_by_equation, 8), round(efficiency_by_output, 8)
        )


class TestElectricOutput(unittest.TestCase):
    """Tests the electric output function."""

    @unittest.skip
    def test_mainline(self) -> None:
        """
        Tests the mainline case.

        The electric power out is computed based on the fraction of the rated power that
        should be expected:

        power = (eta_el / eta_ref)  # Fraction of electrical efficiency expected
                * (G / G_ref)       # Fraction of irradiance incident
                * P_nominal         # Rated output power

        """

        electric_power_out = electric_output(
            (electrical_efficiency := 0.13),
            (nominal_power := 0.35),
            (reference_efficiency := 0.125),
            (solar_irradiance := 950),
        )

        electric_power_out_by_equation = (
            (electrical_efficiency / reference_efficiency)
            * (solar_irradiance / REFERENCE_SOLAR_IRRADIANCE)
            * nominal_power
        )

        # Check that the calculation was carried out correctly.
        self.assertEqual(electric_power_out, electric_power_out_by_equation)
