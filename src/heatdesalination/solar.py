#!/usr/bin/python3.10
########################################################################################
# solar.py - The solar module for the HEATDesalination program.                        #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 14/10/2022                                                             #
# License: MIT, Open-source                                                            #
# For more information, contact: benedict.winchester@gmail.com                         #
########################################################################################

"""
solar.py - The solar module for the HEATDeslination program.

The solar module is responsible for simulating the performance of the various solar
collectors, whether they be solar-thermal, PV-T or straight PV collectors, depending on
various environmanl conditions.

"""

import abc
import dataclasses
import enum
import math

from logging import Logger
from typing import Any, Tuple

from pvlib import temperature

from .__utils__ import (
    AREA,
    COST,
    CostableComponent,
    FlowRateError,
    InputFileError,
    NAME,
    reduced_temperature,
    TEMPERATURE_PRECISION,
    ZERO_CELCIUS_OFFSET,
)


__all__ = (
    "HybridPVTPanel",
    "PerformanceCurve",
    "PVPanel",
    "SolarThermalPanel",
)

# ELECTRIC_PERFORMANCE_CURVE:
#   Keyword for the electric performance curve.
ELECTRIC_PERFORMANCE_CURVE: str = "electric_performance_curve"

# FIRST_ORDER:
#   Keyword for parsing first-order coefficient.
FIRST_ORDER: str = "first_order"

# LAND_USE:
#   Keyword for the land use of each panel.
LAND_USE: str = "land_use"

# MAX_MASS_FLOW_RATE:
#   Keyword for the maximum mass flow rate of HTF through the panel.
MAX_MASS_FLOW_RATE: str = "max_mass_flow_rate"

# MIN_MASS_FLOW_RATE:
#   Keyword for the minimum mass flow rate of HTF through the panel.
MIN_MASS_FLOW_RATE: str = "min_mass_flow_rate"

# NOMINAL_MASS_FLOW_RATE:
#   Keyword for the nominal mass flow rate of HTF through the panel.
NOMINAL_MASS_FLOW_RATE: str = "nominal_mass_flow_rate"

# NOMINAL_POWER:
#   Keyword for the nominal power of the panel.
NOMINAL_POWER: str = "nominal_power"

# PV_MODULE_CHARACTERISTICS:
#   Keyword for module characteristics.
PV_MODULE_CHARACTERISTICS: str = "pv_module_characteristics"

# PV_UNIT:
#   Keyword for the unit of power outputted per PV panel.
PV_UNIT: str = "pv_unit"

# REFERENCE_EFFICIENCY:
#   Keyword for the reference efficiency of a PV panel.
REFERENCE_EFFICIENCY: str = "reference_efficiency"

# REFERENCE_SOLAR_IRRADIANCE:
#   The reference solar irradiance at which collectors are rated.
REFERENCE_SOLAR_IRRADIANCE: float = 1000

# REFERENCE_TEMPERATURE:
#   Keyword for the reference temperature of a PV panel.
REFERENCE_TEMPERATURE: str = "reference_temperature"

# SECOND_ORDER:
#   Keyword for parsing second-order coefficient.
SECOND_ORDER: str = "second_order"

# STAGNATION_TEMPERATURE:
#   Keyword for the stagnation temperature of the collector.
STAGNATION_TEMPERATURE: str = "stagnation_temperature"

# STEFAN_BOLTZMAN_CONSTANT:
#   The Stefan-Boltzman constant in SI units.
STEFAN_BOLTZMAN_CONSTANT: float = 5.670374419 * (10**-8)

# THERMAL_COEFFICIENT:
#   Keyword for the temperature coefficient for the performance of a PV panel.
THERMAL_COEFFICIENT: str = "thermal_coefficient"

# THERMALPERFORMANCE_CURVE:
#   Keyword for the thermal performance curve.
THERMAL_PERFORMANCE_CURVE: str = "thermal_performance_curve"

# ZEROTH_ORDER:
#   Keyword for parsing zeroth-order coefficient.
ZEROTH_ORDER: str = "zeroth_order"


class SolarPanelType(enum.Enum):
    """
    Denotes the type of solar panel being modelled.

    - PV:
        Denotes a PV panel.

    - PV_T:
        Denotes a PV-T panel.

    - SOLAR_THERMAL
        Denotes a solar-thermal panel.

    """

    PV: str = "pv"
    PV_T: str = "pv_t"
    SOLAR_THERMAL: str = "solar_thermal"


def _conductive_air_heat_transfer_coefficient(wind_speed: float) -> float:
    """
    Calculates the conductive heat-transfer coefficient between PV panels and the air.

    The heat transfer coefficient is given by:
        h_air = 3.97(13) sWm^3K v_wind
              + 6.90(5) W/m^2K

    """

    return float(3.97 * wind_speed + 6.90)


def _sky_temperature(ambient_temperature: float) -> float:
    """
    Determines the radiative temperature of the sky.

    The "sky," as a black body, has a radiative temperature different to that of the
    surrounding air, or the ambient temperature. This function converts between them and
    outputs the sky's radiative temperature.

    Outputs:
        The radiative temperature of the "sky" in Kelvin.

    """

    return float(0.0552 * (ambient_temperature**1.5))


def _radiation_to_sky_coefficient(
    collector_emissivity: float, collector_temperature: float, sky_temperature: float
) -> float:
    """
    Returns the heat-transfer coefficient between a collector and the sky in W/m^2.

    Inputs:
        - collector_emissivity:
            The emissivity of the collector.
        - collector_temperature:
            The temperature of the collector, measured in Kelvin.
        - sky_temperature:
            The sky temperature, measured in Kelvin.

    Outputs:
        The heat-transfer coefficient between the collector and the sky, measured in
        Watts per meter squared.

    """
    return (
        STEFAN_BOLTZMAN_CONSTANT  # [W/m^2*K^4]
        * collector_emissivity
        * (collector_temperature**2 + sky_temperature**2)  # [K^2]
        * (collector_temperature + sky_temperature)  # [K]
    )


@dataclasses.dataclass
class PerformanceCurve:
    """
    Represents a performance curve for a solar-thermal collector.

    Solar-thermal collectors can be characterised by a performance curve,

        eta = eta_0 + c_1 * (T_c - T_a) / G + c_2 * (T_c - T_a)^2 / G,

    where `eta_0`, `c_1` and `c_2` give the zeroth-, first- and second-order
    coefficients which characterise the performance of the collector, `T_c` is the
    average temperature of the collector and `T_a` the ambient temperature, both
    measured in either degrees Kelvin or Celsius, but the same unit for each, and `G` is
    the solar irradiance, measured in Watts per meter squared.

    The attributes, `eta_0`, `c_1` and `c_2` are inherent properties of the collector
    and are contained within this class.

    .. attribute:: zeroth_order_cefficient
        The zeroth-order term for the performance curve.

    .. attribute:: first_order_cefficient
        The zeroth-order term for the performance curve.

    .. attribute:: second_order_cefficient
        The zeroth-order term for the performance curve.

    """

    zeroth_order_coefficient: float
    first_order_coefficient: float
    second_order_coefficient: float

    @property
    def eta_0(self) -> float:
        """
        Wrapper around the zeroth-order coefficient.

        Outputs:
            - The zeroth-order coefficient.

        """

        return self.zeroth_order_coefficient

    @property
    def c_1(self) -> float:
        """
        Wrapper around the first-order coefficient.

        Outputs:
            - The first-order coefficient.

        """

        return self.first_order_coefficient

    @property
    def c_2(self) -> float:
        """
        Wrapper around the second-order coefficient.

        Outputs:
            - The second-order coefficient.

        """

        return self.second_order_coefficient


@dataclasses.dataclass
class PVModuleCharacteristics:
    """
    Represents characteristcs of the PV module.

    .. attribute:: nominal_power
        Denoste the nominal power of the PV module in kWp.

    .. attribute:: reference_efficiency
        Denotes the reference efficiency of the PV module.

    .. attribute:: reference_temperature
        Denotes the reference temperature of the PV module, measured in degrees Kelvin.

    .. attribute:: thermal_coefficient
        Denotes the thermal coefficienct of the PV module.

    """

    nominal_power: float
    reference_efficiency: float
    _reference_temperature: float
    thermal_coefficient: float

    @property
    def reference_temperature(self) -> float:
        """
        Return the reference temperature in degrees Kelvin.

        Outputs:
            The reference temperature in degrees Kelvin.

        """

        return self._reference_temperature + ZERO_CELCIUS_OFFSET


def _thermal_performance(
    ambient_temperature: float,
    area: float,
    htf_heat_capacity: float,
    input_temperature: float,
    mass_flow_rate: float,
    performance_curve: PerformanceCurve,
    solar_irradiance: float,
) -> Tuple[float | None, float]:
    """
    Calculates the roots for the thermal performance of the collectors.

    Each collector has a characteristic performance curve, which is related to the
    efficiency of the collector by a simple equation:

        eta = eta_0
            + c_1 * (T_c - T_amb) / G
            + c_2 * (T_c - T_amb) ** 2 / G

    where `eta_0`, `c_1` and `c_2` give the zeroth-, first- and second-order
    coefficients which characterise the performance of the collector, `T_c` is the
    average temperature of the collector and `T_a` the ambient temperature, both
    measured in either degrees Kelvin or Celsius, but the same unit for each, and
    `G` is the solar irradiance, measured in Watts per meter squared. The attributes
    `eta_0`, `c_1` and `c_2` are inherent properties of the collector and are
    contained within the `performance_curve` attribute.

    This equation can be rearranged by expressing the efficiency as the energy
    gained by the heat-transfer fluid within the collector as a fraction of the
    total energy incident on the collector:

        eta = m_htf * c_htf * (T_out - T_in) / (A * G)

    where `T_out` and `T_in` give the output and input HTF temperatures
    respectively, and `m_htf` and `c_htf` give the mass-flow rate and specific heat
    capacityof the HTF through the collector. Combining these two yields

        0 = 4 * eta_0 * A * G                   \\ = c = zeroth_order_coefficient
            + 4 * m_htf * c_htf * T_in            |
            + 2 * c_1 * A * (T_in - T_amb)        |
            + c_2 * A * (T_in - T_amb) ** 2       /
            + (                                   \\ = b = first_order_coefficient
            - 4 * m_htf * c_htf                 |
            + 2 * c_1 * A                       |
            + 2 * c_2 * A * (T_in - T_amb)      /
            ) * T_out
            + (                                   \\ = a = second_order_coefficient
            4 * eta_0 * A * G                   |
            + 4 * m_htf * c_htf * T_in          |
            + 2 * c_1 * A * (T_in - T_amb)      |
            + c_2 * A * (T_in - T_amb) ** 2     /
            ) * T_out ** 2

    which can then be solved quadratically to determine the output temperature of
    HTF leaving the collector.

    Inputs:
        - ambient_temperature:
            The ambient temperature, measured in degrees Kelvin.
        - area:
            The area of the collector, in meters squared.
        - htf_heat_capacity:
            The heat capacity of the HTF entering the collector, measured in Joules
            per kilogram Kelvin (J/kgK).
        - input_temperature:
            The input temperature of the HTF entering the collector, measured in
            in degrees Kelvin.
        - mass_flow_rate:
            The mass-flow rate of HTF passing through the collector, measured in
            kilograms per second.
        - performance_curve:
            The performance curve for the collector.
        - solar_irradiance:
            The solar irradiance incident on the surface of the collector, measured
            in Watts per meter squared.

    Outputs:
        Both roots from the equation:
        - positive_root:
            The positive root taken from solving the quadratic equation, measured in
            Kelvin.
        - negative_root:
            The negative root taken from solving the quadratic equation, measured in
            Kelvin.

    """

    # If noly a linear calculation is required, solve linearly.
    if performance_curve.c_2 == 0:
        return (
            None,
            (
                2 * performance_curve.eta_0 * area * solar_irradiance
                + 2 * mass_flow_rate * htf_heat_capacity * input_temperature
                + performance_curve.c_1
                * area
                * (input_temperature - 2 * ambient_temperature)
            )
            / (2 * mass_flow_rate * htf_heat_capacity - performance_curve.c_1 * area),
        )

    # Compute the various terms of the equation
    a: float = performance_curve.c_2 * area  # pylint: disable=invalid-name

    b: float = (  # pylint: disable=invalid-name
        +2 * performance_curve.c_1 * area
        + 2
        * performance_curve.c_2
        * area
        * (input_temperature - 2 * ambient_temperature)
        - 4 * mass_flow_rate * htf_heat_capacity
    )

    c: float = (  # pylint: disable=invalid-name
        4 * performance_curve.eta_0 * area * solar_irradiance
        + 4 * mass_flow_rate * htf_heat_capacity * input_temperature
        + 2
        * performance_curve.c_1
        * area
        * (input_temperature - 2 * ambient_temperature)
        + performance_curve.c_2
        * area
        * (input_temperature - 2 * ambient_temperature) ** 2
    )

    # Use numpy or Pandas to solve the quadratic to determine the performance of
    # the collector
    positive_root: float = (  # pylint: disable=unused-variable
        -b + math.sqrt(b**2 - 4 * a * c)
    ) / (2 * a)
    negative_root: float = (-b - math.sqrt(b**2 - 4 * a * c)) / (2 * a)

    return positive_root, negative_root


class SolarPanel(abc.ABC, CostableComponent):  # pylint: disable=too-few-public-methods
    """
    Represents a solar panel being considered.

    .. attribute:: area
        The area of the collector in meters squared, used to calculate the input power
        to the collector.

    .. attribute:: name
        The name of the panel being considered.

    .. attribite:: panel_type
        The type of panel being considered.

    """

    panel_type: SolarPanelType

    def __init__(
        self,
        area: float,
        cost: float,
        land_use: float,
        name: str,
    ) -> None:
        """
        Instantiate a :class:`SolarPanel` instance.

        Inputs:
            - area:
                The surface area of the panel in meters squared.
            - cost:
                The cost of the solar panel per unit panel.
            - land_use:
                The land occupied by the panel in meters squared.
            - name:
                The name to assign to the :class:`SolarPanel` in order to uniquely
                identify it.

        """

        self.area: float = area
        self.land_use: float = land_use
        self.name: str = name

        super().__init__(cost)

    def __init_subclass__(cls, panel_type: SolarPanelType) -> None:
        """
        The init_subclass hook, run on instantiation of the :class:`SolarPanel`.

        Inputs:
            - panel_type:
                The type of panel being considered.

        Outputs:
            An instantiated :class:`SolarPanel` instance.

        """

        cls.panel_type = panel_type

        return super().__init_subclass__()

    @abc.abstractmethod
    def calculate_performance(
        self,
        ambient_temperature: float,
        logger: Logger,
        solar_irradiance: float,
        htf_heat_capacity: float | None,
        input_temperature: float | None,
        mass_flow_rate: float | None,
        wind_speed: float | None,
    ) -> Tuple[float | None, float | None, float | None, float | None,]:
        """
        Abstract method for calculation of collector performance.

        Inputs:
            - ambient_temperature:
                The ambient temperature, measured in degrees Kelvin.
            - htf_heat_capacity:
                The heat capacity of the HTF entering the collector, measured in Joules
                per kilogram Kelvin (J/kgK).
            - input_temperature:
                The input temperature of the HTF entering the collector, measured in
                in degrees Kelvin.
            - logger:
                The :class:`logging.Logger` to use for the run.
            - mass_flow_rate:
                The mass-flow rate of HTF passing through the collector, measured in
                kilograms per second.
            - solar_irradiance:
                The solar irradiance incident on the surface of the collector, measured
                in Watts per meter squared.
            - wind_speed:
                The wind speed in meters per second.

        Outputs:
            - electrical_efficiency:
                The electrical efficiency of the PV panel.
            - output_temperature:
                The output temperature of the HTF leaving the collector.
            - reduced_temperature:
                The reduced temperature of the collector.
            - thermal_efficiency:
                The thermal efficiency of the collector.

        """

    @classmethod
    @abc.abstractmethod
    def from_dict(
        cls,
        logger: Logger,
        solar_inputs: dict[str, Any],
    ) -> Any:
        """
        Instantiate a :class:`SolarPanel` instance based on the input data.

        Inputs:
            - logger:
                The :class:`logging.Logger` to use for the run.
            - solar_inputs:
                The solar input data specific to this panel.

        """


class PVPanel(SolarPanel, panel_type=SolarPanelType.PV):
    """
    Represents a photovoltaic panel.

    .. attribute:: absorptivity
        The absorptivity of the panel, defined between 0 (none of the incident light is
        absorbed by the panel) to 1 (all of the incident light is absorbed by the
        panel).

    .. attribute:: emissivity
        The emissivity of the panel, defined between 0 (none of the incident light on
        the panel is emitted) to 1 (the panel radiates all incoming light).

    .. attribute:: pv_unit
        The unit of PV power being considered, defaulting to 1 kWp.

    .. attribute:: reference_efficiency
        The efficiency of the PV layer under standard test conditions.

    .. attribute:: reference_temperature
        The reference temperature of the PV layer of the panel, measured in degrees
        Kelvin.

    .. attribute:: thermal_coefficient
        The thermal coefficient of performance of the PV layer of the panel, measured in
        kelvin^(-1).

    """

    def __init__(
        self,
        absorptivity: float,
        area: float,
        cost: float,
        emissivity: float,
        land_use: float,
        name: str,
        pv_unit: float,
        reference_efficiency: float,
        reference_temperature: float,
        thermal_coefficient: float,
    ) -> None:
        """
        Instantiate a :class:`PVPanel` instance.

        Inputs:
            - absorptivity:
                The absorptivity of the panel.
            - area:
                The surface area of the panel in meters squared.
            - cost:
                The cost of the :class:`PVPanel`.
            - emissivity:
                The emissivity of the panel.
            - land_use:
                The land occupied by the panel in meters squared.
            - name:
                The name to assign to the :class:`PVPanel` in order to uniquely identify
                it.
            - pv_unit:
                The output power, in Watts, of the PV layer of the panel per unit panel
                installed.
            - reference_efficiency:
                The reference efficiency of the panel, if required, otherwise `None`.
            - reference_temperature:
                The temperature, in degrees Celsius, at which the reference efficiency
                is defined, if required, otherwise `None`.
            - thermal_coefficient:
                The thermal coefficient of the PV layer of the panel, if required,
                otherwise `None`.

        """

        super().__init__(
            area,
            cost,
            land_use,
            name,
        )

        self.absorptivity = absorptivity
        self.emissivity = emissivity
        self.pv_unit: float = pv_unit
        self.reference_efficiency: float = reference_efficiency
        self._reference_temperature: float = reference_temperature
        self.thermal_coefficient: float = thermal_coefficient

    @property
    def reference_temperature(self) -> float:
        """
        Return the reference temperature in degrees Kelvin.

        Outputs:
            The reference temperature in degrees Kelvin.

        """

        return self._reference_temperature + ZERO_CELCIUS_OFFSET

    def __repr__(self) -> str:
        """
        Return a nice-looking representation of the panel.

        Outputs:
            - A nice-looking representation of the panel.

        """

        return (
            "PVPanel("
            + f"name={self.name}"
            + f", area={self.area:.2g}"
            + f", reference_efficiency={self.reference_efficiency:.2g}"
            + f", reference_temperature={self.reference_temperature:.2g} K"
            + f" ({self._reference_temperature:.2g} degC"
            + f", thermal_coefficient={self.thermal_coefficient:.2g}"
            + ")"
        )

    @classmethod
    def from_dict(cls, logger: Logger, solar_inputs: dict[str, Any]) -> Any:
        """
        Instantiate a :class:`PVPanel` instance based on the input data.

        Inputs:
            - logger:
                The logger to use for the run.
            - solar_inputs:
                The solar input data for the panel.

        Outputs:
            A :class:`PVPanel` instance.

        """

        logger.debug("Attempting to create PVPanel from solar input data.")
        try:
            solar_inputs.pop("type")
        except KeyError:
            pass

        return cls(
            **solar_inputs,
        )

    def _average_module_temperature(
        self,
        ambient_temperature: float,
        logger: Logger,
        solar_irradiance: float,
        wind_speed: float,
    ) -> float:
        """
        Calculate the temperature of the PV module in Kelvin.

        Uses a heat-balance calculation iteratively to solve for the average temperature of
        the PV module:
            0W = a_pv G (1 - eta_el)
                 + h_air (T_air - T_pv)
                 + e' sigma (T_sky - Tpv),
            where:
                a_pv    is the absorptivity of the collector,
                G       the solar irradiance in Watts per meter squared,
                eta_el  the electrical efficiency of the collector,
                h_air   the conductive heat-transfer coefficient between the collector
                        and the surrounding air,
                e'      the effective emissivity made linear by combining temperature
                        terms of higher orders,
            and sigma   is the Stefan-Boltzman coefficient.

        This can be rearranged to the form employed here:
            T_pv = (
                a_pv G (1 - eta_ref (1 + beta T_ref))
                + h_air T_amb
                + e' sigma T_sky
            ) / (
                e' sigma + h_air - a_pv beta eta_ref G
            )
        which is linear in temperature and can be solved iteratively as the value of e'
        contains higher-order terms in temperature.

        Inputs:
            - ambient_temperature:
                The ambient temperature, measured in degrees Kelvin.
            - logger:
                The logger to use for the run.
            - solar_irradiance:
                The solar irradiance incident on the collector, measured in Watts per
                meter squared.
            - wind_speed:
                The wind speed in meters per second.

        Outputs:
            The average temperature of the PV module installed.

        """

        # Calculate variable values which remain constant throughout the iteration
        sky_temperature: float = _sky_temperature(ambient_temperature)

        # Setup inputs for the iterative loop
        best_guess_average_temperature: float = ambient_temperature
        solution_found: bool = False

        # Loop through until a solution is found
        logger.debug("Beginning iterative calculation for the PV module temperature.")
        while not solution_found:
            # Compute the necessary coefficients
            radiation_to_sky_coefficient = _radiation_to_sky_coefficient(
                self.emissivity, best_guess_average_temperature, sky_temperature
            )

            # Calculate the average temperature of the collector
            average_temperature: float = (
                (self.absorptivity * solar_irradiance)
                * (
                    1
                    - self.reference_efficiency
                    * (1 + self.thermal_coefficient * self.reference_temperature)
                )
                + _conductive_air_heat_transfer_coefficient(wind_speed)
                * ambient_temperature
                + radiation_to_sky_coefficient * sky_temperature
            ) / (
                radiation_to_sky_coefficient
                + _conductive_air_heat_transfer_coefficient(wind_speed)
                - self.absorptivity
                * self.thermal_coefficient
                * self.reference_efficiency
                * solar_irradiance
            )

            # Break if this average temperature is within the required precision.
            if (
                abs(best_guess_average_temperature - average_temperature)
                < TEMPERATURE_PRECISION
            ):
                solution_found = True
                logger.debug(
                    "Solution for PV-module temperature of %s found, returning.",
                    average_temperature,
                )

            best_guess_average_temperature = average_temperature
            logger.debug(
                "No solution found yet, re-iterating. Best guess: %s",
                best_guess_average_temperature,
            )

        return average_temperature

    def calculate_performance(
        self,
        ambient_temperature: float,
        logger: Logger,
        solar_irradiance: float,
        htf_heat_capacity: float | None = None,
        input_temperature: float | None = None,
        mass_flow_rate: float | None = None,
        wind_speed: float | None = None,
    ) -> Tuple[float | None, float | None, float | None, float | None,]:
        """
        Calcuates the electrical performance of the collector.

        NOTE: The default temperature characteristics from
          Faiman, D. (2008). "Assessing the outdoor operating temperature of
          photovoltaic modules." Progress in Photovoltaics 16(4): 307-315.
        as implemented in the pvlib module are provided in a commented-out form should a
        developer or user which to use this calculation in-lieu of the in-house
        calculation.

        Should non-Silicon panels wish to be considered, the default valuesfor the pvlib
        calculation should be altered accordingly or a more-accurate module-temperature
        module employed. For the in-house calculation, the parameters for the collector
        should be adjusted in the inputs file.

        Inputs:
            - ambient_temperature:
                The ambient temperature, measured in degrees Kelvin.
            - htf_heat_capacity:
                The heat capacity of the HTF entering the collector, measured in Joules
                per kilogram Kelvin (J/kgK).
            - input_temperature:
                The input temperature of the HTF entering the collector, measured in
                in degrees Kelvin.
            - logger:
                The :class:`logging.Logger` to use for the run.
            - mass_flow_rate:
                The mass-flow rate of HTF passing through the collector, measured in
                kilograms per second.
            - solar_irradiance:
                The solar irradiance incident on the surface of the collector, measured
                in Watts per meter squared.
            - wind_speed:
                The wind speed in meters per second.

        Outputs:
            - electrical_efficiency:
                The electrical efficiency of the PV panel.
            - `None`:
                There is no HTF outputted from a PV panel.
            - average_temperature:
                The average temperature of the collector.
            - `None`:
                There is no thermal efficiency associated with a PV panel.

        """

        # Determine the average temperature of the panel using empirical PVLib modules.
        # average_temperature = temperature.faiman(
        #     solar_irradiance, ambient_temperature - ZERO_CELCIUS_OFFSET
        # )

        # Determine the average temperature of the panel using an in-house heat-balance
        # calculation.
        average_temperature = self._average_module_temperature(
            ambient_temperature, logger, solar_irradiance, wind_speed
        )

        # Determine the fractional electrical performance and electrical efficiency of
        # the PV panel.
        electrical_efficiency = self.reference_efficiency * (
            1
            - self.thermal_coefficient
            * (ambient_temperature - self.reference_temperature)
        )

        # Determine the reduced temperature of the panel.
        reduced_panel_temperature = reduced_temperature(
            ambient_temperature,
            average_temperature + ZERO_CELCIUS_OFFSET,
            solar_irradiance,
        )

        return electrical_efficiency, None, average_temperature, None


class HybridPVTPanel(SolarPanel, panel_type=SolarPanelType.PV_T):
    """
    Represents a PV-T panel.

    .. attribute:: electric_performance_curve:
        The electrical performance curve of the panel.

    .. attribute:: max_mass_flow_rate
        The maximum mass-flow rate of heat-transfer fluid through the PV-T collector,
        measured in litres per hour.

    .. attribute:: min_mass_flow_rate
        The minimum mass-flow rate of heat-transfer fluid through the PV-T collector,
        measured in litres per hour.

    .. attribute:: thermal_performance_curve
        The thermal performance curve of the panel.

    """

    def __init__(
        self,
        electric_performance_curve: PerformanceCurve,
        pv_module_characteristics: PVModuleCharacteristics,
        solar_inputs: dict[str, Any],
        thermal_performance_curve: PerformanceCurve,
    ) -> None:
        """
        Instantiate a :class:`HybridPVTPanel` instance based on the input data.

        Inputs:
            - electric_performance_curve:
                The electric performance curve associated with the panel.
            - logger:
                The logger to use for the run.
            - solar_inputs:
                The solar input data specific to this panel.
            - thermal_performance_curve:
                The thermal performance curve associated with the panel.

        """

        super().__init__(
            solar_inputs[AREA],
            solar_inputs[COST],
            solar_inputs[LAND_USE],
            solar_inputs[NAME],
        )

        self.electric_performance_curve: PerformanceCurve = electric_performance_curve
        self._max_mass_flow_rate = solar_inputs[MAX_MASS_FLOW_RATE]
        self._min_mass_flow_rate = solar_inputs[MIN_MASS_FLOW_RATE]
        self.pv_module_characteristics = pv_module_characteristics
        self.thermal_performance_curve: PerformanceCurve = thermal_performance_curve

    @property
    def max_mass_flow_rate(self) -> float | None:
        """
        Return the maximum mass flow rate in kg/s.

        Outputs:
            The maximum mass flow rate of HTF through the collectors in kg/s.

        """

        return (
            self._max_mass_flow_rate / 3600
            if self._max_mass_flow_rate is not None
            else None
        )

    @property
    def min_mass_flow_rate(self) -> float:
        """
        Return the minimum mass flow rate in kg/s.

        Outputs:
            The minimum mass flow rate of HTF through the collectors in kg/s.

        """

        return (
            self._min_mass_flow_rate / 3600
            if self._min_mass_flow_rate is not None
            else None
        )

    def __repr__(self) -> str:
        """
        Return a nice-looking representation of the panel.

        Outputs:
            - A nice-looking representation of the panel.

        """

        return (
            "HybridPVTPanel("
            + f"name={self.name}"
            + (
                (
                    f", max_mass_flow_rate={self.max_mass_flow_rate:.2g} kg/s"
                    + f" ({self._max_mass_flow_rate:.2g} l/h)"
                )
                if self.max_mass_flow_rate is not None
                else ""
            )
            + (
                (
                    f", min_mass_flow_rate={self.min_mass_flow_rate:.2g} kg/s"
                    + f" ({self._min_mass_flow_rate:.2g} l/h)"
                )
                if self.min_mass_flow_rate is not None
                else ""
            )
            + ")"
        )

    @classmethod
    def from_dict(
        cls,
        logger: Logger,
        solar_inputs: dict[str, Any],
    ) -> Any:
        """
        Instantiate a :class:`SolarThermalPanel` instance based on the input data.

        Inputs:
            - logger:
                The :class:`logging.Logger` to use for the run.
            - solar_inputs:
                The solar input data specific to this panel.

        """

        logger.debug("Attempting to create SolarThermalPanel from solar input data.")

        try:
            thermal_performance_inputs = solar_inputs[THERMAL_PERFORMANCE_CURVE]
        except KeyError:
            logger.error(
                "No performance curve defined for solar-thermal panel '%s'.",
                solar_inputs["name"],
            )
            raise InputFileError(
                "solar generation inputs",
                f"Solar thermal panel {solar_inputs.get(NAME, '<not supplied>')} "
                "is missing a thermal performance curve.",
            ) from None

        try:
            thermal_performance_curve = PerformanceCurve(
                thermal_performance_inputs[ZEROTH_ORDER],
                thermal_performance_inputs[FIRST_ORDER],
                thermal_performance_inputs[SECOND_ORDER],
            )
        except KeyError as exception:
            logger.error(
                "Missing thermal performance curve input(s): %s",
                str(exception),
            )
            raise

        if ELECTRIC_PERFORMANCE_CURVE in solar_inputs:
            electric_performance_inputs = solar_inputs[ELECTRIC_PERFORMANCE_CURVE]
            try:
                electric_performance_curve: PerformanceCurve | None = PerformanceCurve(
                    electric_performance_inputs[ZEROTH_ORDER],
                    electric_performance_inputs[FIRST_ORDER],
                    electric_performance_inputs[SECOND_ORDER],
                )
            except KeyError as exception:
                logger.error(
                    "Missing electric performance curve input(s): %s",
                    str(exception),
                )
                raise
        else:
            electric_performance_curve = None
            logger.debug(
                "No performance curve defined for solar-thermal panel '%s'.",
                solar_inputs["name"],
            )

        if PV_MODULE_CHARACTERISTICS in solar_inputs:
            pv_module_characteristics_inputs = solar_inputs[PV_MODULE_CHARACTERISTICS]
            try:
                pv_module_characteristics: PVModuleCharacteristics | None = (
                    PVModuleCharacteristics(
                        pv_module_characteristics_inputs[NOMINAL_POWER],
                        pv_module_characteristics_inputs[REFERENCE_EFFICIENCY],
                        pv_module_characteristics_inputs[REFERENCE_TEMPERATURE],
                        pv_module_characteristics_inputs[THERMAL_COEFFICIENT],
                    )
                )
            except KeyError as exception:
                logger.error(
                    "Missing electric performance curve input(s): %s",
                    str(exception),
                )
                raise
        else:
            logger.debug(
                "No performance curve defined for solar-thermal panel '%s'.",
                solar_inputs["name"],
            )
            pv_module_characteristics = None

        return cls(
            electric_performance_curve,
            pv_module_characteristics,
            solar_inputs,
            thermal_performance_curve,
        )

    def calculate_performance(
        self,
        ambient_temperature: float,
        logger: Logger,
        solar_irradiance: float,
        htf_heat_capacity: float | None,
        input_temperature: float | None,
        mass_flow_rate: float | None,
        wind_speed: float | None = None,
    ) -> Tuple[float | None, float | None, float | None, float | None,]:
        """
        Calculates the performance characteristics of the hybrid PV-T collector.

        The technical PV-T model developed by Benedict Winchester is reduced to a
        smaller, quick-to-run model which is loaded and utilised here.

        Inputs:
            - ambient_temperature:
                The ambient temperature, measured in degrees Kelvin.
            - htf_heat_capacity:
                The heat capacity of the HTF entering the collector, measured in Joules
                per kilogram Kelvin (J/kgK).
            - input_temperature:
                The input temperature of the HTF entering the PV-T collector, measured
                in degrees Kelvin.
            - logger:
                The :class:`logging.Logger` to use for the run.
            - mass_flow_rate:
                The mass-flow rate of HTF passing through the collector, measured in
                kilograms per second.
            - solar_irradiance:
                The solar irradiance incident on the surface of the collector, measured
                in Watts per meter squared.
            - wind_speed:
                The wind speed in meters per second.

        Outputs:
            - electrical_efficiency:
                The electrical efficiency of the PV panel.
            - output_temperature:
                The output temperature of the HTF leaving the collector, measured in degrees
                Kelvin.
            - reduced_temperature:
                The reduced temperature of the collector.
            - thermal_efficiency:
                The thermal efficiency of the collector.

        """

        # Raise a flow-rate error if the flow rate is insufficient.
        if (
            mass_flow_rate < self.min_mass_flow_rate
            if self.min_mass_flow_rate is not None
            else False
        ) or (
            mass_flow_rate > self.max_mass_flow_rate
            if self.max_mass_flow_rate is not None
            else False
        ):
            logger.debug(
                "Flow rate of %s kg/s is out of bounds, range is %s to %s kilograms/second.",
                mass_flow_rate,
                self.min_mass_flow_rate,
                self.max_mass_flow_rate,
            )
            raise FlowRateError(
                self.name,
                f"Flow rate of {mass_flow_rate:.2g} kg/s is out of bounds, range is "
                + f"{self.min_mass_flow_rate:.2g} to {self.max_mass_flow_rate:.2g} "
                "kilograms/second.",
            )
            pass

        _, negative_root = _thermal_performance(
            ambient_temperature,
            self.area,
            htf_heat_capacity,
            input_temperature,
            mass_flow_rate,
            self.thermal_performance_curve,
            solar_irradiance,
        )

        # Compute temperature quantities.
        average_temperature = 0.5 * (negative_root + input_temperature)
        reduced_collector_temperature = reduced_temperature(
            ambient_temperature, average_temperature, solar_irradiance
        )

        # If the reduced temperature is None, performance cannot be computed.
        if reduced_collector_temperature is None:
            return (None, negative_root, None, None)

        # Compute the efficiencies of the collector.
        if self.electric_performance_curve is not None:
            electrical_efficiency = (
                self.electric_performance_curve.eta_0
                + self.electric_performance_curve.c_1 * reduced_collector_temperature
                + self.electric_performance_curve.c_2 * reduced_collector_temperature
            )
        elif self.pv_module_characteristics is not None:
            electrical_efficiency = (
                self.pv_module_characteristics.reference_efficiency
                * (
                    1
                    - self.pv_module_characteristics.thermal_coefficient
                    * (
                        average_temperature
                        - self.pv_module_characteristics.reference_temperature
                    )
                )
            )
        else:
            raise Exception(
                f"PV-T collector {self.name} had neither PV module characteristics or "
                "an electric performance curve."
            )

        thermal_efficiency = (
            self.thermal_performance_curve.eta_0
            + self.thermal_performance_curve.c_1 * reduced_collector_temperature
            + self.thermal_performance_curve.c_2 * reduced_collector_temperature
        )

        # Return the output information.
        return (
            electrical_efficiency,
            negative_root,
            reduced_collector_temperature,
            thermal_efficiency,
        )


class SolarThermalPanel(SolarPanel, panel_type=SolarPanelType.SOLAR_THERMAL):
    """
    Represents a solar-thermal panel.

    .. attribute:: max_mass_flow_rate
        The maximum mass-flow rate of heat-transfer fluid through the PV-T collector,
        measured in litres per hour.

    .. attribute:: min_mass_flow_rate
        The minimum mass-flow rate of heat-transfer fluid through the PV-T collector,
        measured in litres per hour.

    .. attribute:: nominal_mass_flow_rate
        The nominal mass-flow rate of heat-transfer fluid through the PV-T collector,
        measured in litres per hour.

    .. attribute:: stagnation_temperature
        The stagnation temperature of the collector. This is the temperature at which
        the heat losses from the collector are equal to the potential heat gain from any
        incident solar irradiance. At temperatures beyond this value, the collector can
        only output heat-transfer fluid (HTF) at its stagnation temperature.

    .. attribute:: thermal_performance_curve
        The performance curve for the collector.

    """

    def __init__(
        self,
        performance_curve: PerformanceCurve,
        solar_inputs: dict[str, Any],
    ) -> None:
        """
        Instantiate a :class:`SolarThermalPanel` instance based on the input data.

        Inputs:
            - performance_curve:
                The :class:`PeformanceCurve` associated with this panel.
            - solar_inputs:
                The solar input data specific to this panel.

        """

        super().__init__(
            solar_inputs[AREA],
            solar_inputs[COST],
            solar_inputs[LAND_USE],
            solar_inputs[NAME],
        )

        self.area = solar_inputs[AREA]
        self._max_mass_flow_rate = solar_inputs[MAX_MASS_FLOW_RATE]
        self._min_mass_flow_rate = solar_inputs[MIN_MASS_FLOW_RATE]
        self._nominal_mass_flow_rate = solar_inputs.get(NOMINAL_MASS_FLOW_RATE)
        self._stagnation_temperature = solar_inputs[STAGNATION_TEMPERATURE]
        self.thermal_performance_curve = performance_curve

    @property
    def max_mass_flow_rate(self) -> float:
        """
        Return the maximum mass flow rate in kg/s.

        Outputs:
            The maximum mass flow rate of HTF through the collectors in kg/s.

        """

        return (
            self._max_mass_flow_rate / 3600
            if self._max_mass_flow_rate is not None
            else None
        )

    @property
    def min_mass_flow_rate(self) -> float:
        """
        Return the minimum mass flow rate in kg/s.

        Outputs:
            The minimum mass flow rate of HTF through the collectors in kg/s.

        """

        return (
            self._min_mass_flow_rate / 3600
            if self._min_mass_flow_rate is not None
            else None
        )

    @property
    def nominal_mass_flow_rate(self) -> float:
        """
        Return the nominal mass flow rate in kg/s.

        Outputs:
            The nominal mass flow rate of HTF through the collectors in kg/s.

        """

        return (
            self._nominal_mass_flow_rate / 3600
            if self._nominal_mass_flow_rate is not None
            else None
        )

    def __repr__(self) -> str:
        """
        Return a nice-looking representation of the panel.

        Outputs:
            - A nice-looking representation of the panel.

        """

        return (
            "SolarThermalPanel("
            + f"area={self.area}"
            + (
                (
                    f", max_mass_flow_rate={self.max_mass_flow_rate:.2g} kg/s"
                    + f" ({self._max_mass_flow_rate:.2g} l/h)"
                )
                if self.max_mass_flow_rate is not None
                else ""
            )
            + (
                (
                    f", min_mass_flow_rate={self.min_mass_flow_rate:.2g} kg/s"
                    + f" ({self._min_mass_flow_rate:.2g} l/h)"
                )
                if self.min_mass_flow_rate is not None
                else ""
            )
            + f", name={self.name}"
            + (
                (
                    f", nominal_mass_flow_rate={self.nominal_mass_flow_rate:.2g} kg/s"
                    + f" ({self._nominal_mass_flow_rate:.2g} l/h)"
                )
                if self.nominal_mass_flow_rate is not None
                else ""
            )
            + ")"
        )

    @property
    def stagnation_temperature(self) -> float:
        """
        Return the stagnation temperature of the collector in degrees Kelvin.

        Outputs:
            The stagnation temperature of the collector, measured in degrees Kelvin.

        """

        return self._stagnation_temperature + ZERO_CELCIUS_OFFSET

    def calculate_performance(
        self,
        ambient_temperature: float,
        logger: Logger,
        solar_irradiance: float,
        htf_heat_capacity: float | None,
        input_temperature: float | None,
        mass_flow_rate: float | None,
        wind_speed: float | None = None,
    ) -> Tuple[float | None, float | None, float | None, float | None,]:
        """
        Calculates the performance characteristics of the solar-thermal collector.

        Inputs:
            - ambient_temperature:
                The ambient temperature, measured in degrees Kelvin.
            - htf_heat_capacity:
                The heat capacity of the HTF entering the collector, measured in Joules
                per kilogram Kelvin (J/kgK).
            - input_temperature:
                The input temperature of the HTF entering the PV-T collector, measured
                in degrees Kelvin.
            - logger:
                The :class:`logging.Logger` to use for the run.
            - mass_flow_rate:
                The mass-flow rate of HTF passing through the collector, measured in
                kilograms per second.
            - solar_irradiance:
                The solar irradiance incident on the surface of the collector, measured
                in Watts per meter squared.
            - wind_speed:
                The wind speed in meters per second.

        Outputs:
            - `None`:
                There is no electrical component to the collector.
            - output_temperature:
                The output temperature of the HTF leaving the collector, measured in
                degrees Kelvin.
            - reduced_temperature:
                The reduced temperature of the collector.
            - thermal_efficiency:
                The thermal efficiency of the collector.

        Raises:
            - FlowRateError:
                Raised if the flow rates are mismatched.

        """

        # Raise a flow-rate error if the flow rate is insufficient.
        if (
            mass_flow_rate < self.min_mass_flow_rate
            if self.min_mass_flow_rate is not None
            else False
        ) or (
            mass_flow_rate > self.max_mass_flow_rate
            if self.max_mass_flow_rate is not None
            else False
        ):
            logger.debug(
                "Flow rate of %s kg/s is out of bounds, range is %s to %s kilograms/second.",
                mass_flow_rate,
                self.min_mass_flow_rate,
                self.max_mass_flow_rate,
            )
            raise FlowRateError(
                self.name,
                f"Flow rate of {mass_flow_rate:.2g} kg/s is out of bounds, range is "
                + f"{self.min_mass_flow_rate} to {self.max_mass_flow_rate} "
                "kilograms/second.",
            )
            pass

        _, negative_root = _thermal_performance(
            ambient_temperature,
            self.area,
            htf_heat_capacity,
            input_temperature,
            mass_flow_rate,
            self.thermal_performance_curve,
            solar_irradiance,
        )

        # Ensure that the collector has not stagnated.
        if negative_root > self.stagnation_temperature:
            logger.debug(
                "Collector %s outputted HTF at %s degC, above its stagnation "
                "temperature of %s degC",
                self.name,
                negative_root,
                self.stagnation_temperature,
            )
            negative_root = self.stagnation_temperature

        # Compute temperature quantities.
        average_temperature = 0.5 * (negative_root + input_temperature)  # [K]
        reduced_collector_temperature = reduced_temperature(
            ambient_temperature, average_temperature, solar_irradiance
        )  # [K/G]

        # If the reduced temperature is None, the performance cannot be computed.
        if reduced_collector_temperature is None:
            return (None, negative_root, None, None)

        # Compute the thermal efficiency of the collector.
        thermal_efficiency = (
            self.thermal_performance_curve.eta_0
            + self.thermal_performance_curve.c_1 * reduced_collector_temperature
            + self.thermal_performance_curve.c_2
            * solar_irradiance
            * reduced_collector_temperature**2
        )

        return (
            None,
            negative_root,
            reduced_collector_temperature,
            thermal_efficiency,
        )

    @classmethod
    def from_dict(
        cls,
        logger: Logger,
        solar_inputs: dict[str, Any],
    ) -> Any:
        """
        Instantiate a :class:`SolarThermalPanel` instance based on the input data.

        Inputs:
            - logger:
                The :class:`logging.Logger` to use for the run.
            - solar_inputs:
                The solar input data specific to this panel.

        """

        logger.debug("Attempting to create SolarThermalPanel from solar input data.")

        try:
            performance_curve_inputs = solar_inputs[THERMAL_PERFORMANCE_CURVE]
        except KeyError:
            logger.error(
                "No performance curve defined for solar-thermal panel '%s'.",
                solar_inputs["name"],
            )
            raise InputFileError(
                "solar generation inputs",
                f"Solar thermal panel {solar_inputs.get(NAME, '<not supplied>')} is "
                "missing a performance curve.",
            ) from None

        try:
            performance_curve = PerformanceCurve(
                performance_curve_inputs[ZEROTH_ORDER],
                performance_curve_inputs[FIRST_ORDER],
                performance_curve_inputs[SECOND_ORDER],
            )
        except KeyError as exception:
            logger.error(
                "Missing performance curve input(s): %s",
                str(exception),
            )
            raise

        return cls(performance_curve, solar_inputs)


# COLLECTOR_FROM_TYPE:
#   Mapping from collector type to collector.
COLLECTOR_FROM_TYPE: dict[str, SolarPanel] = {
    SolarPanelType.PV: PVPanel,
    SolarPanelType.PV_T: HybridPVTPanel,
    SolarPanelType.SOLAR_THERMAL: SolarThermalPanel,
}


def electric_output(
    electrical_efficiency: float,
    nominal_power: float,
    reference_efficiency: float,
    solar_irradiance: float,
) -> float:
    """
    Compute the electric output power.

    Inputs:
        - electrical_efficiency:
            The electrical efficiency of the collector.
        - nominal_power:
            The nominal output power from the collector, measured in kilo Watts.
        - reference_efficiency:
            The reference efficiency of the collector.
        - solar_irradiance:
            The solar irradiance, measured in Watts per meter squared.

    Outputs:
        The electric output power from the collector, measured in kiloWatts.

    """

    return (
        (electrical_efficiency / reference_efficiency)
        * (solar_irradiance / REFERENCE_SOLAR_IRRADIANCE)
        * nominal_power
    )
