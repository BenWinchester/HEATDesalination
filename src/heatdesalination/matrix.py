#!/usr/bin/python3.10
########################################################################################
# matrix.py - The matrix construction and solving module                               #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 14/10/2022                                                             #
# License: MIT, Open-source                                                            #
# For more information, contact: benedict.winchester@gmail.com                         #
########################################################################################

"""
matrix.py - The matrix module for the HEATDeslination program.

The matrix module is responsible for constructing and solving matrix equations required
to simulate the performance of the various components of the heat-supplying solar
system which consists of solar-thermal and PV-T collectors depending on the
configuration.

"""

from logging import Logger
from typing import Tuple


from .__utils__ import (
    ZERO_CELCIUS_OFFSET,
    InputFileError,
    Scenario,
    TEMPERATURE_PRECISION,
)
from .solar import HybridPVTPanel, PVPanel, SolarThermalPanel
from .storage.storage_utils import HotWaterTank

__all__ = ("solve_matrix",)


def _collectors_input_temperature(
    collector_system_output_temperature: float,
    heat_exchanger_efficiency: float,
    htf_heat_capacity: float,
    tank_temperature: float,
    tank_water_heat_capacity: float,
) -> float:
    """
    Calculates the input temperature for the collectors based on tank outputs etc.

    Inputs:
        - collector_system_output_temperature:
            The output temperature from the solar system overall, measured in Kelvin.
        - heat_exchanger_efficiency:
            The efficiency of the heat exchanger.
        - htf_heat_capacity:
            The heat capacity of the HTF, measured in Joules per kilogram Kelvin.
        - tank_temperature:
            The temperature of the contents of the hot-water tank, measured in Kelvin.
        - tank_water_heat_capacity:
            The heat capacity of the water within the hot-water tank.

    Outputs:
        The input temperature of the HTF to the collectors.

    """

    return collector_system_output_temperature + (
        tank_water_heat_capacity * heat_exchanger_efficiency / htf_heat_capacity
    ) * (tank_temperature - collector_system_output_temperature)


def _solar_system_output_temperatures(
    ambient_temperature: float,
    collector_system_input_temperature: float,
    hybrid_pv_t_panel: HybridPVTPanel | None,
    logger: Logger,
    pv_t_mass_flow_rate: float | None,
    scenario: Scenario,
    solar_irradiance: float,
    solar_thermal_collector: SolarThermalPanel | None,
    solar_thermal_mass_flow_rate: float | None,
) -> Tuple[
    float,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    """
    Calculates the output temperatures of the PV-T, solar-thermal and overall solar.

    Inputs:
        - ambient_temperature:
            The ambient temperature, measured in Kelvin.
        - collector_system_input_temperature:
            The intput temperature to the collector system, measured in Kelvin.
        - hybrid_pv_t_panel:
            The :class:`HybridPVTPanel` to use for the run if appropriate.
        - logger:
            The :class:`logging.Logger` to use for the run.
        - pv_t_mass_flow_rate:
            The mass flow rate of HTF through the PV-T collectors, measured in kg/s.
        - scenario:
            The :class:`Scenario` to use for the run.
        - solar_irradiance:
            The solar irradiance, measured in Watts per meter squared.
        - solar_thermal_collector:
            The :class:`SolarThermalPanel` to use for the run if appropriate.
        - solar_thermal_mass_flow_rate:
            The mass flow rate of HTF through the solar-thermal collectors, measured in
            kg/s.

    Outputs:
        - collector_system_output_temperature:
            The output temperature from the solar system overall, measured in Kelvin.
        - pv_t_htf_output_temperature:
            The output temperature from the PV-T collectors, measured in Kelvin;
        - solar_thermal_htf_output_temperature:
            The output temperature from the solar-thermal collectors, measured in
            Kelvin.

    """

    if scenario.pv_t and pv_t_mass_flow_rate is not None:
        (
            pv_t_electrical_efficiency,
            pv_t_htf_output_temperature,
            pv_t_reduced_temperature,
            pv_t_thermal_efficiency,
        ) = hybrid_pv_t_panel.calculate_performance(
            ambient_temperature,
            logger,
            solar_irradiance,
            scenario.htf_heat_capacity,
            collector_system_input_temperature,
            pv_t_mass_flow_rate,
        )
    else:
        pv_t_electrical_efficiency = None
        pv_t_htf_output_temperature = None
        pv_t_reduced_temperature = None
        pv_t_thermal_efficiency = None

    if scenario.solar_thermal and solar_thermal_mass_flow_rate is not None:
        (
            _,
            solar_thermal_htf_output_temperature,
            solar_thermal_reduced_temperature,
            solar_thermal_thermal_efficiency,
        ) = solar_thermal_collector.calculate_performance(
            ambient_temperature,
            logger,
            solar_irradiance,
            scenario.htf_heat_capacity,
            pv_t_htf_output_temperature
            if pv_t_htf_output_temperature is not None
            else collector_system_input_temperature,
            solar_thermal_mass_flow_rate,
        )
    else:
        solar_thermal_htf_output_temperature = None
        solar_thermal_reduced_temperature = None
        solar_thermal_thermal_efficiency = None

    # Determine the output temperature from the whole solar collector system.
    if solar_thermal_htf_output_temperature is not None:
        collector_system_output_temperature: float = (
            solar_thermal_htf_output_temperature
        )
    elif pv_t_htf_output_temperature is not None:
        collector_system_output_temperature = pv_t_htf_output_temperature
    else:
        collector_system_output_temperature = collector_system_input_temperature
        logger.info("Neither PV-T or solar-thermal were requested.")

    return (
        collector_system_output_temperature,
        pv_t_electrical_efficiency,
        pv_t_htf_output_temperature,
        pv_t_reduced_temperature,
        pv_t_thermal_efficiency,
        solar_thermal_htf_output_temperature,
        solar_thermal_reduced_temperature,
        solar_thermal_thermal_efficiency,
    )


def _tank_temperature(
    buffer_tank: HotWaterTank,
    collector_system_output_temperature: float,
    heat_exchanger_efficiency: float,
    htf_heat_capacity: float,
    htf_mass_flow_rate: float,
    load_mass_flow_rate: float,
    logger: Logger,
    previous_tank_temperature: float,
    tank_ambient_temperature: float,
    tank_replacement_water_temperature: float,
    tank_water_heat_capacity: float,
    time_interval: int = 3600,
) -> Tuple[bool, float]:
    """
    Calculate the temperature of the buffer tank.

    Inputs:
        - buffer_tank:
            The :class:`HotWaterTank` to use as a buffer tank for the run.
        - collector_system_output_temperature:
            The temperature of the HTF leaving the collector system, measured in Kelvin.
        - heat_exchanger_efficiency:
            The efficiency of the heat exchanger.
        - htf_heat_capacity:
            The heat capacity of the HTF, measured in Joules per kilogram Kelvin.
        - htf_mass_flow_rate:
            The mass flow rate of the HTF, measured in kilograms per second.
        - load_mass_flow_rate:
            The mass flow rate of the load, measuted in kilograms per second.
        - logger:
            The :class:`logging.Logger` to use for the run.
        - previous_tank_temperature:
            The temperature of the hot-water tank at the previous time step, measured in
            Kelvin.
        - tank_ambient_temperature:
            The ambient temperature of air surrounding the hot-water tank, measured in
            Kelvin.
        - tank_replacement_water_temperature:
            The temperature in Kelvin of the replacement water used to replace water
            withdrawn from the hot-water tank.
        - tank_water_heat_capacity:
            The heat capacity of the water within the hot-water tank.
        - time_interval:
            The time interval being considered, measured in seconds.

    Outputs:
        - collectors_connected:
            Whether the collectors are connected to the tank (True) or disconnected
            (False).
        - tank_temperature:
            The temperature of the buffer tank, measured in Kelvin.

    """

    # Heat remaining in the tank from the previous time step.
    tank_heat_capacity_term: float = (
        buffer_tank.mass * buffer_tank.heat_capacity
    )  # [J/K]
    logger.debug("Tank heat-capacity term: %s J/K", f"{tank_heat_capacity_term:.3g}")

    # Heat transfer through the heat exhcanger between the tank and HTF.
    heat_exchanger_term: float = (
        htf_mass_flow_rate  # [kg/s]
        * htf_heat_capacity  # [J/kgK]
        * heat_exchanger_efficiency
        * time_interval  # [s]
    )  # [J/K]
    logger.debug("Heat exhcnager term: %s J/K", f"{heat_exchanger_term:.3g}")

    # Heat transfer due to the applied load.
    hot_water_load_term: float = (
        load_mass_flow_rate  # [kg/s]
        * tank_water_heat_capacity  # [J/kgK]
        * time_interval  # [s]
    )  # [J/K]
    logger.debug("Hot water load term: %s J/K", f"{hot_water_load_term:.3g}")

    # Heat transfer to and from the environment around the tank.
    environment_heat_transfer_term: float = (
        buffer_tank.heat_transfer_coefficient * time_interval  #  [W/K]  # [s]
    )  # [J/K]
    logger.debug(
        "Tank environment heat-transfer coefficient: %s J/K",
        f"{environment_heat_transfer_term:.3g}",
    )

    # Calculate and return the tank temperature.
    # NOTE: The tank should only interact with the collectors if the temperature of the
    # HTF is higher than the temperature of the tank, i.e., if heat would be added to
    # the tank at this time step. Otherwise, the tank should simply lose heat to the
    # environment.
    predicted_tank_temperature: float = (
        tank_heat_capacity_term * previous_tank_temperature
        + heat_exchanger_term * collector_system_output_temperature
        + hot_water_load_term * tank_replacement_water_temperature
        + environment_heat_transfer_term * tank_ambient_temperature
    ) / (
        tank_heat_capacity_term
        + heat_exchanger_term
        + hot_water_load_term
        + environment_heat_transfer_term
    )

    # If the tank is cooler than the HTF, it gains heat from the collectors.
    if collector_system_output_temperature >= predicted_tank_temperature:
        return True, predicted_tank_temperature

    # Otherwise, the tank should remain decoupled from the collectors.
    return False, (
        tank_heat_capacity_term * previous_tank_temperature
        + hot_water_load_term * tank_replacement_water_temperature
        + environment_heat_transfer_term * tank_ambient_temperature
    ) / (tank_heat_capacity_term + hot_water_load_term + environment_heat_transfer_term)


def solve_matrix(
    ambient_temperature: float,
    buffer_tank: HotWaterTank,
    htf_mass_flow_rate: float,
    hybrid_pv_t_panel: HybridPVTPanel | None,
    load_mass_flow_rate: float,
    logger: Logger,
    previous_tank_temperature: float,
    pv_t_mass_flow_rate: float | None,
    scenario: Scenario,
    solar_irradiance: float,
    solar_thermal_collector: SolarThermalPanel | None,
    solar_thermal_mass_flow_rate: float | None,
    tank_ambient_temperature: float,
    tank_replacement_water_temperature: float,
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
    """
    Solve the matrix equation for the performance of the solar system.

    Inputs:
        - ambient_temperature:
            The ambient temperature measured in Kelvin.
        - buffer_tank:
            The :class:`HotWaterTank` to use as a buffer tank for the run.
        - htf_mass_flow_rate:
            The mass flow rate of the HTF, measured in kilograms per second.
        - hybrid_pv_t_panel:
            The :class:`HybridPVTPanel` to use for the run if appropriate.
        - load_mass_flow_rate:
            The mass flow rate of the load, measuted in kilograms per second.
        - logger:
            The :class:`logging.Logger` to use for the run.
        - previous_tank_temperature:
            The temperature of the hot-water tank at the previous time step.
        - pv_t_mass_flow_rate:
            The mass flow rate of the HTF through the PV-T collectors, measured in
            kilograms per second.
        - scenario:
            The :class:`Scenario` to use for the run.
        - solar_irradiance:
            The solar irradiance.
        - solar_thermal_collector:
            The :class:`SolarThermalPanel` to use for the run if appropriate.
        - solar_thermal_mass_flow_rate:
            The mass flow rate of the HTF through the solar-thermal collectors, measured
            in kilograms per second.
        - tank_ambient_temperature:
            The ambient temperature of air surrounding the hot-water tank, measured in
            Kelvin.
        - tank_replacement_water_temperature:
            The temperature of the water which will replace any hot water consumed from
            the tank during the time interval.

    Outputs:
        - collector_input_temperature:
            The input temperature to the collector system, measured in degrees Celcius.
        - collector_system_output_temperature:
            The output temperature from the solar collector system as a whole, measured
            in degrees Kelvin.
        - pv_t_electrical_efficiency:
            The electrical efficiency of the PV-T panels, if present, defined between 0
            and 1.
        - pv_t_htf_output_temperature:
            The output temperature from the PV-T panels, if present, measured in degrees
            Kelvin.
        - pv_t_reduced_temperature:
            The reduced temperature of the PV-T collectors, if present.
        - pv_t_thermal_efficiency:
            The thermal efficiency of the PV-T collectors, if present.
        - solar_thermal_htf_output_temperature:
            The output temperature from the solar-thermal collectors, if present,
            measured in degrees Kelvin.
        - solar_thermal_reduced_temperature:
            The reduced temperature of the solar-thermal collectors, if present.
        - solar_thermal_thermal_efficiency:
            The thermal efficiency of the solar-thermal collectors, if present.
        - tank_temperature:
            The temperature of the tank, measured in degrees Kelvin.

    """

    # Set up variables to track for a valid solution being found.
    if scenario.pv_t and hybrid_pv_t_panel is None:
        logger.error("No PV-T panel provided despite PV-T being on in the scenario.")
        raise InputFileError(
            "scenario OR solar",
            "No PV-T panel provided despite PV modelling being requested.",
        )
    if scenario.solar_thermal and (solar_thermal_collector is None):
        logger.error(
            "No solar-thermal collector provided despite solar-thermal being on in the "
            "scenario."
        )
        raise InputFileError(
            "scenario OR solar",
            "No solar-thermal collector provided despite PV modelling being requested.",
        )

    best_guess_collector_htf_input_temperature: float = ambient_temperature
    best_guess_tank_temperature: float = previous_tank_temperature
    solution_found: bool = False

    # Iterate until a valid solution is found within the hard-coded precision.
    while not solution_found:
        # Calculate the various coefficients which go into the matrix.
        (
            collector_system_output_temperature,
            pv_t_electrical_efficiency,
            pv_t_htf_output_temperature,
            pv_t_reduced_temperature,
            pv_t_thermal_efficiency,
            solar_thermal_htf_output_temperature,
            solar_thermal_reduced_temperature,
            solar_thermal_thermal_efficiency,
        ) = _solar_system_output_temperatures(
            ambient_temperature,
            best_guess_collector_htf_input_temperature,
            hybrid_pv_t_panel,
            logger,
            pv_t_mass_flow_rate,
            scenario,
            solar_irradiance,
            solar_thermal_collector,
            solar_thermal_mass_flow_rate,
        )

        # Calculate the tank temperature based on these parameters.
        collectors_connected, tank_temperature = _tank_temperature(
            buffer_tank,
            collector_system_output_temperature,
            scenario.heat_exchanger_efficiency,
            scenario.htf_heat_capacity,
            htf_mass_flow_rate,
            load_mass_flow_rate,
            logger,
            previous_tank_temperature,
            tank_ambient_temperature,
            tank_replacement_water_temperature,
            buffer_tank.heat_capacity,
        )

        # Solve for the input temperature.
        if collectors_connected:
            collector_input_temperature: float = _collectors_input_temperature(
                collector_system_output_temperature,
                scenario.heat_exchanger_efficiency,
                scenario.htf_heat_capacity,
                tank_temperature,
                buffer_tank.heat_capacity,
            )
        else:
            collector_input_temperature = collector_system_output_temperature

        # Check whether the solution is valid given the hard-coded precision specified.
        if all(
            entry < TEMPERATURE_PRECISION
            for entry in (
                abs(
                    collector_input_temperature
                    - best_guess_collector_htf_input_temperature
                ),
                abs(tank_temperature - best_guess_tank_temperature),
            )
        ):
            solution_found = True

        best_guess_collector_htf_input_temperature = collector_input_temperature
        best_guess_tank_temperature = tank_temperature

    # Return the outputs.
    return (
        collector_input_temperature,
        collector_system_output_temperature,
        pv_t_electrical_efficiency,
        pv_t_htf_output_temperature,
        pv_t_reduced_temperature,
        pv_t_thermal_efficiency,
        solar_thermal_htf_output_temperature,
        solar_thermal_reduced_temperature,
        solar_thermal_thermal_efficiency,
        tank_temperature,
    )
