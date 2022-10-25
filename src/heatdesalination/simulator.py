#!/usr/bin/python3.10
########################################################################################
# simulator.py - The simulation module for the HEATDesalination program.               #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 14/10/2022                                                             #
# License: MIT, Open-source                                                            #
# For more information, contact: benedict.winchester@gmail.com                         #
########################################################################################

"""
simulator.py - The simulation module for the HEATDeslination program.

The simulation module is responsible for simnulating the performance of the system over
a given set of input conditions to determine its performance characteristics and
profiles.

"""

from collections import defaultdict
from logging import Logger
from typing import DefaultDict, Dict, Optional, Tuple

from tqdm import tqdm

from .__utils__ import ZERO_CELCIUS_OFFSET, Scenario
from .matrix import solve_matrix
from .plant import DesalinationPlant
from .solar import HybridPVTPanel, PVPanel, SolarThermalPanel
from .storage.storage_utils import HotWaterTank


__all__ = ("run_simulation",)


def _collector_mass_flow_rate(htf_mass_flow_rate: float, system_size: int) -> float:
    """
    Calculate the mass flow rate through the collector.

    Divides the overall mass flow rate by the number of collectors and returns the
    value.

    Inputs:
        - htf_mass_flow_rate:
            The mass flow rate through the system in kg/s.
        - system_size:
            The number of collectors of this particular type which are installed.

    Outputs:
        The mass flow rate through each individual collector.

    """

    return htf_mass_flow_rate / system_size


def _tank_ambient_temperature(ambient_temperature: float) -> float:
    """
    Calculate the ambient temperature of the air surrounding the hot-water tank.

    Inputs:
        - ambient_temperature:
            The ambient temperature.

    Outputs:
        The temperature of the air surrounding the hot-water tank, measured in Kelvin.

    """

    return ambient_temperature


def _tank_replacement_temperature(hour: int) -> float:
    """
    Return the temperature of water which is replacing that taken from the tank.

    Inputs:
        - hour:
            The time of day.

    Outputs:
        The temperature of water replacing that taken from the hot-water tank in Kelvin.

    """

    return ZERO_CELCIUS_OFFSET + 20


def run_simulation(
    ambient_temperatures: Dict[int, float],
    buffer_tank: HotWaterTank,
    desalination_plant: DesalinationPlant,
    htf_mass_flow_rate: float,
    hybrid_pvt_panel: HybridPVTPanel | None,
    logger: Logger,
    pv_panel: PVPanel | None,
    pvt_system_size: int | None,
    scenario: Scenario,
    solar_irradiances: Dict[int, float],
    solar_thermal_collector: SolarThermalPanel | None,
    solar_thermal_system_size: int | None,
    *,
    disable_tqdm: bool = False,
) -> Tuple[
    Dict[int, float],
    Dict[int, float],
    Dict[int, float] | None,
    Dict[int, float] | None,
    Dict[int, float],
]:
    """
    Runs a simulation for the intergrated system specified.

    Inputs:
        - ambient_temperatures:
            The ambient temperature at each time step, measured in Kelvin.
        - buffer_tank:
            The :class:`HotWaterTank` associated with the system.
        - desalination_plant:
            The :class:`DesalinationPlant` for which the systme is being simulated.
        - htf_mass_flow_rate:
            The mass flow rate of the HTF through the collectors.
        - hybrid_pvt_panel:
            The :class:`HybridPVTPanel` associated with the run.
        - logger:
            The :class:`logging.Logger` for the run.
        - pv_panel:
            The :class:`PVPanel` associated with the system.
        - pvt_system_size:
            The size of the PV-T system installed.
        - scenario:
            The :class:`Scenario` for the run.
        - solar_irradiances:
            The solar irradiances at each time step, measured in Kelvin.
        - solar_thermal_collector:
            The :class:`SolarThermalCollector` associated with the run.
        - solar_thermal_system_size:
            The size of the solar-thermal system.

    Outputs:
        - collector_input_temperatures:
            The input temperature to the collector system at each time step.
        - collector_system_output_temperatures:
            The output temperature from the solar collectors at each time step.
        - pvt_htf_output_temperatures:
            The output temperature from the PV-T collectors at each time step.
        - solar_thermal_htf_output_temperatures:
            The output temperature from the solar-thermal collectors at each time step
            if present.
        - tank_temperatures:
            The temperature of the hot-water tank at each time step.

    """

    # Determine the mass flow rate through each type of collector.
    pvt_mass_flow_rate: float = _collector_mass_flow_rate(
        htf_mass_flow_rate, pvt_system_size
    )
    solar_thermal_mass_flow_rate: float = _collector_mass_flow_rate(
        htf_mass_flow_rate, solar_thermal_system_size
    )
    logger.debug("PV-T mass-flow rate determined: %s", f"{pvt_mass_flow_rate:.3g}")
    logger.debug(
        "Solar-thermal mass-flow rate determined: %s",
        f"{solar_thermal_mass_flow_rate:.3g}",
    )

    # Set up maps for storing variables.
    collector_input_temperatures: Dict[int, float] = {}
    collector_system_output_temperatures: Dict[int, float] = {}
    pvt_electrical_efficiencies: Dict[int, float] = {}
    pvt_htf_output_temperatures: Dict[int, float | None] = {}
    pvt_reduced_temperatures: Dict[int, float | None] = {}
    pvt_thermal_efficiencies: Dict[int, float | None] = {}
    solar_thermal_htf_output_temperatures: Dict[int, float | None] = {}
    solar_thermal_reduced_temperatures: Dict[int, float | None] = {}
    solar_thermal_thermal_efficiencies: Dict[int, float | None] = {}
    tank_temperatures: DefaultDict[int, float] = defaultdict(
        lambda: _tank_ambient_temperature(ambient_temperatures[0])
    )

    # At each time step, call the matrix equation solver.
    logger.info("Beginning hourly simulation.")
    for hour in tqdm(
        range(24),
        desc="simulation",
        disable=disable_tqdm,
        leave=False,
        unit="hour",
    ):
        (
            collector_input_temperature,
            collector_system_output_temperature,
            pvt_electrical_efficiency,
            pvt_htf_output_temperature,
            pvt_reduced_temperature,
            pvt_thermal_efficiency,
            solar_thermal_htf_output_temperature,
            solar_thermal_reduced_temperature,
            solar_thermal_thermal_efficiency,
            tank_temperature,
        ) = solve_matrix(
            ambient_temperatures[hour],
            buffer_tank,
            htf_mass_flow_rate,
            hybrid_pvt_panel,
            desalination_plant.requirements(hour).hot_water_volume,
            logger,
            tank_temperatures[hour - 1],
            pvt_mass_flow_rate,
            scenario,
            solar_irradiances[hour],
            solar_thermal_collector,
            solar_thermal_mass_flow_rate,
            _tank_ambient_temperature(ambient_temperatures[hour]),
            _tank_replacement_temperature(hour),
        )

        # Save these outputs in mappings.
        collector_input_temperatures[hour] = collector_input_temperature
        collector_system_output_temperatures[hour] = collector_system_output_temperature
        pvt_electrical_efficiencies[hour] = pvt_electrical_efficiency
        pvt_htf_output_temperatures[hour] = pvt_htf_output_temperature
        pvt_reduced_temperatures[hour] = pvt_reduced_temperature
        pvt_thermal_efficiencies[hour] = pvt_thermal_efficiency
        solar_thermal_htf_output_temperatures[
            hour
        ] = solar_thermal_htf_output_temperature
        solar_thermal_reduced_temperatures[hour] = solar_thermal_reduced_temperature
        solar_thermal_thermal_efficiencies[hour] = solar_thermal_thermal_efficiency
        tank_temperatures[hour] = tank_temperature

    if scenario.pv:
        logger.info("Computing PV performance characteristics.")
        pv_electrical_efficiencies: Dict[int, float] | None = {
            hour: pv_panel.calculate_performance(
                ambient_temperatures[hour], logger, solar_irradiances[hour]
            )
            for hour in tqdm(
                range(24), desc="pv performance", leave=disable_tqdm, unit="hour"
            )
        }
    else:
        pv_electrical_efficiencies = None

    logger.info("Hourly simulation complete, returning outputs.")
    return (
        collector_input_temperatures,
        collector_system_output_temperatures,
        pv_electrical_efficiencies,
        pvt_electrical_efficiencies if scenario.pv_t else None,
        pvt_htf_output_temperatures if scenario.pv_t else None,
        pvt_reduced_temperatures if scenario.pv_t else None,
        pvt_thermal_efficiencies if scenario.pv_t else None,
        solar_thermal_htf_output_temperatures if scenario.solar_thermal else None,
        solar_thermal_reduced_temperatures if scenario.solar_thermal else None,
        solar_thermal_thermal_efficiencies if scenario.solar_thermal else None,
        {key: value for key, value in tank_temperatures.items() if 0 <= key < 24},
    )
