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
from typing import DefaultDict, Dict

from tqdm import tqdm

from .__utils__ import Scenario, Solution, ZERO_CELCIUS_OFFSET
from .heat_pump import calculate_heat_pump_electricity_consumption
from .matrix import solve_matrix
from .plant import DesalinationPlant
from .solar import HybridPVTPanel, PVPanel, SolarThermalPanel, electric_output
from .storage.storage_utils import HotWaterTank


__all__ = ("run_simulation",)


# Temperature precision:
#   The precision required when solving for a steady-state solution.
STEADY_STATE_TEMPERATURE_PRECISION: float = 0.001


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

    The number used is that for the final effect of the plant being considered.

    Inputs:
        - hour:
            The time of day.

    Outputs:
        The temperature of water replacing that taken from the hot-water tank in Kelvin.

    """

    return ZERO_CELCIUS_OFFSET + 40


def run_simulation(
    ambient_temperatures: Dict[int, float],
    buffer_tank: HotWaterTank,
    desalination_plant: DesalinationPlant,
    htf_mass_flow_rate: float,
    hybrid_pv_t_panel: HybridPVTPanel | None,
    logger: Logger,
    pv_panel: PVPanel | None,
    pv_system_size: int | None,
    pv_t_system_size: int | None,
    scenario: Scenario,
    solar_irradiances: Dict[int, float],
    solar_thermal_collector: SolarThermalPanel | None,
    solar_thermal_system_size: int | None,
    *,
    disable_tqdm: bool = False,
    tank_start_temperature: float,
) -> Solution:
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
        - hybrid_pv_t_panel:
            The :class:`HybridPVTPanel` associated with the run.
        - logger:
            The :class:`logging.Logger` for the run.
        - pv_panel:
            The :class:`PVPanel` associated with the system.
        - pv_system_size:
            The size of the PV system installed.
        - pv_t_system_size:
            The size of the PV-T system installed.
        - scenario:
            The :class:`Scenario` for the run.
        - solar_irradiances:
            The solar irradiances at each time step, measured in Kelvin.
        - solar_thermal_collector:
            The :class:`SolarThermalCollector` associated with the run.
        - solar_thermal_system_size:
            The size of the solar-thermal system.
        - disable_tqdm:
            Whether to disable the progress bar.
        - tank_start_temperature:
            The default tank temperature to use for running for consistency.

    Outputs:
        The steady-state solution.

    """

    # Determine the mass flow rate through each type of collector.
    if scenario.pv_t and pv_t_system_size > 0:
        pv_t_mass_flow_rate: float | None = _collector_mass_flow_rate(
            htf_mass_flow_rate, pv_t_system_size
        )
        logger.debug("PV-T mass-flow rate determined: %s", f"{pv_t_mass_flow_rate:.3g}")
    else:
        pv_t_mass_flow_rate = None
        logger.debug("No PV-T mass flow rate because disabled or zero size.")

    if scenario.solar_thermal and solar_thermal_system_size > 0:
        solar_thermal_mass_flow_rate: float = _collector_mass_flow_rate(
            htf_mass_flow_rate, solar_thermal_system_size
        )
        logger.debug(
            "Solar-thermal mass-flow rate determined: %s",
            f"{solar_thermal_mass_flow_rate:.3g}",
        )
    else:
        solar_thermal_mass_flow_rate = None
        logger.debug("No solar-thermal mass flow rate because disabled or zero size.")

    # Set up maps for storing variables.
    collector_input_temperatures: Dict[int, float] = {}
    collector_system_output_temperatures: Dict[int, float] = {}
    electricity_demands: DefaultDict[int, float] = defaultdict(float)
    hot_water_demand_temperatures: Dict[int, float | None] = {}
    hot_water_demand_volumes: Dict[int, float | None] = {}
    pv_t_electrical_efficiencies: Dict[int, float | None] = {}
    pv_t_electrical_output_power: Dict[int, float | None] = {}
    pv_t_htf_output_temperatures: Dict[int, float | None] = {}
    pv_t_reduced_temperatures: Dict[int, float | None] = {}
    pv_t_thermal_efficiencies: Dict[int, float | None] = {}
    solar_thermal_htf_output_temperatures: Dict[int, float | None] = {}
    solar_thermal_reduced_temperatures: Dict[int, float | None] = {}
    solar_thermal_thermal_efficiencies: Dict[int, float | None] = {}
    tank_temperatures: DefaultDict[int, float] = defaultdict(
        lambda: tank_start_temperature
    )

    # At each time step, call the matrix equation solver.
    logger.info("Beginning hourly simulation.")
    for hour in tqdm(
        range(len(ambient_temperatures)),
        desc="simulation",
        disable=disable_tqdm,
        leave=False,
        unit="hour",
    ):
        (
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
        ) = solve_matrix(
            ambient_temperatures[hour],
            buffer_tank,
            htf_mass_flow_rate,
            hybrid_pv_t_panel,
            (
                hot_water_demand_volume := desalination_plant.requirements(
                    hour
                ).hot_water_volume
            ),
            logger,
            tank_temperatures[hour - 1],
            pv_t_mass_flow_rate,
            scenario,
            solar_irradiances[hour],
            solar_thermal_collector,
            solar_thermal_mass_flow_rate,
            _tank_ambient_temperature(ambient_temperatures[hour]),
            _tank_replacement_temperature(hour),
        )

        # Determine the electricity demands of the plant including any auxiliary
        # heating.
        if desalination_plant.operating(hour):
            auxiliary_heating_demand = (
                desalination_plant.requirements(hour).hot_water_volume  # [kg/s]
                * buffer_tank.heat_capacity  # [J/kg*K]
                * (
                    desalination_plant.requirements(hour).hot_water_temperature
                    - tank_temperature  # [K]
                )
                / 1000  # [W/kW]
            )  # [kW]
            electricity_demand: float = desalination_plant.requirements(  # [kW]
                hour
            ).electricity + (
                calculate_heat_pump_electricity_consumption(
                    desalination_plant.requirements(hour).hot_water_temperature,
                    ambient_temperatures[hour],
                    auxiliary_heating_demand,
                    scenario.heat_pump_efficiency,
                )
            )
        else:
            electricity_demand = desalination_plant.requirements(hour).electricity

        # Save these outputs in mappings.
        collector_input_temperatures[hour] = collector_input_temperature
        collector_system_output_temperatures[hour] = collector_system_output_temperature
        electricity_demands[hour] = electricity_demand
        hot_water_demand_temperatures[hour] = desalination_plant.requirements(
            hour
        ).hot_water_temperature
        hot_water_demand_volumes[hour] = hot_water_demand_volume
        pv_t_electrical_efficiencies[hour] = pv_t_electrical_efficiency
        pv_t_electrical_output_power[hour] = (
            electric_output(
                pv_t_electrical_efficiency
                if pv_t_electrical_efficiency is not None
                else 0,
                hybrid_pv_t_panel.pv_module_characteristics.nominal_power,
                hybrid_pv_t_panel.pv_module_characteristics.reference_efficiency,
                solar_irradiances[hour],
            )
            if solar_irradiances[hour] > 0
            else 0
        )
        pv_t_htf_output_temperatures[hour] = pv_t_htf_output_temperature
        pv_t_reduced_temperatures[hour] = pv_t_reduced_temperature
        pv_t_thermal_efficiencies[hour] = pv_t_thermal_efficiency
        solar_thermal_htf_output_temperatures[
            hour
        ] = solar_thermal_htf_output_temperature
        solar_thermal_reduced_temperatures[hour] = solar_thermal_reduced_temperature
        solar_thermal_thermal_efficiencies[hour] = solar_thermal_thermal_efficiency
        tank_temperatures[hour] = tank_temperature

    # Compute the PV performance characteristics.
    if scenario.pv:
        logger.info("Computing PV performance characteristics.")
        pv_electrical_efficiencies: Dict[int, float] | None = {
            hour: pv_panel.calculate_performance(
                ambient_temperatures[hour], logger, solar_irradiances[hour]
            )
            for hour in tqdm(
                range(len(ambient_temperatures)),
                desc="pv performance",
                leave=disable_tqdm,
                unit="hour",
            )
        }
        pv_electrical_output_power: Dict[int, float] | None = {
            hour: (
                electric_output(
                    pv_electrical_efficiencies[hour],
                    pv_panel.pv_unit,
                    pv_panel.reference_efficiency,
                    solar_irradiance,
                )
                if solar_irradiance > 0
                else None
            )
            for hour, solar_irradiance in solar_irradiances.items()
        }
        pv_system_electrical_output_power: Dict[int, float] | None = {
            hour: value * pv_system_size
            for hour, value in pv_electrical_output_power.items()
        }
    else:
        pv_electrical_efficiencies = None
        pv_electrical_output_power = None
        pv_system_electrical_output_power = None

    # Compute the output power from the various collectors.
    logger.info("Hourly simulation complete, compute the output power.")
    pv_t_system_electrical_output_power: Dict[int, float] = {
        hour: value * pv_t_system_size
        for hour, value in pv_t_electrical_output_power.items()
    }

    logger.info("Simulation complete, returning outputs.")
    return Solution(
        ambient_temperatures,
        collector_input_temperatures,
        collector_system_output_temperatures,
        electricity_demands,
        hot_water_demand_temperatures,
        hot_water_demand_volumes,
        pv_electrical_efficiencies if scenario.pv else None,
        pv_electrical_output_power if scenario.pv else None,
        pv_system_electrical_output_power if scenario.pv else None,
        pv_t_electrical_efficiencies if scenario.pv_t else None,
        pv_t_electrical_output_power if scenario.pv_t else None,
        pv_t_htf_output_temperatures if scenario.pv_t else None,
        pv_t_reduced_temperatures if scenario.pv_t else None,
        pv_t_system_electrical_output_power if scenario.pv_t else None,
        pv_t_thermal_efficiencies if scenario.pv_t else None,
        solar_thermal_htf_output_temperatures if scenario.solar_thermal else None,
        solar_thermal_reduced_temperatures if scenario.solar_thermal else None,
        solar_thermal_thermal_efficiencies if scenario.solar_thermal else None,
        {
            key: value
            for key, value in tank_temperatures.items()
            if 0 <= key < len(ambient_temperatures)
        },
    )


def determine_steady_state_simulation(
    ambient_temperatures: Dict[int, float],
    buffer_tank: HotWaterTank,
    desalination_plant: DesalinationPlant,
    htf_mass_flow_rate: float,
    hybrid_pv_t_panel: HybridPVTPanel | None,
    logger: Logger,
    pv_panel: PVPanel | None,
    pv_system_size: int | None,
    pv_t_system_size: int | None,
    scenario: Scenario,
    solar_irradiances: Dict[int, float],
    solar_thermal_collector: SolarThermalPanel | None,
    solar_thermal_system_size: int | None,
    *,
    disable_tqdm: bool = False,
) -> Solution:
    """
    Determines steady-state simulation conditions.

    In order to determine a steady-state solution for the temperatures of the various
    components of the system, the simulation function needs to be run until the start
    and end points of the simulation (at midnight and (+1) midnight, i.e., midnight the
    next day) produce the same results across the board.

    Inputs:
        - ambient_temperatures:
            The ambient temperature at each time step, measured in Kelvin.
        - buffer_tank:
            The :class:`HotWaterTank` associated with the system.
        - desalination_plant:
            The :class:`DesalinationPlant` for which the systme is being simulated.
        - htf_mass_flow_rate:
            The mass flow rate of the HTF through the collectors.
        - hybrid_pv_t_panel:
            The :class:`HybridPVTPanel` associated with the run.
        - logger:
            The :class:`logging.Logger` for the run.
        - pv_panel:
            The :class:`PVPanel` associated with the system.
        - pv_system_size:
            The size of the PV system installed.
        - pv_t_system_size:
            The size of the PV-T system installed.
        - scenario:
            The :class:`Scenario` for the run.
        - solar_irradiances:
            The solar irradiances at each time step, measured in Kelvin.
        - solar_thermal_collector:
            The :class:`SolarThermalCollector` associated with the run.
        - solar_thermal_system_size:
            The size of the solar-thermal system.
        - default_tank_temperature:
            The default tank temperature to use for running for consistency.
        - disable_tqdm:
            Whether to disable the progress bar.

    Outputs:
        - The steady-state solution.

    """

    # Start of with an assumption that the tank begins at the ambient temperature.
    tank_start_temperature = ambient_temperatures[0]

    # Run an initial simulation to determine the start point for seeking a sufficient
    # solution
    solution = run_simulation(
        ambient_temperatures,
        buffer_tank,
        desalination_plant,
        htf_mass_flow_rate,
        hybrid_pv_t_panel,
        logger,
        pv_panel,
        pv_system_size,
        pv_t_system_size,
        scenario,
        solar_irradiances,
        solar_thermal_collector,
        solar_thermal_system_size,
        tank_start_temperature=tank_start_temperature,
        disable_tqdm=disable_tqdm,
    )

    tank_temperatures = solution.tank_temperatures
    try:
        convergence_distance = abs(tank_temperatures[23] - tank_start_temperature)
    except KeyError:
        logger.error(
            "Tank temperature profile did not contain overlapping hours, overlapping "
            "hours are required to find a convergent solution."
        )
        raise

    # Continue to iterate until a sufficient solution is found.
    with tqdm(
        total=round(convergence_distance, 2),
        desc="steady-state solution",
        disable=disable_tqdm,
        leave=True,
        unit="degC",
    ) as pbar:
        while convergence_distance > STEADY_STATE_TEMPERATURE_PRECISION:
            solution = run_simulation(
                ambient_temperatures,
                buffer_tank,
                desalination_plant,
                htf_mass_flow_rate,
                hybrid_pv_t_panel,
                logger,
                pv_panel,
                pv_system_size,
                pv_t_system_size,
                scenario,
                solar_irradiances,
                solar_thermal_collector,
                solar_thermal_system_size,
                disable_tqdm=disable_tqdm,
                tank_start_temperature=tank_start_temperature,  # The final tank temperature
            )

            # Update the progress bar based on the convergence of the tank temperatures
            tank_temperatures = solution.tank_temperatures
            pbar.update(
                round(
                    convergence_distance
                    - abs(tank_temperatures[23] - tank_start_temperature),
                    2,
                )
            )
            convergence_distance = abs(tank_temperatures[23] - tank_start_temperature)
            tank_start_temperature = tank_temperatures[23]

    return solution
