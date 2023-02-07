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
from typing import Defaultdict, Tuple

import math

from tqdm import tqdm

from .__utils__ import (
    DAYS_PER_YEAR,
    ProfileDegradationType,
    ProgrammerJudgementFault,
    Scenario,
    Solution,
    ZERO_CELCIUS_OFFSET,
)
from .heat_pump import calculate_heat_pump_electricity_consumption_and_cost, HeatPump
from .matrix import solve_matrix
from .plant import DesalinationPlant
from .solar import HybridPVTPanel, PVPanel, SolarThermalPanel, electric_output
from .storage.storage_utils import Battery, HotWaterTank


__all__ = ("run_simulation",)


# DEFAULT_HOT_WATER_RETURN_TEMPERATURE:
#   The default return temperature, in degrees Celsius, for hot water leaving the
# desalination plant.
DEFAULT_HOT_WATER_RETURN_TEMPERATURE: int = 40

# DEGRADATION_PRECISION:
#   The precision required when solving recursively for the degradation in terms of
# decimal places.
DEGRADATION_PRECISION: int = 5

# Temperature precision:
#   The precision required when solving for a steady-state solution.
STEADY_STATE_TEMPERATURE_PRECISION: float = 0.001


def _calculate_collector_degradation(
    scenario: Scenario, solution: Solution, system_lifetime: int
) -> None:
    """
    Calculate the average degradation rate for collectors.

    Collectors degrade fractionally based on time. Hence, the fraction of the
    degradation which has occurred as an average over their lifetime will require
    integration of the function
        (1 - d)^n
    where d is the degradation rate and n the number of years.

    Inegrating this yields:
        ((1 - d)^N - 1) / log(1 - d)

    which is evaluated in this function.

    Inputs:
        - scenario:
            The scenario for the run.
        - solution:
            The solution from the system.
        - system_lifetime:
            The lifetime of the system.

    """

    # Helper function for carrying out repeated degradation flow.
    def _degrade(
        profile: dict[int, float | None], rate: float
    ) -> dict[int, float | None]:
        return {
            key: (value * rate) if value is not None else None
            for key, value in profile.items()
        }

    # Compute the average degradation rate.
    average_degradation_rate = (
        (1 - scenario.pv_degradation_rate) ** system_lifetime - 1
    ) / (math.log(1 - scenario.pv_degradation_rate) * system_lifetime)

    # Degrade the PV and PV-T electrical profiles.
    if scenario.pv:
        solution.pv_electrical_efficiencies[
            ProfileDegradationType.DEGRADED.value
        ] = _degrade(
            solution.pv_electrical_efficiencies[
                ProfileDegradationType.UNDEGRADED.value
            ],
            average_degradation_rate,
        )
        solution.pv_electrical_output_power[
            ProfileDegradationType.DEGRADED.value
        ] = _degrade(
            solution.pv_electrical_output_power[
                ProfileDegradationType.UNDEGRADED.value
            ],
            average_degradation_rate,
        )
        solution.pv_system_electrical_output_power[
            ProfileDegradationType.DEGRADED.value
        ] = _degrade(
            solution.pv_system_electrical_output_power[
                ProfileDegradationType.UNDEGRADED.value
            ],
            average_degradation_rate,
        )

    if scenario.pv_t:
        solution.pv_t_electrical_efficiencies[
            ProfileDegradationType.DEGRADED.value
        ] = _degrade(
            solution.pv_t_electrical_efficiencies[
                ProfileDegradationType.UNDEGRADED.value
            ],
            average_degradation_rate,
        )
        solution.pv_t_electrical_output_power[
            ProfileDegradationType.DEGRADED.value
        ] = _degrade(
            solution.pv_t_electrical_output_power[
                ProfileDegradationType.UNDEGRADED.value
            ],
            average_degradation_rate,
        )
        solution.pv_t_system_electrical_output_power[
            ProfileDegradationType.DEGRADED.value
        ] = _degrade(
            solution.pv_t_system_electrical_output_power[
                ProfileDegradationType.UNDEGRADED.value
            ],
            average_degradation_rate,
        )


def _storage_profile_iteration_step(
    battery: Battery,
    battery_system_size: int,
    maximum_charge_degradation: float,
    solution: Solution,
    total_collector_generation_profile: dict[int, float],
    *,
    initial_storage_profile_value: float,
) -> Tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """
    Carry out an iteration step for determining the storage profiles.

    Inputs:
        - battery:
            The battery being modelled.
        - battery_capacity:
            The capacity of the batteries installed.
        - maximum_charge_degradation:
            The factor of degradation to apply to the maximum charge
        - solution:
            The profiles being modelled.
        - total_collector_generation_profile:
            The total power generation profile from the solar average-degraded
            solar collectors.
        - initial_storage_profile_value:
            An initial value for the storage profile to use.

    Outputs:
        - power_to_storage_map:
            The power supplied to the batteries at each hour in kWh.
        - storage_power_supplied:
            The storage power supplied in kWh at each hour.
        - storage_profile:
            The profile of power stored in the batteries at each hour.

    """

    # Setup a map to keep track of the profiles.
    power_to_storage_map: dict[int, float] = {}
    storage_power_supplied_map: dict[int, float] = {}
    storage_profile: Defaultdict[int, float] = defaultdict(
        lambda: initial_storage_profile_value
    )

    for hour in range(24):
        net_storage_flow: float = (
            total_collector_generation_profile[hour]
            - solution.electricity_demands[hour]
        )

        if net_storage_flow < 0:
            # Batteries can only discharge based on:
            #   - the total power stored in the batteries above the minimum level,
            #   - the total load being requested (excess discharge would be wasted),
            #   - limited by the c-rate.
            storage_power_supplied = min(
                max(
                    (
                        storage_profile[hour - 1] * (1 - battery.leakage)
                        - battery_system_size
                        * battery.capacity
                        * battery.minimum_charge
                    ),
                    0,
                ),
                abs(net_storage_flow),
                battery_system_size * battery.capacity * battery.c_rate_discharging,
            )
            power_to_storage_map[hour] = 0
            storage_power_supplied_map[hour] = storage_power_supplied
            storage_profile[hour] = (
                storage_profile[hour - 1]
                - storage_power_supplied / battery.conversion_out
            )
        if net_storage_flow > 0:
            # Batteries can only charge:
            #   - limited by the amount of power being inputted,
            #   - limited by the c-rate,
            #   - limited by the electricity that they can hold.
            electricity_to_batteries = min(
                net_storage_flow * battery.conversion_in,
                battery_system_size * battery.capacity * battery.c_rate_charging,
                battery_system_size
                * battery.capacity
                * maximum_charge_degradation
                * battery.maximum_charge
                - storage_profile[hour - 1] * (1 - battery.leakage),
            )
            power_to_storage_map[hour] = (
                electricity_to_batteries / battery.conversion_in
            )
            storage_power_supplied_map[hour] = 0
            storage_profile[hour] = (
                storage_profile[hour - 1] * (1 - battery.leakage)
                + electricity_to_batteries
            )
        if net_storage_flow == 0:
            # Batteries leak provide there is power to leak out.
            electricity_to_batteries = 0
            power_to_storage_map[hour] = 0
            storage_power_supplied_map[hour] = 0
            storage_profile[hour] = max(
                storage_profile[hour - 1] * (1 - battery.leakage)
                + electricity_to_batteries,
                battery.capacity * battery.minimum_charge,
            )

    return power_to_storage_map, storage_power_supplied_map, storage_profile


def _maximum_charge_degradation_factor(
    lifetime_degradation: float,
    undegraded_maximum_charge: float,
    undegraded_minimum_charge: float,
) -> float:
    """
    Calculates the factor by which to degrade the maximum charge.

    The total charge-storing capacity of the batteries is degraded as time goes
    on such that:
        max* - min = (1 - eta) (max - min),
    where - max* is the new maximum charge capacity of the batteries,
          - max is the original maximum charge capacity of the batteries,
          - min the original minimum ---------------- "" ---------------,
          - and eta the degradation factor for the charge carying capacity such that,
            when there is no degradation, eta is zero and, when there is full
            degradation, eta is 1.

    Rearranging this yields:
        max* / max = (1 - eta) + eta * min / max
    for the factor by which the maximum charge should be degraded based on a lifetime
    loss by a factor of eta.

    Inputs:
        - lifetime_degradation:
            The lifetime degradation to the charge-holding capacity of the batteries.
        - undegraded_maximum_charge:
            The maximum charge factor for undegraded batteries.
        - undegraded_minimum_charge:
            The minimum charge factor for undegraded batteries.

    Outputs:
        The degradation factor for the maximum charge of the batteries.

    """

    return (1 - lifetime_degradation) + lifetime_degradation * (
        undegraded_minimum_charge / undegraded_maximum_charge
    )


def _storage_solver(
    battery: Battery,
    battery_system_size: int,
    maximum_charge_degradation: float,
    solution: Solution,
    total_collector_generation_profile: dict[int, float],
    *,
    initial_storage_profile_value: float,
) -> Tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """
    Solve the storage profile until a consistent start value is found.

    Inputs:
        - battery:
            The battery being modelled.
        - battery_capacity:
            The capacity of the batteries installed.
        - maximum_charge_degradation:
            The factor of degradation to apply to the maximum charge
        - solution:
            The profiles being modelled.
        - total_collector_generation_profile:
            The total power generation profile from the solar average-degraded
            solar collectors.
        - initial_storage_profile_value:
            An initial value for the storage profile to use.

    Outputs:
        - power_to_storage_map:
            The power supplied to the batteries at each hour in kWh.
        - storage_power_supplied:
            The storage power supplied in kWh at each hour.
        - storage_profile:
            The profile of power stored in the batteries at each hour.

    """

    # Call the storage profile iteration step.
    (
        power_to_storage_map,
        storage_power_supplied_map,
        storage_profile,
    ) = _storage_profile_iteration_step(
        battery,
        battery_system_size,
        maximum_charge_degradation,
        solution,
        total_collector_generation_profile,
        initial_storage_profile_value=initial_storage_profile_value,
    )

    # Set up variables to keep track of whether a solution has been found.
    eleventh_hour_storage: float = storage_profile[23]
    solution_found: bool = initial_storage_profile_value == eleventh_hour_storage

    # Loop until a consistent solution is found.
    while not solution_found:
        # Call the storage profile iteration step.
        (
            power_to_storage_map,
            storage_power_supplied_map,
            storage_profile,
        ) = _storage_profile_iteration_step(
            battery,
            battery_system_size,
            maximum_charge_degradation,
            solution,
            total_collector_generation_profile,
            initial_storage_profile_value=(
                initial_storage_profile_value := eleventh_hour_storage
            ),
        )
        eleventh_hour_storage = storage_profile[23]

        solution_found = round(eleventh_hour_storage, DEGRADATION_PRECISION) == round(
            initial_storage_profile_value, DEGRADATION_PRECISION
        )

    return power_to_storage_map, storage_power_supplied_map, storage_profile


def _recursive_degraded_storage_solver(
    battery: Battery,
    battery_system_size: int,
    solution: Solution,
    system_lifetime: int,
    total_collector_generation_profile: dict[int, float],
    *,
    input_lifetime_degradation: float,
) -> Tuple[float, dict[int, float], dict[int, float], dict[int, float]]:
    """
    Recursively solve the degradation level of the storage profile until consistent.

    Inputs:
        - battery:
            The battery being modelled.
        - battery_capacity:
            The capacity of the batteries installed.
        - solution:
            The profiles being modelled.
        - system_lifetime:
            The lifetime in years of the system.
        - total_collector_generation_profile:
            The total power generation profile from the solar average-degraded
            solar collectors.
        - input_lifetime_degradation:
            An initial value for the lifetime degradation of the battery system.

    Outputs:
        - lifetime_degradation:
            The lifetime degradation of the storage system expressed as a fraction
            between 0 (no degradation) and 1 (full degradation).
        - power_to_storage_map:
            The power supplied to the batteries at each hour in kWh.
        - storage_power_supplied:
            The storage power supplied in kWh at each hour.
        - storage_profile:
            The profile of power stored in the batteries at each hour.

    """

    # Refactor the lifetime degradation in case it is greater than 1.
    def _refactor_lifetime_degradation(x: float) -> float:
        """
        Refactor the lifetime degradation if it is less than 1.

        """

        try:
            return 0.5 * (x // 1) / x + 0.5 * (x % 1) ** 2 / x
        except ZeroDivisionError:
            return 0

    # Begin with the supplied degradation factor.
    maximum_degradation_factor = _maximum_charge_degradation_factor(
        _refactor_lifetime_degradation(input_lifetime_degradation) / 2,
        battery.maximum_charge,
        battery.minimum_charge,
    )
    power_to_storage_map, storage_power_supplied_map, storage_profile = _storage_solver(
        battery,
        battery_system_size,
        maximum_degradation_factor,
        solution,
        total_collector_generation_profile,
        initial_storage_profile_value=battery_system_size * battery.minimum_charge,
    )

    # Compute the total power through the storage system over the lifetime of the
    # batteries and hence the system degradation/
    total_storage_power_supplied = (
        system_lifetime * DAYS_PER_YEAR * sum(storage_power_supplied_map.values())
    )
    lifetime_degradation = (
        total_storage_power_supplied
        * battery.lifetime_loss
        / (
            battery_system_size
            * battery.cycle_lifetime
            * (battery.maximum_charge - battery.minimum_charge)
            * battery.capacity
        )
    )

    # If the lifetime degradation matches, return.
    if round(lifetime_degradation, DEGRADATION_PRECISION) == round(
        input_lifetime_degradation, DEGRADATION_PRECISION
    ):
        return (
            lifetime_degradation,
            power_to_storage_map,
            storage_power_supplied_map,
            storage_profile,
        )

    # Degrade the storage capacity by this and re-run.
    return _recursive_degraded_storage_solver(
        battery,
        battery_system_size,
        solution,
        system_lifetime,
        total_collector_generation_profile,
        input_lifetime_degradation=lifetime_degradation,
    )


def _calculate_storage_profile(
    battery: Battery,
    battery_system_size: int | None,
    solution: Solution,
    system_lifetime: int,
) -> Tuple[
    float, dict[int, float], dict[int, float], dict[int, float], dict[int, float]
]:
    """
    Calculate the storage profile for the batteries.

    Inputs:
        - battery:
            The battery being modelled.
        - battery_capacity:
            The capacity of the batteries installed.
        - solution:
            The profiles being modelled.
        - system_lifetime:
            The lifetime, in years, of the system being considered.

    Outputs:
        - battery_lifetime_degradation:
            The lifetime degradation of the storage system expressed as a fraction
            between 0 (no degradation) and 1 (full degradation).
        - battery_power_input:
            The power supplied to the batteries at each hour in kWh.
        - battery_power_supplied:
            The storage power supplied in kWh at each hour.
        - battery_storage_profile:
            The profile of power stored in the batteries at each hour.
        - solar_power_supplied_map:
            The power supplied by the solar collectors at each hour.

    """

    # Determine the total solar generation.
    total_collector_generation_profile = (
        solution.total_collector_electrical_output_power[
            ProfileDegradationType.DEGRADED.value
        ]
    )

    # Compute the solar profile.
    solar_power_supplied_map: dict[int, float] = {
        hour: min(
            solution.electricity_demands[hour],
            total_collector_generation_profile[hour]
            if total_collector_generation_profile[hour] is not None
            else 0,
        )
        for hour in solution.electricity_demands
    }

    # If there is no storage, return purely the solar power map.
    if battery_system_size == 0:
        return 0, None, None, None, solar_power_supplied_map

    return _recursive_degraded_storage_solver(
        battery,
        battery_system_size,
        solution,
        system_lifetime,
        total_collector_generation_profile,
        input_lifetime_degradation=0,
    ) + (solar_power_supplied_map,)


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


def _tank_replacement_temperature(
    ambient_temperature: float, hot_water_return_temperature: float | None
) -> float:
    """
    Return the temperature of water which is replacing that taken from the tank.

    The number used is that for the final effect of the plant being considered.

    Inputs:
        - ambient_temperature:
            The ambient temperature, measured in degrees Kelvin.
        - hot_water_return_temperature:
            The return temperature of hot water from the plant.

    Outputs:
        The temperature of water replacing that taken from the hot-water tank in Kelvin.

    """

    # If the plant return temperature isn't specified, return the ambient temperature.
    if hot_water_return_temperature is None:
        return ambient_temperature

    # Otherwise, use the plant-specific value.
    return ZERO_CELCIUS_OFFSET + hot_water_return_temperature


def run_simulation(
    ambient_temperatures: dict[int, float],
    buffer_tank: HotWaterTank,
    desalination_plant: DesalinationPlant,
    heat_pump: HeatPump,
    htf_mass_flow_rate: float,
    hybrid_pv_t_panel: HybridPVTPanel | None,
    logger: Logger,
    pv_panel: PVPanel | None,
    pv_system_size: int | None,
    pv_t_system_size: int | None,
    scenario: Scenario,
    solar_irradiances: dict[int, float],
    solar_thermal_collector: SolarThermalPanel | None,
    solar_thermal_system_size: int | None,
    wind_speeds: dict[int, float],
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
        - heat_pump:
            The :class:`HeatPump` to use for the run.
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
        - wind_speeds:
            The wind speeds at each time step, measured in meters per second.
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

    if (
        scenario.solar_thermal
        and solar_thermal_system_size is not None
        and solar_thermal_system_size > 0
    ):
        solar_thermal_mass_flow_rate: float | None = _collector_mass_flow_rate(
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
    auxiliary_heating_demands: dict[int, float] = {}
    auxiliary_heating_electricity_demands: dict[int, float] = {}
    base_electricity_demands: dict[int, float] = {}
    collector_input_temperatures: dict[int, float] = {}
    collector_system_output_temperatures: dict[int, float] = {}
    electricity_demands: Defaultdict[int, float] = defaultdict(float)
    hot_water_demand_temperatures: dict[int, float | None] = {}
    hot_water_demand_volumes: dict[int, float | None] = {}
    max_heat_pump_cost: float = 0
    pv_t_electrical_efficiencies: dict[int, float | None] = {}
    pv_t_electrical_output_power: dict[int, float | None] = {}
    pv_t_htf_output_temperatures: dict[int, float | None] = {}
    pv_t_reduced_temperatures: dict[int, float | None] = {}
    pv_t_thermal_efficiencies: dict[int, float | None] = {}
    solar_thermal_htf_output_temperatures: dict[int, float | None] = {}
    solar_thermal_reduced_temperatures: dict[int, float | None] = {}
    solar_thermal_thermal_efficiencies: dict[int, float | None] = {}
    tank_temperatures: Defaultdict[int, float] = defaultdict(
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
            _tank_replacement_temperature(
                ambient_temperatures[hour],
                desalination_plant.outputs(hour).hot_water_return_temperature,
            ),
        )

        # Determine the electricity demands of the plant including any auxiliary
        # heating.
        if desalination_plant.operating(hour):
            if (
                hot_water_volume := desalination_plant.requirements(
                    hour
                ).hot_water_volume
            ) is None or (
                hot_water_temperature := desalination_plant.requirements(
                    hour
                ).hot_water_temperature
            ) is None:
                logger.error(
                    "Desalination plant requirements not defined for hour %s.", hour
                )
                raise ProgrammerJudgementFault(
                    "simulator::run_simulation",
                    "Desalination plant requirements not defined for plant "
                    f"{desalination_plant.name} for hour {hour} despite plant "
                    "operating.",
                )

            auxiliary_heating_demand = max(
                (
                    hot_water_volume  # [kg/s]
                    * buffer_tank.heat_capacity  # [J/kg*K]
                    * (hot_water_temperature - tank_temperature)  # [K]
                    / 1000  # [W/kW]
                ),
                0,
            )  # [kW]

            # Calculate the power consumption.

            (
                heat_pump_cost,
                heat_pump_power_consumpion,
            ) = calculate_heat_pump_electricity_consumption_and_cost(
                hot_water_temperature,
                ambient_temperatures[hour],
                auxiliary_heating_demand,
                heat_pump,
            )

            auxiliary_heating_electricity_demand: float = max(
                heat_pump_power_consumpion,
                0,
            )  # [kW]
            electricity_demand: float = (
                desalination_plant.requirements(hour).electricity  # [kW]
                + auxiliary_heating_electricity_demand
            )
            max_heat_pump_cost = max(heat_pump_cost, max_heat_pump_cost) * (
                1 + scenario.fractional_heat_pump_cost_change
            )
        else:
            auxiliary_heating_demand = 0
            auxiliary_heating_electricity_demand = 0
            electricity_demand = desalination_plant.requirements(hour).electricity
            hot_water_temperature = None
            hot_water_volume = None

        # Save these outputs in mappings.
        auxiliary_heating_demands[hour] = auxiliary_heating_demand
        auxiliary_heating_electricity_demands[
            hour
        ] = auxiliary_heating_electricity_demand
        base_electricity_demands[hour] = desalination_plant.requirements(  # [kW]
            hour
        ).electricity
        collector_input_temperatures[hour] = collector_input_temperature
        collector_system_output_temperatures[hour] = collector_system_output_temperature
        electricity_demands[hour] = electricity_demand
        hot_water_demand_temperatures[hour] = hot_water_temperature
        hot_water_demand_volumes[hour] = hot_water_demand_volume
        pv_t_electrical_efficiencies[hour] = pv_t_electrical_efficiency
        if hybrid_pv_t_panel is not None:
            pv_t_electrical_output_power[hour] = (
                (
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
                if scenario.pv_t
                else None
            )
        else:
            pv_t_electrical_output_power[hour] = None
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
        pv_performance_characteristics: dict[
            int, Tuple[float, float | None, float, float | None]
        ] = {
            hour: pv_panel.calculate_performance(
                ambient_temperatures[hour],
                logger,
                solar_irradiances[hour],
                wind_speed=wind_speeds[hour],
            )
            for hour in range(len(ambient_temperatures))
        }
        pv_electrical_efficiencies: dict[int, float] | None = {
            hour: entry[0] for hour, entry in pv_performance_characteristics.items()
        }
        pv_average_temperatures: dict[int, float] | None = {
            hour: entry[2] for hour, entry in pv_performance_characteristics.items()
        }
        pv_electrical_output_power: dict[int, float] | None = {
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
        pv_system_electrical_output_power: dict[int, float] | None = {
            hour: (value * pv_system_size if value is not None else 0)
            for hour, value in pv_electrical_output_power.items()
        }
    else:
        pv_electrical_efficiencies = None
        pv_electrical_output_power = None
        pv_average_temperatures = None
        pv_system_electrical_output_power = None

    # Compute the output power from the various collectors.
    logger.info("Hourly simulation complete, compute the output power.")
    pv_t_system_electrical_output_power: dict[int, float] = {
        hour: (value * pv_t_system_size if value is not None else 0)
        for hour, value in pv_t_electrical_output_power.items()
    }

    logger.info("Simulation complete, returning outputs.")
    return Solution(
        ambient_temperatures,
        auxiliary_heating_demands,
        auxiliary_heating_electricity_demands,
        base_electricity_demands,
        collector_input_temperatures,
        collector_system_output_temperatures,
        electricity_demands,
        max_heat_pump_cost,
        hot_water_demand_temperatures,
        hot_water_demand_volumes,
        pv_average_temperatures if scenario.pv else None,
        {ProfileDegradationType.UNDEGRADED.value: pv_electrical_efficiencies}
        if scenario.pv
        else None,
        {ProfileDegradationType.UNDEGRADED.value: pv_electrical_output_power}
        if scenario.pv
        else None,
        {ProfileDegradationType.UNDEGRADED.value: pv_system_electrical_output_power}
        if scenario.pv
        else None,
        {ProfileDegradationType.UNDEGRADED.value: pv_t_electrical_efficiencies}
        if scenario.pv_t
        else None,
        {ProfileDegradationType.UNDEGRADED.value: pv_t_electrical_output_power}
        if scenario.pv_t
        else None,
        pv_t_htf_output_temperatures if scenario.pv_t else None,
        pv_t_reduced_temperatures if scenario.pv_t else None,
        {ProfileDegradationType.UNDEGRADED.value: pv_t_system_electrical_output_power}
        if scenario.pv_t
        else None,
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
    ambient_temperatures: dict[int, float],
    battery: Battery | None,
    battery_capacity: float | int | None,
    buffer_tank: HotWaterTank,
    desalination_plant: DesalinationPlant,
    heat_pump: HeatPump,
    htf_mass_flow_rate: float,
    hybrid_pv_t_panel: HybridPVTPanel | None,
    logger: Logger,
    pv_panel: PVPanel | None,
    pv_system_size: float | int | None,
    pv_t_system_size: float | int | None,
    scenario: Scenario,
    solar_irradiances: dict[int, float],
    solar_thermal_collector: SolarThermalPanel | None,
    solar_thermal_system_size: float | int | None,
    system_lifetime: int,
    wind_speeds: dict[int, float],
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
        - battery:
            The battery installed or `None` if not battery is installed.
        - battery_capacity:
            The capacity in kWh of electrical storage installed, or `None` if none is
            installed.
        - buffer_tank:
            The :class:`HotWaterTank` associated with the system.
        - desalination_plant:
            The :class:`DesalinationPlant` for which the systme is being simulated.
        - heat_pump:
            The :class:`HeatPump` to use for the run.
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
        - system_lifetime:
            The lifetime of the system in years.
        - wind_speeds:
            The wind speeds at each time step, measured in meters per second.
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
        heat_pump,
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
        wind_speeds,
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
                heat_pump,
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
                wind_speeds,
                disable_tqdm=disable_tqdm,
                tank_start_temperature=tank_start_temperature,
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

    # Determine the degraded profiles.
    _calculate_collector_degradation(scenario, solution, system_lifetime)

    # Determine the storage profile.
    (
        battery_lifetime_degradation,
        battery_power_input_profile,
        battery_power_supplied_profile,
        battery_storage_profile,
        solar_power_supplied,
    ) = _calculate_storage_profile(battery, battery_capacity, solution, system_lifetime)

    if battery_storage_profile is not None:
        battery_storage_profile.pop(-1)

    # Determine the grid profile.
    grid_profile = {
        hour: solution.electricity_demands[hour]
        - solar_power_supplied[hour]
        - (
            battery_power_supplied_profile[hour]
            if battery_power_supplied_profile is not None
            else 0
        )
        for hour in solar_power_supplied
    }

    # Determine the dumped solar energy.
    dumped_solar = {
        hour: solution.total_collector_electrical_output_power[
            ProfileDegradationType.DEGRADED.value
        ][hour]
        - solar_power_supplied[hour]
        - (
            battery_power_input_profile[hour]
            if battery_power_supplied_profile is not None
            else 0
        )
        for hour in solar_power_supplied
    }

    # Save these to the output tuple.
    solution = solution._replace(
        battery_lifetime_degradation=battery_lifetime_degradation
    )
    solution = solution._replace(
        battery_replacements=int(battery_lifetime_degradation // 1)
    )
    solution = solution._replace(
        battery_electricity_suppy_profile=battery_power_supplied_profile
    )
    solution = solution._replace(dumped_solar=dumped_solar)
    solution = solution._replace(battery_storage_profile=battery_storage_profile)
    solution = solution._replace(
        battery_power_input_profile=battery_power_input_profile
    )
    solution = solution._replace(grid_electricity_supply_profile=grid_profile)
    solution = solution._replace(solar_power_supplied=solar_power_supplied)

    return solution
