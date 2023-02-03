#!/usr/bin/python3.10
########################################################################################
# optimiser.py - The optimisation module for the HEATDesalination program.             #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 14/10/2022                                                             #
# License: MIT, Open-source                                                            #
# For more information, contact: benedict.winchester@gmail.com                         #
########################################################################################

"""
optimiser.py - The optimisation module for the HEATDeslination program.

The optimisation module is responsible for determining the optimum system for supplying
the needs of the desalination plant.

"""

__all__ = ("run_optimisation",)

import abc
from logging import Logger
import os
from typing import Callable, Tuple

import json
import numpy

from scipy import optimize

from .__utils__ import (
    CostableComponent,
    CostType,
    DAYS_PER_YEAR,
    FlowRateError,
    GridCostScheme,
    InputFileError,
    OptimisableComponent,
    OptimisationParameters,
    ProfileDegradationType,
    ProgrammerJudgementFault,
    Scenario,
    Solution,
)
from .heat_pump import HeatPump
from .plant import DesalinationPlant
from .simulator import determine_steady_state_simulation
from .solar import HybridPVTPanel, PVPanel, SolarPanel, SolarThermalPanel
from .storage.storage_utils import Battery, HotWaterTank

# UPPER_LIMIT:
#   Value used to throw the optimizer off of solutions that have a flow-rate error.
UPPER_LIMIT: float = 10**8


def _inverter_cost(
    component_sizes: dict[CostableComponent | None, float],
    scenario: Scenario,
    system_lifetime: int,
) -> float:
    """
    Calculate the costs associated with the inverter in the system.

    NOTE: The fractional change in the inverter costs are taken account of in this
    funciton.

    Inputs:
        - component_sizes:
            The sizes of the various components which are costable.
        - logger:
            The :class:`logging.Logger` to use for the run.
        - scenario:
            The scenario being considered.
        - system_lifetime:
            The lifetime of the system in years.

    Outputs:
        The costs associated with the inverter installed.

    """

    solar_component_sizes = {
        key: value
        for key, value in component_sizes.items()
        if isinstance(key, SolarPanel) and not isinstance(key, SolarThermalPanel)
    }

    # Determine the PV and PV-T capacities.
    if scenario.pv:
        pv_system_size = sum(
            key.pv_unit * value
            for key, value in solar_component_sizes.items()
            if isinstance(key, PVPanel)
        )
    else:
        pv_system_size = 0

    if scenario.pv_t:
        pv_t_system_size = sum(
            key.pv_module_characteristics.nominal_power * value
            for key, value in solar_component_sizes.items()
            if isinstance(key, HybridPVTPanel)
        )
    else:
        pv_t_system_size = 0

    # Determine the inverter sizing and costs associated
    inverter_cost = (
        (pv_system_size + pv_t_system_size)
        * scenario.inverter_cost
        * (system_lifetime // scenario.inverter_lifetime)
    ) * (1 + scenario.fractional_inverter_cost_change)

    return inverter_cost


def _total_component_costs(
    component_sizes: dict[CostableComponent | None, float],
    logger: Logger,
    scenario: Scenario,
) -> float:
    """
    Calculate the total cost of the costable components installed.

    NOTE: The cost-increase factors are accounted for in this function.

    Inputs:
        - component_sizes:
            The mapping between :class:`CostableComponent` instances and their installed
            capacities.
        - logger:
            The :class:`logging.Logger` to use for the run.
        - scenario:
            The scenario being considered.

    Outputs:
        The total cost of these components.

    """

    component_costs = {
        component: component.cost * abs(size)
        for component, size in component_sizes.items()
    }

    # Cycle through the component costs and multiply by the fractional change values.
    # (Apologies for the inelegant switch statement...)
    for component in component_costs:
        if isinstance(component, Battery):
            component_costs[component] *= 1 + scenario.fractional_grid_cost_change
        if isinstance(component, HotWaterTank):
            component_costs[component] *= 1 + scenario.fractional_hw_tank_cost_change
        if isinstance(component, PVPanel):
            component_costs[component] *= 1 + scenario.fractional_pv_cost_change
        if isinstance(component, HybridPVTPanel):
            component_costs[component] *= 1 + scenario.fractional_pvt_cost_change
        if isinstance(component, SolarThermalPanel):
            component_costs[component] *= 1 + scenario.fractional_st_cost_change

    logger.debug(
        "Component costs: %s",
        json.dumps(
            {str(key): value for key, value in component_costs.items()}, indent=4
        ),
    )

    return sum(component_costs.values())


def _grid_infrastructure_cost() -> float:
    """
    Calculate the costs associated with the infrastructure required for grid connections

    NOTE: Currently, this function includes no calculation for these costs but is left
    here as a hook for future development.

    Outputs:
        The costs, in USD, associated with the grid infrastructure.

    """

    return 0


def _total_grid_cost(
    logger: Logger,
    scenario: Scenario,
    solution: Solution,
    system_lifetime: int,
) -> float:
    """
    Calculate the total cost of the grid electricity used.

    NOTE: The fractional change in the grid electricity costs are accounted for within
    this function.

    NOTE: There are currently no fixed grid infrastructure costs available, though a
    hook has been provided for including this functionality later on if required.

    Inputs:
        - logger:
            The :class:`logging.Logger` to use for the run.
        - scenario:
            The scenario being considered.
        - solution:
            The steady-state solution for the simulation.
        - system_lifetime:
            The lifetime of the system in years.

    Outputs:
        The total costs associated with the grid.

    """

    # Calculate the fixed grid infrastructure costs
    fixed_grid_infrastructure_cost = _grid_infrastructure_cost()

    # Calculate the undiscounted cost of grid electricity.
    fractional_cost_change = scenario.fractional_grid_cost_change
    # total_grid_cost = (
    #     DAYS_PER_YEAR  # [days/year]
    #     * system_lifetime  # [year]
    #     * sum(solution.grid_electricity_supply_profile.values())  # [kWh/day]
    #     * scenario.grid_cost  # [$/kWh]
    # ) * (1 + fractional_cost_change)

    if (grid_supply_profile := solution.grid_electricity_supply_profile) is None:
        logger.error("No grid-supply profile provided despite grid cost required.")
        raise ProgrammerJudgementFault(
            "optimiser:_total_grid_cost", "No grid-supply profile defined."
        )

    daily_grid_consumption: float = sum(
        [entry for entry in grid_supply_profile.values() if entry is not None]
    )
    peak_grid_power = max(grid_supply_profile.values())

    if scenario.grid_cost_scheme == GridCostScheme.DUBAI_UAE:
        # Dubai, UAE-specific code - a tiered tariff applied based on monthly usage.
        # The industrial slab tariff is used with an exchange rate to USD applied of
        # 1 AED to 0.27 USD as fixed due to currency pegging.
        monthly_grid_consumption = daily_grid_consumption * (
            days_per_month := 30
        )  # [kWh/month]
        lower_tier_consumption = min(monthly_grid_consumption, 10000)
        upper_tier_consumption = max(monthly_grid_consumption - 10000, 0)
        return (
            (DAYS_PER_YEAR / days_per_month)  # [months/year]
            * system_lifetime  # [years]
            * (
                lower_tier_consumption * (0.063 * (1 + fractional_cost_change))
                + upper_tier_consumption * (0.10 * (1 + fractional_cost_change))
            )
        ) + fixed_grid_infrastructure_cost  # [USD]

    # The following schemes use lifetime power consumption, so calculate this
    grid_lifetime_electricity_consumption: float = (
        DAYS_PER_YEAR  # [days/year]
        * system_lifetime  # [years]
        * daily_grid_consumption  # [kWh/day]
    )

    if scenario.grid_cost_scheme == GridCostScheme.ABU_DHABI_UAE:
        # Abu Dhabi, UAE-specific code - a tiered tariff applied based on monthly usage.
        # The industrial fixed-rate tariff for <1MW installations is used.
        return (
            grid_lifetime_electricity_consumption
            * 0.078
            * fixed_grid_infrastructure_cost
        )  # [USD/kWh]

    if scenario.grid_cost_scheme == GridCostScheme.GRAN_CANARIA_SPAIN:
        # Gran-Canaria-specific code - a flat tariff per kWh consumed.
        # Gran Canaria grid-cost information obtained from:
        # Qiblawey Y, Alassi A, Zain ul Abideen M, Banales S.
        # Techno-economic assessment of increasing the renewable energy supply in the
        # Canary Islands: The case of Tenerife and Gran Canaria.
        # Energy Policy 2022;162:112791.
        # doi: 10.1016/j.enpol.2022.112791.
        return (
            grid_lifetime_electricity_consumption
            * 0.1537
            * fixed_grid_infrastructure_cost
        )  # [USD/kWh]

    if scenario.grid_cost_scheme in {
        GridCostScheme.TIJUANA_MEXICO,
        GridCostScheme.LA_PAZ_MEXICO,
    }:
        # Mexico grid costs operate using a tiered structure and three costs:
        #   - a monthly flat-rate cost for using a grid connection,
        #   - a specific cost which depends on the amount of electricity used,
        #   - and a cost based on the peak power consumption.
        # All these values were obtained from the Comisión Federal de Electricidad.
        if scenario.grid_cost_scheme == GridCostScheme.TIJUANA_MEXICO:
            # Tijuana-specific code - a two-tier tariff based on power consumption.
            if 0 < peak_grid_power <= 25:
                fixed_monthly_cost: float = 59.85  # [USD/month]
                power_cost: float = 0  # [USD/kW]
                specific_electricity_cost: float = 2.466  # [USD/kWh]
            elif peak_power > 25:
                fixed_monthly_cost = 598.55
                power_cost = 499.39
                specific_electricity_cost = 0.826
            else:
                fixed_monthly_cost = 0
                power_cost = 0
                specific_electricity_cost = 0
        elif scenario.grid_cost_scheme == GridCostScheme.LA_PAZ_MEXICO:
            # La-Paz-specific code - a two-tier tariff based on power consumption.
            if 0 < peak_grid_power <= 25:
                fixed_monthly_cost = 59.85
                power_cost = 0
                specific_electricity_cost = 3.817
            elif peak_power > 25:
                fixed_monthly_cost = 598.55
                power_cost = 454.36
                specific_electricity_cost = 2.907
            else:
                fixed_monthly_cost = 0
                power_cost = 0
                specific_electricity_cost = 0
        else:
            logger.error(
                "Grid cost scheme undefined: %s", scenario.grid_cost_scheme.value
            )
            raise InputFileError(
                os.path.join("inputs", "scenarios.yaml"),
                f"Grid cost scheme f{scenario.grid_cost_scheme.value} not well defined.",
            )

        # Use the fixed monthly cost along with the electricity specific costs to
        # determine the total grid cost.
        total_fixed_monthly_cost = (
            system_lifetime  # [years]
            * 12  # [months/year]
            * fixed_monthly_cost  # [USD/month]
        )
        total_power_cost = peak_grid_power * power_cost  # [kW]  # [USD/kW]
        total_specific_electricity_cost = (
            specific_electricity_cost  # [USD/kWh]
            * grid_lifetime_electricity_consumption
        )

        return (
            fixed_grid_infrastructure_cost
            + total_fixed_monthly_cost
            + total_power_cost
            + total_specific_electricity_cost
        )

    logger.error("Grid cost scheme undefined: %s", scenario.grid_cost_scheme.value)
    raise InputFileError(
        os.path.join("inputs", "scenarios.yaml"),
        f"Grid cost scheme f{scenario.grid_cost_scheme.value} not well defined.",
    )


def _total_cost(
    component_sizes: dict[CostableComponent | None, float],
    logger: Logger,
    scenario: Scenario,
    solution: Solution,
    system_lifetime: int,
) -> float:
    """
    Compute the total cost of the system which was optimisable.

    Inputs:
        - component_sizes:
            The sizes of the various components which are costable.
        - logger:
            The :class:`logging.Logger` to use for the run.
        - scenario:
            The scenario being considered.
        - solution:
            The steady-state solution for the simulation.
        - system_lifetime:
            The lifetime of the system in years.

    Outputs:
        The total cost of the system components.

    """

    # Calculate the cost of the various components which can be costed.
    total_component_cost = _total_component_costs(component_sizes, logger, scenario)

    total_grid_cost = _total_grid_cost(logger, scenario, solution, system_lifetime)

    # Add the costs of installing an inverter for dealing with solar power
    # generated
    inverter_cost = _inverter_cost(component_sizes, scenario, system_lifetime)

    # Add the costs of any consumables such as diesel fuel or grid electricity.
    total_cost = (
        total_component_cost  # Already adjusted for cost change
        + max(total_grid_cost, 0)  # Already adjusted for cost change
        + max(solution.heat_pump_cost, 0)  # Already adjusted for cost change
        + max(inverter_cost, 0)  # Already adjusted for cost change
    )  # + diesel_fuel_cost + grid_cost
    logger.info(
        "Total cost: %s, Total component cost: %s, Total grid cost %s, Heat-pump cost: "
        "%s",
        total_cost,
        total_component_cost,
        total_grid_cost,
        solution.heat_pump_cost,
    )

    return total_cost


def _total_electricity_supplied(solution: Solution, system_lifetime: int) -> float:
    """
    Calculates the total electricity supplied.

    Inputs:
        - solution:
            The solution from the steady-state calculation.
        - system_lifetime:
            The lifetime of the system in years.

    Outputs:
        The total electricity supplied.

    """

    # Sum the battery power supplied to the system.
    if (battery_output_power := solution.battery_electricity_suppy_profile) is None:
        total_battery_output_power: float = 0
    else:
        total_battery_output_power = sum(
            [entry for entry in battery_output_power.values() if entry is not None]
        )  # [kWh/day]

    # Sum the collector power supplied to the system.
    if (
        collector_output_power := solution.total_collector_electrical_output_power[
            ProfileDegradationType.DEGRADED
        ]
    ) is None:
        total_collector_output_power: float = 0
    else:
        total_collector_output_power = sum(
            [entry for entry in collector_output_power.values() if entry is not None]
        )  # [kWh/day]

    # Sum the grid power supplied to the system.
    if (grid_power := solution.grid_electricity_supply_profile) is None:
        total_grid_power: float = 0
    else:
        total_grid_power = sum(
            [entry for entry in grid_power.values() if entry is not None]
        )  # [kWh/day]

    return (
        DAYS_PER_YEAR  # [days/year]
        * system_lifetime  # [year]
        * (total_battery_output_power + total_collector_output_power + total_grid_power)
    )


class Criterion(abc.ABC):
    """
    Represents a criterion that can be optimised.

    .. attribute:: calculate_value
        A map between the crietia name and the methods for the values that need to be
        calculated.

    """

    calculate_value_map: dict[str, Callable] = {}
    name: str

    def __init_subclass__(cls, criterion_name: str):
        """
        Hook for storing the methods for the calculation of the various functions.

        Inputs:
            - name:
                The name of the criterion.

        """

        cls.calculate_value_map[criterion_name] = cls.calculate_value
        cls.name = criterion_name

    @classmethod
    @abc.abstractmethod
    def calculate_value(
        cls,
        component_sizes: dict[CostableComponent | None, float],
        logger: Logger,
        scenario: Scenario,
        solution: Solution,
        system_lifetime: int,
    ) -> float:
        """
        Calculate the value of the criterion.

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
            - logger:
                The :class:`logging.Logger` to use for the run.
            - scenario:
                The scenario being considered.
            - solution:
                The solution from running the simulation.
            - system_lifetime:
                The lifetime of the system in years.

        Outputs:
            The value being calculated.

        """


class DumpedElectricity(Criterion, criterion_name="dumped_electricity"):
    """Contains the calculation for the assessment of dumped energy."""

    @classmethod
    def calculate_value(
        cls,
        component_sizes: dict[CostableComponent | None, float],
        logger: Logger,
        scenario: Scenario,
        solution: Solution,
        system_lifetime: int,
    ) -> float:
        """
        Calculate the total amount of dumped electricity over the system lifetime.

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
            - logger:
                The :class:`logging.Logger` to use for the run.
            - scenario:
                The scenario being considered.
            - solution:
                The solution from running the simulation.
            - system_lifetime:
                The lifetime of the system in years.

        Outputs:
            The dumped electricity over the lifetime of the system.

        """

        # If no solar was dumped, return 0.
        if (dumped_solar := solution.dumped_solar) is None:
            return 0

        return sum(dumped_solar.values()) * DAYS_PER_YEAR * system_lifetime


class GridElectricityFraction(Criterion, criterion_name="grid_electricity_fraction"):
    """Contains the calculation for the fraction of electricity from grid."""

    @classmethod
    def calculate_value(
        cls,
        component_sizes: dict[CostableComponent | None, float],
        logger: Logger,
        scenario: Scenario,
        solution: Solution,
        system_lifetime: int,
    ) -> float:
        """
        Calculate the fraction of electricity that came from the grid.

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
            - logger:
                The :class:`logging.Logger` to use for the run.
            - scenario:
                The scenario being considered.
            - solution:
                The solution from running the simulation.
            - system_lifetime:
                The lifetime of the system in years.

        Outputs:
            The fraction of electricity generated that came from the grid.

        """

        # Return 0 if no grid power was used.
        if (grid_profile := solution.grid_electricity_supply_profile) is None:
            return 0

        grid_power_supplied = sum(
            [entry for entry in grid_profile.values() if entry is not None]
        )
        total_electricity_demand = sum(solution.electricity_demands.values())
        return grid_power_supplied / total_electricity_demand


class LCUE(Criterion, criterion_name="lcue"):
    """
    Contains the calculation for the levilised cost of used electricity.

    """

    @classmethod
    def calculate_value(
        cls,
        component_sizes: dict[CostableComponent | None, float],
        logger: Logger,
        scenario: Scenario,
        solution: Solution,
        system_lifetime: int,
    ) -> float:
        """
        Calculate the value of the LCUE.

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
            - logger:
                The :class:`logging.Logger` to use for the run.
            - scenario:
                The scenario being considered.
            - solution:
                The solution from running the simulation.
            - system_lifetime:
                The lifetime of the system in years.

        Outputs:
            The LCUE (Levilised Cost of Used Electricity) in USD/kWh.

        """

        return _total_cost(
            component_sizes, logger, scenario, solution, system_lifetime
        ) / _total_electricity_supplied(solution, system_lifetime)


class RenewableElectricityFraction(
    Criterion, criterion_name="renewable_electricity_fraction"
):
    """
    Contains the calculation for the renewable electricity fraction.

    The renewable electricity fraction is the fraction of the electricity demand which
    was supplied by renewable means, i.e., by solar PV or PV-T collectors or from
    storage which was renewably filled.

    """

    @classmethod
    def calculate_value(
        cls,
        component_sizes: dict[CostableComponent | None, float],
        logger: Logger,
        scenario: Scenario,
        solution: Solution,
        system_lifetime: int,
    ) -> float:
        """
        Calculate the value of the renewable electricity fraction.

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
            - logger:
                The :class:`logging.Logger` to use for the run.
            - scenario:
                The scenario being considered.
            - solution:
                The solution from running the simulation.
            - system_lifetime:
                The lifetime of the system in years.

        """


class RenewableHeatingFraction(Criterion, criterion_name="renewable_heating_fraction"):
    """
    Contains the calculation for the renewable heating fraction.

    The renewable heating fraction is the fraction of the heating demand which was
    supplied by renewable means, i.e., the fraction of the temperature demand which was
    met by the buffer tanks when compared with the ambient temperature.

    """

    @classmethod
    def calculate_value(
        cls,
        component_sizes: dict[CostableComponent | None, float],
        logger: Logger,
        scenario: Scenario,
        solution: Solution,
        system_lifetime: int,
    ) -> float:
        """
        Calculate the value of the renewable heating fraction.

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
            - logger:
                The :class:`logging.Logger` to use for the run.
            - scenario:
                The scenario being considered.
            - solution:
                The solution from running the simulation.
            - system_lifetime:
                The lifetime of the system in years.

        """

        # Determine times for which the renewable heating fraction is not None.
        non_none_renewable_heating_fraction = [
            entry
            for entry in solution.renewable_heating_fraction.values()
            if entry is not None
        ]

        return sum(non_none_renewable_heating_fraction) / len(
            non_none_renewable_heating_fraction
        )


class AuxiliaryHeatingFraction(Criterion, criterion_name="auxiliary_heating_fraction"):
    """
    Contains the calculation for the auxiliary heating fraction.

    The auxiliary heating fraction is the fraction of the heating demand which was
    supplied by auxiliary means, i.e., which was needed to heat the water from the
    buffer tank(s) up to the plant input temperatures.

    """

    @classmethod
    def calculate_value(
        cls,
        component_sizes: dict[CostableComponent | None, float],
        logger: Logger,
        scenario: Scenario,
        solution: Solution,
        system_lifetime: int,
    ) -> float:
        """
        Calculate the value of the auxiliary heating fraction.

        The fraction of the heat that needed to be supplied by the auxiliary heater will
        simply be "1 - that which was supplied renewably."

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
            - logger:
                The :class:`logging.Logger` to use for the run.
            - scenario:
                The scenario being considered.
            - solution:
                The solution from running the simulation.
            - system_lifetime:
                The lifetime of the system in years.

        """

        return 1 - super().calculate_value_map[RenewableHeatingFraction.name](  # type: ignore [no-any-return]
            component_sizes, logger, scenario, solution, system_lifetime
        )


class SolarElectricityFraction(Criterion, criterion_name="solar_electricity_fraction"):
    """Contains the calculation for the fraction of electricity from solar."""

    @classmethod
    def calculate_value(
        cls,
        component_sizes: dict[CostableComponent | None, float],
        logger: Logger,
        scenario: Scenario,
        solution: Solution,
        system_lifetime: int,
    ) -> float:
        """
        Calculate the fraction of electricity that came from solar collectors.

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
            - logger:
                The :class:`logging.Logger` to use for the run.
            - scenario:
                The scenario being considered.
            - solution:
                The solution from running the simulation.
            - system_lifetime:
                The lifetime of the system in years.

        Outputs:
            The fraction of electricity generated that came from solar.

        """

        # If no solar power was generated, return 0 as the solar fraction.
        if (solar_power_map := solution.solar_power_supplied) is None:
            return 0

        solar_power_supplied = sum(solar_power_map.values())
        total_electricity_demand = sum(solution.electricity_demands.values())
        return solar_power_supplied / total_electricity_demand


class StorageElectricityFraction(
    Criterion, criterion_name="storage_electricity_fraction"
):
    """Contains the calculation for the fraction of electricity from storage."""

    @classmethod
    def calculate_value(
        cls,
        component_sizes: dict[CostableComponent | None, float],
        logger: Logger,
        scenario: Scenario,
        solution: Solution,
        system_lifetime: int,
    ) -> float:
        """
        Calculate the fraction of electricity that came from storage.

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
            - logger:
                The :class:`logging.Logger` to use for the run.
            - scenario:
                The scenario being considered.
            - solution:
                The solution from running the simulation.
            - system_lifetime:
                The lifetime of the system in years.

        Outputs:
            The fraction of electricity generated that came from storage.

        """

        if (battery_profile := solution.battery_electricity_suppy_profile) is None:
            logger.error("No battery profile provided despite sum requested.")
            raise ProgrammerJudgementFault(
                "optimiser:StorageElectricityFraction:calculate_value",
                "No battery profile provided despite summation requested.",
            )

        try:
            storage_power_supplied: float = sum(
                [entry for entry in battery_profile.values() if entry is not None]
            )
        except AttributeError:
            print("No battery profile for scenario %s", str(scenario))
            storage_power_supplied = 0
        total_electricity_demand = sum(solution.electricity_demands.values())
        return storage_power_supplied / total_electricity_demand


class TotalCost(Criterion, criterion_name="total_cost"):
    """
    Contains the calculation for the total cost of the system.

    """

    @classmethod
    def calculate_value(
        cls,
        component_sizes: dict[CostableComponent | None, float],
        logger: Logger,
        scenario: Scenario,
        solution: Solution,
        system_lifetime: int,
    ) -> float:
        """
        Calculate the value of the total cost.

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
            - logger:
                The :class:`logging.Logger` to use for the run.
            - scenario:
                The scenario being considered.
            - solution:
                The solution from running the simulation.
            - system_lifetime:
                The lifetime of the system in years.

        Outputs:
            The total cost of the system's costable components.

        """

        return _total_cost(component_sizes, logger, scenario, solution, system_lifetime)


# class UnmetElectricity(Criterion, criterion_name="unmet_electricity_fraction"):
#     """
#     Contains the calculation for the unmet electricity demand.

#     """

#     def calculate_value(cls, solution: Solution) -> float:
#         """
#         Calculate the value of the unmet electricity.

#         """

#         return (
#             1
#             - _total_electricity_supplied(solution) / solution.total_electricity_demand
#         )


# class UnmetHeating(Criterion, criterion_name="unmet_heating_fraction"):
#     """
#     Contains the calculation for the unmet heating demand.

#     """

#     def calculate_value(cls, solution: Solution) -> float:
#         """
#         Calculate the value of the unmet heating.

#         """

#         return (
#             1
#             - _total_heat_supplied(solution) / solution.total_heating_demand
#         )


def _simulate_and_calculate_criterion(
    parameter_vector: numpy.ndarray,
    ambient_temperatures: dict[int, float],
    battery: Battery,
    battery_capacity: int | None,
    buffer_tank: HotWaterTank,
    buffer_tank_capacity: float | None,
    desalination_plant: DesalinationPlant,
    heat_pump: HeatPump,
    htf_mass_flow_rate: float | None,
    hybrid_pv_t_panel: HybridPVTPanel | None,
    logger: Logger,
    optimisation_criterion: str,
    pv_panel: PVPanel | None,
    pv_panel_system_size: int | None,
    pv_t_system_size: int | None,
    scenario: Scenario,
    start_hour: int | None,
    solar_irradiances: dict[int, float],
    solar_thermal_collector: SolarThermalPanel | None,
    solar_thermal_system_size: int | None,
    system_lifetime: int,
    wind_speeds: dict[int, float],
    disable_tqdm: bool = False,
) -> float:
    """
    Runs the simulation function and computes the requested criterion.

    The parameter vector, in order, contains, if specified:
    - battery_capacity:
        The capacity in kWh of the batteries.
    - buffer_tank_capacity:
        The capacity in kg of the buffer tank.
    - htf_mass_flow_rate:
        The HTF mass flow rate.
    - pv_panel_system_size:
        The PV-panel capacity.
    - pv_t_system_size:
        The PV-T system size.
    - start_hour:
        The start hour for the plant.
    - solar_thermal_system_size:
        The solar-thermal system size.

    Inputs:
        - ambient_temperatures:
            The ambient temperature at each time step, measured in Kelvin.
        - battery:
            The :class:`Battery` associated with the system.
        - battery_capacity:
            The capacity of the batteries installed if this should not be optimised or
            `None` if the capacity of the batteries should be optimised.
        - buffer_tank:
            The :class:`HotWaterTank` associated with the system.
        - buffer_tank_capacity:
            The capacity of the buffer tank if this should not be optimised or `None` if
            the capacity of the tank should be optimised.
        - desalination_plant:
            The :class:`DesalinationPlant` for which the systme is being simulated.
        - heat_pump:
            The :class:`HeatPump` to use for the run.
        - htf_mass_flow_rate:
            The mass flow rate of the HTF through the collectors if this should not be
            optimised, or `None` if it should be optimised.
        - hybrid_pv_t_panel:
            The :class:`HybridPVTPanel` associated with the run.
        - logger:
            The :class:`logging.Logger` for the run.
        - optimisation_criterion:
            The name of the optimisation criterion to use.
        - pv_panel:
            The :class:`PVPanel` associated with the system.
        - pv_panel_system_size:
            The size of the PV system installed if this should not be optimised, or
            `None` if it should be optimised.
        - pv_t_system_size:
            The size of the PV-T system installed if this should not be optimised, or
            `None` if it should be optimised.
        - scenario:
            The :class:`Scenario` for the run.
        - solar_irradiances:
            The solar irradiances at each time step, measured in Kelvin.
        - solar_thermal_collector:
            The :class:`SolarThermalCollector` associated with the run.
        - solar_thermal_system_size:
            The size of the solar-thermal system if this should not be optimised, or
            `None` if it should be optimised.
        - system_lifetime:
            The lifetime of the system measured in years.
        - wind_speeds:
            The wind speeds at each time step, measured in meters per second.
        - disable_tqdm:
            Whether to disable the progress bar.

    """

    # Convert the parameter vector to a list
    parameter_list: list[float] = parameter_vector.tolist()

    # Setup input parameters from the vector.
    _battery_capacity: float = (
        battery_capacity if battery_capacity is not None else parameter_list.pop(0)
    )

    # Buffer-tank capacities
    existing_buffer_tank_capacity: float = buffer_tank.capacity
    buffer_tank.capacity = (
        buffer_tank_capacity
        if buffer_tank_capacity is not None
        else parameter_list.pop(0)
    )
    # Sanitise the units on the area of the buffer tank
    buffer_tank.capacity = buffer_tank.capacity * 1000
    _buffer_tank_capacity = buffer_tank.capacity

    # Increase the area of the tank by a factor of the volume increase accordingly.
    buffer_tank.area *= (buffer_tank.capacity / existing_buffer_tank_capacity) ** (
        2 / 3
    )

    # Collector parameters
    _htf_mass_flow_rate: float = (
        htf_mass_flow_rate if htf_mass_flow_rate is not None else parameter_list.pop(0)
    )
    _pv_panel_system_size: float = (
        pv_panel_system_size
        if pv_panel_system_size is not None
        else parameter_list.pop(0)
    )
    _pv_t_system_size: float = (
        pv_t_system_size if pv_t_system_size is not None else parameter_list.pop(0)
    )
    _solar_thermal_system_size: float = (
        solar_thermal_system_size
        if solar_thermal_system_size is not None
        else parameter_list.pop(0)
    )

    # Setup plant parameters
    _start_hour: float = start_hour if start_hour is not None else parameter_list.pop(0)
    desalination_plant.start_hour = start_hour
    desalination_plant.reset_operating_hours()

    # Determine the steady-state solution.
    try:
        steady_state_solution: Solution = determine_steady_state_simulation(
            ambient_temperatures,
            battery,
            _battery_capacity,
            buffer_tank,
            desalination_plant,
            heat_pump,
            _htf_mass_flow_rate,
            hybrid_pv_t_panel,
            logger,
            pv_panel,
            _pv_panel_system_size,
            _pv_t_system_size,
            scenario,
            solar_irradiances,
            solar_thermal_collector,
            _solar_thermal_system_size,
            system_lifetime,
            wind_speeds,
            disable_tqdm=disable_tqdm,
        )
    except FlowRateError:
        logger.info(
            "Flow-rate error encountered, doubling component sizes to throw optimizer.",
        )
        return UPPER_LIMIT

    # Assemble the component sizes mapping.
    component_sizes: dict[CostableComponent, int | float] = {
        battery: _battery_capacity * (1 + steady_state_solution.battery_replacements),  # type: ignore [dict-item,operator]
        buffer_tank: _buffer_tank_capacity,  # type: ignore [dict-item]
        hybrid_pv_t_panel: _pv_t_system_size,  # type: ignore [dict-item]
        pv_panel: _pv_panel_system_size,  # type: ignore [dict-item]
        solar_thermal_collector: _solar_thermal_system_size,  # type: ignore [dict-item]
    }

    # Return the value of the criterion.
    return (  # type: ignore [no-any-return]
        Criterion.calculate_value_map[optimisation_criterion](
            component_sizes, logger, scenario, steady_state_solution, system_lifetime
        )
        / 10**6
    ) ** 3


def _callback_function(current_vector: numpy.ndarray, *args) -> None:
    """
    Callback function to execute after each itteration.

    Inputs:
        - current_vector:
            The current vector.

    """

    print("Current vector: ({})".format([f"{entry:.1f}" for entry in current_vector]))


def _constraint_function(
    current_vector: numpy.ndarray,
    hybrid_pv_t_panel: HybridPVTPanel,
    optimisation_parameters: OptimisationParameters,
    solar_thermal_collector: SolarThermalPanel,
) -> float:
    """
    Represents constraints on the solution.

    The mass flow-rate needs to sit within certain bounds. If this is out of bounds,
    then 0 will be returned stating that the system is invalid. Otherwise, 1 will be
    returned.

    """

    # Sanity check on bounds.
    if (
        battery_index := optimisation_parameters.optimisable_component_to_index.get(
            OptimisableComponent.BATTERY_CAPACITY, None
        )
    ) is not None:
        if current_vector[battery_index] < 0:
            return -10
    if (
        pv_index := optimisation_parameters.optimisable_component_to_index.get(
            OptimisableComponent.PV, None
        )
    ) is not None:
        if current_vector[pv_index] < 0:
            return -20

    # Determine the mass flow rate if fixed or variable.
    if optimisation_parameters.fixed_mass_flow_rate_value is not None:
        system_mass_flow_rate: float = (
            optimisation_parameters.fixed_mass_flow_rate_value
        )
    else:
        system_mass_flow_rate = current_vector[
            optimisation_parameters.optimisable_component_to_index[
                OptimisableComponent.MASS_FLOW_RATE
            ]
        ]

    # If there are PV-T collectors present, ensure that the mass flow rate is valid.
    if (
        pv_t_index := optimisation_parameters.optimisable_component_to_index.get(
            OptimisableComponent.PV_T, None
        )
    ) is not None:
        collector_flow_rate = system_mass_flow_rate / (
            pv_t_system_size := current_vector[pv_t_index]
        )
        if (
            collector_flow_rate < hybrid_pv_t_panel.min_mass_flow_rate
            or collector_flow_rate > hybrid_pv_t_panel.max_mass_flow_rate
        ):
            return -30

    # If there are solar-thermal collectors present, ensure that the mass flow rate is
    # valid.
    if (
        solar_thermal_index := optimisation_parameters.optimisable_component_to_index.get(
            OptimisableComponent.SOLAR_THERMAL, None
        )
    ) is not None:
        collector_flow_rate = system_mass_flow_rate / (
            solar_thermal_system_size := current_vector[solar_thermal_index]
        )
        if (
            collector_flow_rate < solar_thermal_collector.min_mass_flow_rate
            or collector_flow_rate > solar_thermal_collector.max_mass_flow_rate
        ):
            return -40

    # Should also return 0 if both the ST and PV-T sizes are zero.
    if pv_t_system_size is not None and solar_thermal_system_size is not None:
        if (pv_t_system_size == 0) and (solar_thermal_system_size == 0):
            return -50

    return 0


def run_optimisation(
    ambient_temperatures: dict[int, float],
    battery: Battery,
    buffer_tank: HotWaterTank,
    desalination_plant: DesalinationPlant,
    heat_pump: HeatPump,
    hybrid_pv_t_panel: HybridPVTPanel | None,
    logger: Logger,
    optimisation_parameters: OptimisationParameters,
    pv_panel: PVPanel | None,
    scenario: Scenario,
    solar_irradiances: dict[int, float],
    solar_thermal_collector: SolarThermalPanel | None,
    system_lifetime: int,
    wind_speeds: dict[int, float],
    *,
    disable_tqdm: bool = True,
) -> Tuple[dict[str, float], list[float]]:
    """
    Determine the optimum system conditions.

    Inputs:
        - ambient_temperatures:
            The ambient temperature at each time step, measured in Kelvin.
        - battery:
            The :class:`Battery` associated with the system.
        - buffer_tank:
            The :class:`HotWaterTank` associated with the system.
        - desalination_plant:
            The :class:`DesalinationPlant` for which the systme is being simulated.
        - heat_pump:
            The :class:`HeatPump` to use for the run.
        - hybrid_pv_t_panel:
            The :class:`HybridPVTPanel` associated with the run.
        - logger:
            The :class:`logging.Logger` for the run.
        - optimisation_parameters:
            Parameters which govern the optimisation.
        - pv_panel:
            The :class:`PVPanel` associated with the system.
        - scenario:
            The :class:`Scenario` for the run.
        - solar_irradiances:
            The solar irradiances at each time step, measured in Kelvin.
        - solar_thermal_collector:
            The :class:`SolarThermalCollector` associated with the run.
        - system_lifetime:
            The lifetime of the system, measured in years.
        - wind_speeds:
            The wind speeds at each time step, measured in meters per second.
        - disable_tqdm:
            Whether to disable the progress bar.

    Outputs:
        - A mapping containing information about the values of all the optimisation
          criteria defined, as well as the costs of the various parts of the overall
          system;
        - The optimised system.

    """

    # Determine the alrogithm based on whether constraints are needed.
    if optimisation_parameters.constraints is not None:
        # algorithm = "COBYLA"
        algorithm = "SLSQP"
        constraints = {
            "type": "ineq",
            # "type": "eq",
            "fun": _constraint_function,
            "args": (
                hybrid_pv_t_panel,
                optimisation_parameters,
                solar_thermal_collector,
            ),
        }
    else:
        algorithm = "Nelder-Mead"
        # algorithm = "Powell"
        # algorithm = "CG"
        # algorithm = "BFGS"
        # algorithm = "L-BFGS-B"
        # algorithm = "TNC"
        # algorithm = "COBYLA"
        # algorithm = "SLSQP"
        # algorithm = "trust-constr"
        constraints = None
        # algorithm = None

    # Construct an initial guess vector and additional arguments.
    try:
        (
            initial_guess_vector,
            bounds,
        ) = optimisation_parameters.get_initial_guess_vector_and_bounds()
    except KeyError as error:
        logger.error("Missing optimisation parameters: %s", str(error))
        raise

    # Determine the additional arguments vector required.
    additional_arguments = (
        ambient_temperatures,
        battery,
        optimisation_parameters.fixed_battery_capacity_value,
        buffer_tank,
        optimisation_parameters.fixed_buffer_tank_capacity_value,
        desalination_plant,
        heat_pump,
        optimisation_parameters.fixed_mass_flow_rate_value,
        hybrid_pv_t_panel,
        logger,
        optimisation_parameters.target_criterion,
        pv_panel,
        optimisation_parameters.fixed_pv_value,
        optimisation_parameters.fixed_pv_t_value,
        scenario,
        optimisation_parameters.fixed_start_hour_value,
        solar_irradiances,
        solar_thermal_collector,
        optimisation_parameters.fixed_st_value,
        system_lifetime,
        wind_speeds,
        disable_tqdm,
    )

    class Bounds:
        def __init__(self, bounds: list[Tuple[float | None, float | None]]) -> None:
            self.bounds = bounds

        def __call__(self, **kwargs) -> bool:
            """Determines whether to accept (True) or reject (False)"""

            new_point = kwargs.get("x_new")
            for index, entry in enumerate(new_point):  # type: ignore [arg-type]
                if self.bounds[index][0] is not None and entry < self.bounds[index][0]:
                    return False
                if self.bounds[index][1] is not None and entry > self.bounds[index][1]:
                    return False
            return True

    bounds_instance = Bounds(bounds)

    # Optimise the system.
    # return optimize.basinhopping(
    #     _simulate_and_calculate_criterion,
    #     initial_guess_vector,
    #     accept_test=bounds_instance,
    #     callback=_callback_function,
    #     minimizer_kwargs={"args": additional_arguments}
    # )

    optimisation_result = optimize.minimize(
        _simulate_and_calculate_criterion,
        initial_guess_vector,
        additional_arguments,
        method=algorithm,
        bounds=bounds,
        callback=_callback_function,
        # callback=_callback_function if not disable_tqdm else None,
        constraints=constraints if constraints is not None else None,
        options={
            "disp": False,
            "fatol": 10 ** -(6),
            "ftol": 2.22 * 10 ** (-12),
            "gtol": 10 ** (-6),
            "maxiter": 10000,
            "maxfev": 10000,
            "return_all": True,
            "xatol": 10 ** -(6),
        },
        # options={"disp": not disable_tqdm, "maxiter": 10000},
    )

    # Calculate the optimisation criterion value and return this also.
    parameter_list: list[float] = list(optimisation_result.x)

    # Setup input parameters from the vector.
    battery_capacity: float = (
        optimisation_parameters.fixed_battery_capacity_value
        if optimisation_parameters.fixed_battery_capacity_value is not None
        else parameter_list.pop(0)
    )
    buffer_tank_capacity: float = (
        optimisation_parameters.fixed_buffer_tank_capacity_value
        if optimisation_parameters.fixed_buffer_tank_capacity_value is not None
        else parameter_list.pop(0)
    )
    buffer_tank.capacity = buffer_tank_capacity * 1000

    # Collector parameters
    htf_mass_flow_rate: float = (
        optimisation_parameters.fixed_mass_flow_rate_value
        if optimisation_parameters.fixed_mass_flow_rate_value is not None
        else parameter_list.pop(0)
    )
    pv_system_size: float = (
        optimisation_parameters.fixed_pv_value
        if optimisation_parameters.fixed_pv_value is not None
        else parameter_list.pop(0)
    )
    pv_t_system_size: float = (
        optimisation_parameters.fixed_pv_t_value
        if optimisation_parameters.fixed_pv_t_value is not None
        else parameter_list.pop(0)
    )
    solar_thermal_system_size: float = (
        optimisation_parameters.fixed_st_value
        if optimisation_parameters.fixed_st_value is not None
        else parameter_list.pop(0)
    )

    # Carry out a simulation.
    solution = determine_steady_state_simulation(
        ambient_temperatures,
        battery,
        battery_capacity,
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
        system_lifetime,
        wind_speeds,
        disable_tqdm=disable_tqdm,
    )

    # Assemble the component sizes mapping.
    component_sizes: dict[CostableComponent | None, float] = {
        battery: battery_capacity * (1 + solution.battery_replacements),
        buffer_tank: buffer_tank_capacity,
        hybrid_pv_t_panel: pv_t_system_size,
        pv_panel: pv_system_size,
        solar_thermal_collector: solar_thermal_system_size,
    }

    # Compute various criteria values.
    criterion_map = {
        criterion.name: criterion.calculate_value(
            component_sizes, logger, scenario, solution, system_lifetime
        )
        for criterion in [
            AuxiliaryHeatingFraction,
            DumpedElectricity,
            GridElectricityFraction,
            SolarElectricityFraction,
            StorageElectricityFraction,
            TotalCost,
        ]
    }

    # Compute the costs of the various parts of the system and append this.
    criterion_map.update(
        {
            CostType.COMPONENTS.value: _total_component_costs(
                component_sizes, logger, scenario
            ),
            CostType.GRID.value: _total_grid_cost(
                logger, scenario, solution, system_lifetime
            ),
            CostType.HEAT_PUMP.value: float(max(solution.heat_pump_cost, 0))
            * (1 + scenario.fractional_heat_pump_cost_change),
            CostType.INVERTERS.value: _inverter_cost(
                component_sizes, scenario, system_lifetime
            ),
        }
    )

    # Return the value of the criterion along with the result from the simulation.
    return criterion_map, list(optimisation_result.x)
