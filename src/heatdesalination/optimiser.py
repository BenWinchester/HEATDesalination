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
from typing import Dict, List, Tuple

import numpy

from scipy import optimize

from .__utils__ import (
    DAYS_PER_YEAR,
    CostableComponent,
    FlowRateError,
    OptimisableComponent,
    OptimisationParameters,
    Scenario,
    Solution,
)
from .plant import DesalinationPlant
from .simulator import determine_steady_state_simulation
from .solar import HybridPVTPanel, PVPanel, SolarThermalPanel
from .storage.storage_utils import Battery, HotWaterTank

# UPPER_LIMIT:
#   Value used to throw the optimizer off of solutions that have a flow-rate error.
UPPER_LIMIT: float = 10**8


def _total_cost(
    component_sizes: Dict[CostableComponent | None, float],
    scenario: Scenario,
    solution: Solution,
    system_lifetime: int,
) -> float:
    """
    Compute the total cost of the system which was optimisable.

    Inputs:
        - component_sizes:
            The sizes of the various components which are costable.
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
    component_costs = {
        component: component.cost * abs(size)
        for component, size in component_sizes.items()
    }
    total_component_cost = sum(component_costs.values())

    # # Calculate the undiscounted cost of grid electricity.
    # total_grid_cost = (
    #     DAYS_PER_YEAR  # [days/year]
    #     * system_lifetime  # [year]
    #     * sum(solution.grid_electricity_supply_profile.values())  # [kWh/day]
    #     * scenario.grid_cost  # [$/kWh]
    # )

    # UAE-specific code
    discount_rate = scenario.discount_rate
    monthly_grid_consumption = (
        sum(solution.grid_electricity_supply_profile.values()) * 30
    )  # [kWh/month]
    lower_tier_consumption = min(monthly_grid_consumption, 10000)
    upper_tier_consumption = max(monthly_grid_consumption - 10000, 0)
    total_grid_cost = (
        (DAYS_PER_YEAR / 30)
        * system_lifetime
        * (
            lower_tier_consumption * (0.23 * ((1 - discount_rate) ** system_lifetime))
            + upper_tier_consumption * (0.38 * ((1 - discount_rate) ** system_lifetime))
        )
    )

    # Add the costs of any consumables such as diesel fuel or grid electricity.
    total_cost = total_component_cost + abs(
        total_grid_cost
    )  # + diesel_fuel_cost + grid_cost

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

    return (
        DAYS_PER_YEAR  # [days/year]
        * system_lifetime  # [year]
        * (
            sum(solution.battery_electricity_suppy_profile.values())  # [kWh/day]
            + sum(solution.grid_electricity_supply_profile.values())  # [kWh/day]
            + sum(
                solution.total_collector_electrical_output_power.values()
            )  # [kWh/day]
        )
    )


class Criterion(abc.ABC):
    """
    Represents a criterion that can be optimised.

    .. attribute:: calculate_value
        A map between the crietia name and the methods for the values that need to be
        calculated.

    """

    calculate_value_map = {}
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
        component_sizes: Dict[CostableComponent | None, float],
        scenario: Scenario,
        solution: Solution,
        system_lifetime: int,
    ) -> float:
        """
        Calculate the value of the criterion.

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
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
        component_sizes: Dict[CostableComponent | None, float],
        scenario: Scenario,
        solution: Solution,
        system_lifetime: int,
    ) -> float:
        """
        Calculate the total amount of dumped electricity over the system lifetime.

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
            - scenario:
                The scenario being considered.
            - solution:
                The solution from running the simulation.
            - system_lifetime:
                The lifetime of the system in years.

        Outputs:
            The dumped electricity over the lifetime of the system.

        """

        return sum(solution.dumped_solar.values()) * DAYS_PER_YEAR * system_lifetime


class GridElectricityFraction(Criterion, criterion_name="grid_electricity_fraction"):
    """Contains the calculation for the fraction of electricity from grid."""

    @classmethod
    def calculate_value(
        cls,
        component_sizes: Dict[CostableComponent | None, float],
        scenario: Scenario,
        solution: Solution,
        system_lifetime: int,
    ) -> float:
        """
        Calculate the fraction of electricity that came from the grid.

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
            - scenario:
                The scenario being considered.
            - solution:
                The solution from running the simulation.
            - system_lifetime:
                The lifetime of the system in years.

        Outputs:
            The fraction of electricity generated that came from the grid.

        """

        grid_power_supplied = sum(solution.grid_electricity_supply_profile.values())
        total_electricity_demand = sum(solution.electricity_demands.values())
        return grid_power_supplied / total_electricity_demand


class LCUE(Criterion, criterion_name="lcue"):
    """
    Contains the calculation for the levilised cost of used electricity.

    """

    @classmethod
    def calculate_value(
        cls,
        component_sizes: Dict[CostableComponent | None, float],
        scenario: Scenario,
        solution: Solution,
        system_lifetime: int,
    ) -> float:
        """
        Calculate the value of the LCUE.

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
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
            component_sizes, scenario, solution, system_lifetime
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
        component_sizes: Dict[CostableComponent | None, float],
        scenario: Scenario,
        solution: Solution,
        system_lifetime: int,
    ) -> float:
        """
        Calculate the value of the renewable electricity fraction.

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
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
        component_sizes: Dict[CostableComponent | None, float],
        scenario: Scenario,
        solution: Solution,
        system_lifetime: int,
    ) -> float:
        """
        Calculate the value of the renewable heating fraction.

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
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
        component_sizes: Dict[CostableComponent | None, float],
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
            - scenario:
                The scenario being considered.
            - solution:
                The solution from running the simulation.
            - system_lifetime:
                The lifetime of the system in years.

        """

        return 1 - super().calculate_value_map[RenewableHeatingFraction.name](
            component_sizes, scenario, solution, system_lifetime
        )


class SolarElectricityFraction(Criterion, criterion_name="solar_electricity_fraction"):
    """Contains the calculation for the fraction of electricity from solar."""

    @classmethod
    def calculate_value(
        cls,
        component_sizes: Dict[CostableComponent | None, float],
        scenario: Scenario,
        solution: Solution,
        system_lifetime: int,
    ) -> float:
        """
        Calculate the fraction of electricity that came from solar collectors.

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
            - scenario:
                The scenario being considered.
            - solution:
                The solution from running the simulation.
            - system_lifetime:
                The lifetime of the system in years.

        Outputs:
            The fraction of electricity generated that came from solar.

        """

        solar_power_supplied = sum(solution.solar_power_supplied.values())
        total_electricity_demand = sum(solution.electricity_demands.values())
        return solar_power_supplied / total_electricity_demand


class StorageElectricityFraction(
    Criterion, criterion_name="storage_electricity_fraction"
):
    """Contains the calculation for the fraction of electricity from storage."""

    @classmethod
    def calculate_value(
        cls,
        component_sizes: Dict[CostableComponent | None, float],
        scenario: Scenario,
        solution: Solution,
        system_lifetime: int,
    ) -> float:
        """
        Calculate the fraction of electricity that came from storage.

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
            - scenario:
                The scenario being considered.
            - solution:
                The solution from running the simulation.
            - system_lifetime:
                The lifetime of the system in years.

        Outputs:
            The fraction of electricity generated that came from storage.

        """

        try:
            storage_power_supplied: float = sum(
                solution.battery_electricity_suppy_profile.values()
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
        component_sizes: Dict[CostableComponent | None, float],
        scenario: Scenario,
        solution: Solution,
        system_lifetime: int,
    ) -> float:
        """
        Calculate the value of the total cost.

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
            - scenario:
                The scenario being considered.
            - solution:
                The solution from running the simulation.
            - system_lifetime:
                The lifetime of the system in years.

        Outputs:
            The total cost of the system's costable components.

        """

        return _total_cost(component_sizes, scenario, solution, system_lifetime)


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
    ambient_temperatures: Dict[int, float],
    battery: Battery,
    battery_capacity: int | None,
    buffer_tank: HotWaterTank,
    buffer_tank_capacity: float | None,
    desalination_plant: DesalinationPlant,
    htf_mass_flow_rate: float | None,
    hybrid_pv_t_panel: HybridPVTPanel | None,
    logger: Logger,
    optimisation_criterion: str,
    pv_panel: PVPanel | None,
    pv_panel_system_size: int | None,
    pv_t_system_size: int | None,
    scenario: Scenario,
    start_hour: int | None,
    solar_irradiances: Dict[int, float],
    solar_thermal_collector: SolarThermalPanel | None,
    solar_thermal_system_size: int | None,
    system_lifetime: int,
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
        - disable_tqdm:
            Whether to disable the progress bar.

    """

    # Convert the parameter vector to a list
    parameter_list: List[float] = parameter_vector.tolist()

    # Setup input parameters from the vector.
    battery_capacity: float = (
        battery_capacity if battery_capacity is not None else parameter_list.pop(0)
    )
    existing_buffer_tank_capacity: float = buffer_tank.capacity
    buffer_tank.capacity = (
        buffer_tank_capacity
        if buffer_tank_capacity is not None
        else parameter_list.pop(0)
    )
    # Sanitise the units on the area of the buffer tank
    buffer_tank.capacity = buffer_tank_capacity * 1000

    # Increase the area of the tank by a factor of the volume increase accordingly.
    buffer_tank.area *= (buffer_tank.capacity / existing_buffer_tank_capacity) ** (
        2 / 3
    )

    # Collector parameters
    htf_mass_flow_rate: float = (
        htf_mass_flow_rate if htf_mass_flow_rate is not None else parameter_list.pop(0)
    )
    pv_panel_system_size: float = (
        pv_panel_system_size
        if pv_panel_system_size is not None
        else parameter_list.pop(0)
    )
    pv_t_system_size: float = (
        pv_t_system_size if pv_t_system_size is not None else parameter_list.pop(0)
    )
    solar_thermal_system_size: float = (
        solar_thermal_system_size
        if solar_thermal_system_size is not None
        else parameter_list.pop(0)
    )

    # Setup plant parameters
    start_hour: float = start_hour if start_hour is not None else parameter_list.pop(0)
    desalination_plant.start_hour = start_hour
    desalination_plant.reset_operating_hours()

    # Determine the steady-state solution.
    try:
        steady_state_solution: Solution = determine_steady_state_simulation(
            ambient_temperatures,
            battery,
            battery_capacity,
            buffer_tank,
            desalination_plant,
            htf_mass_flow_rate,
            hybrid_pv_t_panel,
            logger,
            pv_panel,
            pv_panel_system_size,
            pv_t_system_size,
            scenario,
            solar_irradiances,
            solar_thermal_collector,
            solar_thermal_system_size,
            system_lifetime,
            disable_tqdm=disable_tqdm,
        )
    except FlowRateError:
        logger.info(
            "Flow-rate error encountered, doubling component sizes to throw optimizer.",
        )
        return UPPER_LIMIT

    # Assemble the component sizes mapping.
    component_sizes: Dict[CostableComponent, float] = {
        battery: battery_capacity * (1 + steady_state_solution.battery_replacements),
        buffer_tank: buffer_tank_capacity,
        hybrid_pv_t_panel: pv_t_system_size,
        pv_panel: pv_panel_system_size,
        solar_thermal_collector: solar_thermal_system_size,
    }

    # Return the value of the criterion.
    return (
        Criterion.calculate_value_map[optimisation_criterion](
            component_sizes, scenario, steady_state_solution, system_lifetime
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

    def _constraint_value():
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

    return _constraint_value()


def run_optimisation(
    ambient_temperatures: Dict[int, float],
    battery: Battery,
    buffer_tank: HotWaterTank,
    desalination_plant: DesalinationPlant,
    hybrid_pv_t_panel: HybridPVTPanel,
    logger: Logger,
    optimisation_parameters: OptimisationParameters,
    pv_panel: PVPanel,
    scenario: Scenario,
    solar_irradiances: Dict[int, float],
    solar_thermal_collector: SolarThermalPanel,
    system_lifetime: int,
    *,
    disable_tqdm: bool = True,
) -> Tuple[Dict[str, float], List[float]]:
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
        - disable_tqdm:
            Whether to disable the progress bar.

    Outputs:
        The optimised system.

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
        disable_tqdm,
    )

    class Bounds:
        def __init__(self, bounds: List[Tuple[int | None]]) -> None:
            self.bounds = bounds

        def __call__(self, **kwargs) -> bool:
            """Determines whether to accept (True) or reject (False)"""

            new_point = kwargs.get("x_new")
            for index, entry in enumerate(new_point):
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
            "disp": True,
            "fatol": 10 ** -(21),
            "ftol": 2.22 * 10 ** (-12),
            "gtol": 10 ** (-21),
            "maxiter": 10000,
            "maxfev": 10000,
            "return_all": True,
            "xatol": 10 ** -(6),
        },
        # options={"disp": not disable_tqdm, "maxiter": 10000},
    )

    # Calculate the optimisation criterion value and return this also.
    parameter_list: List[float] = list(optimisation_result.x)

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
        disable_tqdm=disable_tqdm,
    )

    # Assemble the component sizes mapping.
    component_sizes: Dict[CostableComponent, float] = {
        battery: battery_capacity * (1 + solution.battery_replacements),
        buffer_tank: buffer_tank_capacity,
        hybrid_pv_t_panel: pv_t_system_size,
        pv_panel: pv_system_size,
        solar_thermal_collector: solar_thermal_system_size,
    }

    # Compute various criteria values.
    criterion_map = {
        criterion.name: criterion.calculate_value(
            component_sizes, scenario, solution, system_lifetime
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

    # Return the value of the criterion along with the result from the simulation.
    return criterion_map, list(optimisation_result.x)
