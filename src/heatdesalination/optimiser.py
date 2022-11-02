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
from typing import Any, Dict, List

import numpy

from scipy import optimize

from .__utils__ import (
    CostableComponent,
    FlowRateError,
    OptimisationMode,
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
UPPER_LIMIT: float = 10**10


def _total_cost(component_sizes: Dict[CostableComponent | None, float]) -> float:
    """
    Compute the total cost of the system which was optimisable.

    Inputs:
        - component_sizes:
            The sizes of the various components which are costable.

    Outputs:
        The total cost of the system components.

    """

    # Calculate the cost of the various components which can be costed.
    component_costs = {
        component: component.cost * size
        for component, size in component_sizes.items()
        if isinstance(component, CostableComponent)
    }
    total_component_cost = sum(component_costs.values())

    # Add the costs of any consumables such as diesel fuel or grid electricity.
    total_cost = total_component_cost  # + diesel_fuel_cost + grid_cost

    return total_cost


def _total_electricity_supplied(solution: Solution) -> float:
    """
    Calculates the total electricity supplied.

    Inputs:
        - solution:
            The solution from the steady-state calculation.

    Outputs:
        The total electricity supplied.

    """


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
        cls, component_sizes: Dict[CostableComponent | None, float], solution: Solution
    ) -> float:
        """
        Calculate the value of the criterion.

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
            - solution:
                The solution from running the simulation.

        Outputs:
            The value being calculated.

        """


class LCUE(Criterion, criterion_name="lcue"):
    """
    Contains the calculation for the levilised cost of used electricity.

    """

    @classmethod
    def calculate_value(
        cls, component_sizes: Dict[CostableComponent | None, float], solution: Solution
    ) -> float:
        """
        Calculate the value of the LCUE.

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
            - solution:
                The solution from running the simulation.

        Outputs:
            The LCUE (Levilised Cost of Used Electricity) in USD/kWh.

        """

        return _total_cost(component_sizes) / _total_electricity_supplied(solution)


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
        cls, component_sizes: Dict[CostableComponent | None, float], solution: Solution
    ) -> float:
        """
        Calculate the value of the renewable electricity fraction.

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
            - solution:
                The solution from running the simulation.

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
        cls, component_sizes: Dict[CostableComponent | None, float], solution: Solution
    ) -> float:
        """
        Calculate the value of the renewable heating fraction.

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
            - solution:
                The solution from running the simulation.

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
        cls, component_sizes: Dict[CostableComponent | None, float], solution: Solution
    ) -> float:
        """
        Calculate the value of the auxiliary heating fraction.

        The fraction of the heat that needed to be supplied by the auxiliary heater will
        simply be "1 - that which was supplied renewably."

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
            - solution:
                The solution from running the simulation.

        """

        return 1 - super().calculate_value_map[RenewableHeatingFraction.name](
            component_sizes, solution
        )


class TotalCost(Criterion, criterion_name="total_cost"):
    """
    Contains the calculation for the total cost of the system.

    """

    @classmethod
    def calculate_value(
        cls, component_sizes: Dict[CostableComponent | None, float], solution: Solution
    ) -> float:
        """
        Calculate the value of the total cost.

        Inputs:
            - component_sizes:
                The sizes of the various components which are costable.
            - solution:
                The solution from running the simulation.

        Outputs:
            The total cost of the system's costable components.

        """

        return _total_cost(component_sizes)


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
    storage_size: int | None,
    disable_tqdm: bool = False,
) -> float:
    """
    Runs the simulation function and computes the requested criterion.

    The parameter vector, in order, contains, if specified:
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
    - storage_size:
        The battery capacity.

    Inputs:
        - ambient_temperatures:
            The ambient temperature at each time step, measured in Kelvin.
        - battery:
            The :class:`Battery` associated with the system.
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
        - disable_tqdm:
            Whether to disable the progress bar.

    """

    # Convert the parameter vector to a list
    parameter_list: List[float] = parameter_vector.tolist()

    # Setup input parameters from the vector.
    buffer_tank_capacity: float = (
        buffer_tank_capacity
        if buffer_tank_capacity is not None
        else parameter_list.pop(0)
    )
    buffer_tank.capacity = buffer_tank_capacity

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

    # Storage parameters
    storage_size: float = (
        storage_size if storage_size is not None else parameter_list.pop(0)
    )

    # Determine the steady-state solution.
    try:
        steady_state_solution = determine_steady_state_simulation(
            ambient_temperatures,
            buffer_tank,
            desalination_plant,
            htf_mass_flow_rate,
            hybrid_pv_t_panel,
            logger,
            pv_panel,
            pv_t_system_size,
            scenario,
            solar_irradiances,
            solar_thermal_collector,
            solar_thermal_system_size,
            disable_tqdm=disable_tqdm,
        )
    except FlowRateError:
        logger.info(
            "Flow-rate error encountered, using upper bound of %s to throw optimizer.",
            UPPER_LIMIT,
        )
        return UPPER_LIMIT

    # Assemble the component sizes mapping.
    component_sizes: Dict[CostableComponent, float] = {
        buffer_tank: buffer_tank_capacity,
        hybrid_pv_t_panel: pv_t_system_size,
        pv_panel: pv_panel_system_size,
        solar_thermal_collector: solar_thermal_system_size,
        battery: storage_size,
    }

    # Return the value of the criterion.
    return Criterion.calculate_value_map[optimisation_criterion](
        component_sizes, steady_state_solution
    )


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
    *,
    disable_tqdm: bool = True,
) -> Any:
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
        - disable_tqdm:
            Whether to disable the progress bar.

    Outputs:
        The optimised system.

    """

    # Determine the alrogithm based on whether constraints are needed.
    if (constraints := optimisation_parameters.constraints) is not None and len(
        optimisation_parameters.constraints
    ) > 0:
        algorithm = "COBYLA"
    else:
        algorithm = "Nelder-Mead"

    # Construct an initial guess vector and additional arguments.
    (
        initial_guess_vector,
        bounds,
    ) = optimisation_parameters.get_initial_guess_vector_and_bounds()

    # Determine the additional arguments vector required.
    additional_arguments = (
        ambient_temperatures,
        battery,
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
        optimisation_parameters.fixed_storage_value,
        disable_tqdm,
    )

    # Optimise the system.
    return optimize.minimize(
        _simulate_and_calculate_criterion,
        initial_guess_vector,
        additional_arguments,
        method=algorithm,
        bounds=bounds,
        constraints=constraints if constraints is not None else None,
        options={"disp": True},
    )
