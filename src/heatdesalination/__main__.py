#!/usr/bin/python3.10
########################################################################################
# __main__.py - The main module for HEAT-Desalination simulation and optimisation.     #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 14/10/2022                                                             #
# License: MIT, Open-source                                                            #
# For more information, contact: benedict.winchester@gmail.com                         #
########################################################################################

"""
__main__.py - Main module for the deslination optimisation and simulation program.

This module ties together the functionality and code required in order to simulate
and optimise the desalination systems.

"""

import os
import sys

from typing import Any, Tuple

import json

from tqdm import tqdm

from .__utils__ import (
    CLI_TO_PROFILE_TYPE,
    DEFAULT_SIMULATION_OUTPUT_FILE,
    ProfileType,
    Solution,
    get_logger,
)
from .argparser import MissingParametersError, parse_args, validate_args
from .fileparser import parse_input_files
from .optimiser import (
    AuxiliaryHeatingFraction,
    Criterion,
    DumpedElectricity,
    GridElectricityFraction,
    run_optimisation,
    SolarElectricityFraction,
    StorageElectricityFraction,
    TotalCost,
)
from .simulator import determine_steady_state_simulation

__all__ = ("main",)

# __version__:
#   The version of the software being used.
__version__: str = "v1.0.0a1"

# ANALYSIS_REQUESTS:
#   Names of criteria to evaluate.
ANALYSIS_REQUESTS = {
    AuxiliaryHeatingFraction.name,
    DumpedElectricity.name,
    GridElectricityFraction.name,
    # LCUE.name,
    SolarElectricityFraction.name,
    StorageElectricityFraction.name,
    TotalCost.name,
}

# SIMULATION_OUTPUTS_DIRECTORY:
#   The outputs dierctory for simulations.
SIMULATION_OUTPUTS_DIRECTORY: str = "simulation_outputs"

# OPTIMISATION_OUTPUTS_DIRECTORY:
#   The outputs directory for optimisations.
OPTIMISATION_OUTPUTS_DIRECTORY: str = "optimisation_outputs"

# OPTIMISATION_PARAMETERS_KEYWORD:
#   Keyword for saving optimisation parameters.
OPTIMISATION_PARAMETERS: str = "parameters"


def save_simulation(
    output: str,
    profile_type: str,
    simulation_outputs: Solution,
    solar_irradiance: dict[int, float],
) -> None:
    """
    Save the outputs from the simulation run.

    Inputs:
        - simulation_outputs:
            Outputs from the simulation.
        - output:
            The name of the output file to use.

    """

    # Assemble the CSV datafile structure
    output_data = simulation_outputs.as_dataframe
    output_data["Solar irradiance / W/m^2"] = solar_irradiance

    # Write to the output file.
    os.makedirs(SIMULATION_OUTPUTS_DIRECTORY, exist_ok=True)
    with open(
        f"{os.path.join(SIMULATION_OUTPUTS_DIRECTORY, output)}_{profile_type}" ".csv",
        "w",
        encoding="UTF-8",
    ) as output_file:
        output_data.to_csv(output_file)


def save_optimisation(
    optimisation_outputs: list[Any],
    output: str,
) -> None:
    """
    Save the outputs from the optimisation run.

    Inputs:
        - optimisation_results:
            The results from the optimisation.
        - output:
            The name of the output file to use.

    """

    # Create the outputs directory if it doesn't exist already.
    os.makedirs(OPTIMISATION_OUTPUTS_DIRECTORY, exist_ok=True)

    # Write to the output file.
    with open(
        f"{os.path.join(OPTIMISATION_OUTPUTS_DIRECTORY, output)}.json",
        "w",
        encoding="UTF-8",
    ) as output_file:
        json.dump(optimisation_outputs, output_file)


def main(
    location: str,
    profile_types: list[ProfileType],
    scenario_name: str,
    system_lifetime: int,
    battery_capacity: float | None = None,
    buffer_tank_capacity: float | None = None,
    mass_flow_rate: float | None = None,
    optimisation: bool = False,
    output: str = DEFAULT_SIMULATION_OUTPUT_FILE,
    pv_t_system_size: int | None = None,
    pv_system_size: int | None = None,
    simulation: bool = False,
    solar_thermal_system_size: int | None = None,
    start_hour: int | None = None,
    *,
    disable_tqdm: bool = False,
    hpc: bool = False,
    save_outputs: bool = True,
    verbose: bool = False,
) -> Any:
    """
    Main module responsible for the flow of the HEATDesalination program.

    Inputs:
        - location:
            The name of the location to be modelled.
        - profile_types:
            The `list` of valid :class:`ProfileType` instances to consider.
        - scenario_name:
            The name of the scenario to use.
        - system_lifetime:
            The lifetime of the system being considered.
        - battery_capacity:
            The battery capacity if running a simulation or `None` if not.
        - buffer_tank_capacity:
            The buffer tank capacity if running a simulation or `None` if the default
            value should be used.
        - mass_flow_rate:
            The mass flow rate if running a simulation or `None` if not.
        - output:
            The name of the output file to use or `None` if the default should be used.
        - optimisation:
            Whether an optimisation is being run (True) or not (False).
        - pv_t_system_size:
            The PV-T system size, measured in number of collectors, if running a
            simulation or `None` if not.
        - pv_system_size:
            The PV system size, measured in number of collectors, if running a
            simulation or `None` if not.
        - simulation:
            Whether a simulation is being run (True) or not (False).
        - solar_thermal_system_size:
            The solar-thermal system size, measured in number of collectors if running
            a simulation or `None` if not.
        - start_hour:
            The start hour for the operation of the plant if running a simulation or
            `None` if not.
        - disable_tqdm:
            Whether to disable tqdm or not.
        - hpc:
            Whether running on the HPC (True) or not (False).
        - save_outputs:
            Whether to save the outputs from the simulation/optimisation (True) or
            return them (False).
        - verbose:
            Whether to use verbose logging (True) or not (False).

    Outputs:
        - The result of the optimisations or simulations.

    """

    logger = get_logger(f"{location}_heat_desalination", hpc, verbose)
    logger.info("Heat-desalination module instantiated. Main method.")

    # Parse the various input files.
    (
        ambient_temperatures,
        battery,
        buffer_tank,
        desalination_plant,
        heat_pump,
        hybrid_pv_t_panel,
        optimisations,
        pv_panel,
        scenario,
        solar_irradiances,
        solar_thermal_collector,
        wind_speeds,
    ) = parse_input_files(location, logger, scenario_name, start_hour)
    logger.info("Input files successfully parsed.")

    if simulation:
        # Raise exceptions if the arguments are invalid.
        missing_parameters: list[str] = []
        if scenario.battery and battery_capacity is None:
            logger.error(
                "Must specify battery capacity if batteries included in scenario."
            )
            missing_parameters.append("Storage system size")
        if scenario.pv_t and pv_t_system_size is None:
            logger.error(
                "Must specify PV-T system size if PV-T collectors included in scenario."
            )
            missing_parameters.append("PV-T system size")
        if scenario.solar_thermal and solar_thermal_system_size is None:
            logger.error(
                "Must specify solar-thermal system size if solar-thermal collectors "
                "included in scenario."
            )
            missing_parameters.append("Solar-thermal system size")
        if mass_flow_rate is None:
            logger.error("Must specify HTF mass flow rate if running a simulation.")
            missing_parameters.append("HTF mass flow rate")

        if len(missing_parameters) > 0:
            raise MissingParametersError(", ".join(missing_parameters))

        # Update the buffer-tank capacity.
        existing_buffer_tank_capacity: float = buffer_tank.capacity
        buffer_tank.capacity = (
            buffer_tank_capacity
            if buffer_tank_capacity is not None
            else buffer_tank.capacity
        )
        # Increase the area of the tank by a factor of the volume increase accordingly.
        buffer_tank.area *= (buffer_tank.capacity / existing_buffer_tank_capacity) ** (
            2 / 3
        )

        # Run the simulations.
        simulation_outputs = {
            profile_type.value: determine_steady_state_simulation(
                ambient_temperatures[profile_type],
                battery,
                battery_capacity,  # type: ignore [arg-type]
                buffer_tank,
                desalination_plant,
                heat_pump,
                mass_flow_rate,  # type: ignore [arg-type]
                hybrid_pv_t_panel,
                logger,
                pv_panel,
                pv_system_size,
                pv_t_system_size,
                scenario,
                solar_irradiances[profile_type],
                solar_thermal_collector,
                solar_thermal_system_size,
                system_lifetime,
                wind_speeds[profile_type],
                disable_tqdm=disable_tqdm,
            )
            for profile_type in profile_types
        }

        # Output information to the command-line interface.
        for profile_type, result in simulation_outputs.items():
            logger.info(
                "Battery lifetime degradation for %s was %s necessitating %s replacements.",
                profile_type,
                f"{result.battery_lifetime_degradation:.3g}",
                result.battery_replacements,
            )
            if result.battery_replacements > 0 and not disable_tqdm:
                print(
                    "Batteries were replaced "
                    f"{result.battery_replacements} time"
                    f"{'s' if result.battery_replacements > 1 else ''} "
                    "during the simulation."
                )
            elif not disable_tqdm:
                print("Batteries were not replaced during the simulation period.")
            if save_outputs:
                save_simulation(
                    output,
                    profile_type,
                    result,
                    solar_irradiances[ProfileType(profile_type)],
                )

        # Analyse the outputs
        logger.info("Analysing results")
        analysis = {
            key: {
                criterion: Criterion.calculate_value_map[criterion](
                    {
                        battery: battery_capacity,
                        buffer_tank: buffer_tank_capacity,
                        pv_panel: pv_system_size,
                        hybrid_pv_t_panel: pv_t_system_size,
                        solar_thermal_collector: solar_thermal_system_size,
                    },
                    logger,
                    scenario,
                    result,
                    system_lifetime,
                )
                for criterion in ANALYSIS_REQUESTS
            }
            for key, result in simulation_outputs.items()
        }

        # Return the outputs
        return {
            key: (entry.as_dataframe.to_dict(), analysis[key])
            for key, entry in simulation_outputs.items()
        }

    elif optimisation:
        # Setup a variable for storing the optimisation results.
        optimisation_results: list[
            Tuple[dict[str, Any], dict[Any, Tuple[dict[str, float], list[float]]]]
        ] = []

        for optimisation_parameters in tqdm(
            optimisations,
            desc="optimisations",
            disable=disable_tqdm,
            leave=True,
            unit="opt.",
        ):
            optimisation_results.append(
                (
                    {OPTIMISATION_PARAMETERS: optimisation_parameters.asdict},
                    {
                        profile_type.value: run_optimisation(
                            ambient_temperatures[profile_type],
                            battery,
                            buffer_tank,
                            desalination_plant,
                            heat_pump,
                            hybrid_pv_t_panel,
                            logger,
                            optimisation_parameters,
                            pv_panel,
                            scenario,
                            solar_irradiances[profile_type],
                            solar_thermal_collector,
                            system_lifetime,
                            wind_speeds[profile_type],
                        )
                        for profile_type in tqdm(
                            profile_types,
                            desc="profile type",
                            disable=disable_tqdm,
                            leave=False,
                            unit="profile",
                        )
                    },
                )
            )

        if save_outputs:
            save_optimisation(optimisation_results, output)

        return optimisation_results
    else:
        logger.error("Neither simulation or optimisation was specified. Quitting.")
        raise Exception(
            "Simultion or optimisation must be specified. Run with `--help` for more "
            "information."
        )


if __name__ == "__main__":
    # Parse the command-line arguments.
    parsed_args = parse_args(sys.argv[1:])

    # Validate that these are valid arguments.
    validate_args(parsed_args)

    # Determine the profile types that shuld be considered.
    try:
        profile_types = [
            CLI_TO_PROFILE_TYPE[entry] for entry in parsed_args.profile_types
        ]
    except KeyError as err:
        print(f"Invalid profile type. Valid types: {', '.join(CLI_TO_PROFILE_TYPE)}")
        raise

    main(
        parsed_args.location,
        profile_types,
        parsed_args.scenario,
        parsed_args.system_lifetime,
        parsed_args.battery_capacity,
        parsed_args.buffer_tank_capacity,
        parsed_args.mass_flow_rate,
        parsed_args.optimisation,
        parsed_args.output,
        parsed_args.pv_t_system_size,
        parsed_args.pv_system_size,
        parsed_args.simulation,
        parsed_args.solar_thermal_system_size,
        parsed_args.start_hour,
        verbose=parsed_args.verbose,
    )
