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

from typing import Any, Dict, List

from tqdm import tqdm

from .__utils__ import ProfileType, Solution, get_logger
from .argparser import MissingParametersError, parse_args, validate_args
from .fileparser import parse_input_files
from .optimiser import run_optimisation
from .simulator import determine_steady_state_simulation

# SIMULATION_OUTPUTS_DIRECTORY:
#   The outputs dierctory for simulations.
SIMULATION_OUTPUTS_DIRECTORY: str = "simulation_outputs"


def save_simulation(
    output: str,
    profile_type: ProfileType,
    simulation_outputs: Solution,
    solar_irradiance: Dict[int, float],
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
    output_data = simulation_outputs.to_dataframe()
    output_data["Solar irradiance / W/m^2"] = solar_irradiance

    # Write to the output file.
    os.makedirs(SIMULATION_OUTPUTS_DIRECTORY, exist_ok=True)
    with open(
        f"{os.path.join(SIMULATION_OUTPUTS_DIRECTORY, output)}_{profile_type.value}"
        ".csv",
        "w",
        encoding="UTF-8",
    ) as output_file:
        output_data.to_csv(output_file)


def main(args: List[Any]) -> None:
    """
    Main module responsible for the flow of the HEATDesalination program.

    Inputs:
        - args:
            The un-parsed command-line arguments.

    """

    # Parse the command-line arguments.
    parsed_args = parse_args(args)
    validate_args(parsed_args)
    logger = get_logger(f"{parsed_args.location}_heat_desalination")

    # Parse the various input files.
    (
        ambient_temperatures,
        battery,
        buffer_tank,
        desalination_plant,
        hybrid_pv_t_panel,
        optimisations,
        pv_panel,
        scenario,
        solar_irradiances,
        solar_thermal_collector,
    ) = parse_input_files(
        parsed_args.location, logger, parsed_args.scenario, parsed_args.start_hour
    )

    if parsed_args.simulation:
        # Raise exceptions if the arguments are invalid.
        missing_parameters: List[str] = []
        if scenario.battery and not parsed_args.battery_capacity:
            logger.error(
                "Must specify battery capacity if batteries included in scenario."
            )
            missing_parameters.append("Storage system size")
        if scenario.pv_t and not parsed_args.pv_t_system_size:
            logger.error(
                "Must specify PV-T system size if PV-T collectors included in scenario."
            )
            missing_parameters.append("PV-T system size")
        if scenario.solar_thermal and not parsed_args.solar_thermal_system_size:
            logger.error(
                "Must specify solar-thermal system size if solar-thermal collectors "
                "included in scenario."
            )
            missing_parameters.append("Solar-thermal system size")
        if not parsed_args.mass_flow_rate:
            logger.error("Must specify HTF mass flow rate if running a simulation.")
            missing_parameters.append("HTF mass flow rate")

        if len(missing_parameters) > 0:
            raise MissingParametersError(", ".join(missing_parameters))

        # Run the simulation.
        for profile_type in ProfileType:
            simulation_outputs = determine_steady_state_simulation(
                ambient_temperatures[profile_type],
                battery,
                parsed_args.battery_capacity,
                buffer_tank,
                desalination_plant,
                parsed_args.mass_flow_rate,
                hybrid_pv_t_panel,
                logger,
                pv_panel,
                parsed_args.pv_system_size,
                parsed_args.pv_t_system_size,
                scenario,
                solar_irradiances[profile_type],
                solar_thermal_collector,
                parsed_args.solar_thermal_system_size,
                parsed_args.system_lifetime,
            )
            save_simulation(
                parsed_args.output,
                profile_type,
                simulation_outputs,
                solar_irradiances[profile_type],
            )
    elif parsed_args.optimisation:
        for optimisation_parameters in tqdm(
            optimisations, desc="optimisations", leave=True, unit="opt."
        ):
            for profile_type in tqdm(
                ProfileType, desc="profile type", leave=False, unit="profile"
            ):
                result = run_optimisation(
                    ambient_temperatures[profile_type],
                    battery,
                    buffer_tank,
                    desalination_plant,
                    hybrid_pv_t_panel,
                    logger,
                    optimisation_parameters,
                    pv_panel,
                    scenario,
                    solar_irradiances[profile_type],
                    solar_thermal_collector,
                    parsed_args.system_lifetime,
                )
                import pdb

                pdb.set_trace()
    else:
        logger.error("Neither simulation or optimisation was specified. Quitting.")
        raise Exception(
            "Simultion or optimisation must be specified. Run with `--help` for more "
            "information."
        )


if __name__ == "__main__":
    main(sys.argv[1:])
