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

import sys

from typing import Any, List


from .__utils__ import ProfileType, get_logger
from .argparser import parse_args, validate_args
from .fileparser import parse_input_files
from .simulator import run_simulation


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
        pv_panel,
        scenario,
        solar_irradiances,
        solar_thermal_collector,
    ) = parse_input_files(
        parsed_args.location, logger, parsed_args.scenario, parsed_args.start_hour
    )

    if parsed_args.simulation:
        # Raise exceptions if the arguments are invalid.
        if scenario.pv_t and not parsed_args.pv_t_system_size:
            logger.error(
                "Must specify PV-T system size if PV-T collectors included in scenario."
            )
            raise Exception("Missing PV-T system size argument.")
        if scenario.solar_thermal and not parsed_args.solar_thermal_system_size:
            logger.error(
                "Must specify solar-thermal system size if solar-thermal collectors "
                "included in scenario."
            )
            raise Exception("Missing solar-thermal system size argument.")

        # Run the simulation.
        for profile_type in ProfileType:
            (
                collector_input_temperatures,
                collector_system_output_temperatures,
                pv_electrical_efficiencies,
                pv_electrical_output_power,
                pv_t_electrical_efficiencies,
                pv_t_electrical_output_power,
                pv_t_htf_output_temperatures,
                pv_t_reduced_temperatures,
                pv_t_thermal_efficiencies,
                solar_thermal_htf_output_temperatures,
                solar_thermal_reduced_temperatures,
                solar_thermal_thermal_efficiencies,
                tank_temperatures,
            ) = run_simulation(
                ambient_temperatures[profile_type],
                buffer_tank,
                desalination_plant,
                parsed_args.mass_flow_rate,
                hybrid_pv_t_panel,
                logger,
                pv_panel,
                parsed_args.pv_t_system_size,
                scenario,
                solar_irradiances[profile_type],
                solar_thermal_collector,
                parsed_args.solar_thermal_system_size,
            )
            import pdb

            pdb.set_trace()
    elif parsed_args.optimisation:
        run_optimisation()
    else:
        logger.error("Neither simulation or optimisation was specified. Quitting.")
        raise Exception(
            "Simultion or optimisation must be specified. Run with `--help` for more "
            "information."
        )


if __name__ == "__main__":
    main(sys.argv[1:])
