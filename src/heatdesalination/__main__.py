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

from heatdesalination.__utils__ import get_logger

from .argparser import parse_args
from .fileparser import parse_input_files


def main(args: List[Any]) -> None:
    """
    Main module responsible for the flow of the HEATDesalination program.

    Inputs:
        - args:
            The un-parsed command-line arguments.

    """

    # Parse the command-line arguments.
    parsed_args = parse_args(args)
    logger = get_logger(f"{parsed_args.location}_heat_desalination")

    # Parse the various input files.
    parse_input_files(
        parsed_args.location, logger, parsed_args.scenario, parsed_args.start_hour
    )

    if parsed_args.simulation:
        run_simulation()
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
