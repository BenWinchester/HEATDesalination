#!/usr/bin/python3.10
########################################################################################
# argparser.py - The argument-parsing module                                           #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 14/10/2022                                                             #
# License: MIT, Open-source                                                            #
# For more information, contact: benedict.winchester@gmail.com                         #
########################################################################################

"""
argparser.py - The argument parser module for the HEATDeslination program.

"""


import argparse
import os

from logging import Logger
from typing import Any, Tuple

import json

from src.heatdesalination.__utils__ import (
    CLI_TO_PROFILE_TYPE,
    DEFAULT_SIMULATION_OUTPUT_FILE,
    HPCSimulation,
    WALLTIME,
)

__all__ = (
    "parse_args",
    "validate_args",
)


class MissingParametersError(Exception):
    """
    Raised when not all parameters have been specified on the command line.
    """

    def __init__(self, missing_parameter: str) -> None:
        """
        Instantiate a missing parameters error.

        Inputs:
            - missing_parameter:
                The parameter which has not been specified.

        """

        super().__init__(
            f"Missing command-line parameters: {missing_parameter}. "
            + "Run `heat-desalination --help` for more information."
        )


def parse_args(args: list[Any]) -> argparse.Namespace:
    """
    Parses command-line arguments into a :class:`argparse.NameSpace`.

    Inputs:
        The unparsed command-line arguments.

    Outputs:
        The parsed command-line arguments.

    """

    parser = argparse.ArgumentParser()

    required_arguments = parser.add_argument_group("required arguments")
    simulation_arguments = parser.add_argument_group("simulation-only arguments")
    optimisation_arguments = parser.add_argument_group("optimisation-only arguments")

    ######################
    # Required arguments #
    ######################

    # Location/Weather:
    #   The weather information to use.
    required_arguments.add_argument(
        "--location",
        "--weather",
        "-l",
        "-w",
        help="The name of the weather inputs file to use.",
        type=str,
    )

    # Profile types:
    #   The profile types to use for the modelling.
    required_arguments.add_argument(
        "--profile-types",
        "-pt",
        default=[],
        help="The profile types to use. Valid options: "
        f"{', '.join(CLI_TO_PROFILE_TYPE)}",
        nargs="*",
        type=str,
    )

    # Scenario:
    #   The scenario to use for the modelling.
    required_arguments.add_argument(
        "--scenario",
        help="The scenario to use for the modelling.",
        type=str,
    )

    # Start hour:
    #   The start hour at which to being modelling.
    required_arguments.add_argument(
        "--start-hour",
        "-sh",
        help="The start time for the desalination plant to begin operation.",
        type=int,
    )
    required_arguments.add_argument(
        "--system-lifetime",
        "-sl",
        help="The lifetime of the installed system, measured in years.",
        type=int,
    )

    #############################
    # Simulation-only arguments #
    #############################

    simulation_arguments.add_argument(
        "--battery-capacity",
        "-b",
        help="The capacity of the installed storage system, measured in kWh.",
        type=float,
    )
    simulation_arguments.add_argument(
        "--buffer-tank-capacity",
        "-t",
        help="The capacity of the installed buffer tank, measured in kg or litres.",
        type=float,
    )
    simulation_arguments.add_argument(
        "--mass-flow-rate",
        "-m",
        help="The mass flow rate through the collector system in kilograms per second.",
        type=float,
    )
    simulation_arguments.add_argument(
        "--output",
        "-o",
        default=DEFAULT_SIMULATION_OUTPUT_FILE,
        help="The name of the output file.",
        type=str,
    )
    simulation_arguments.add_argument(
        "--pv-system-size",
        "-pv",
        help="The number of PV panels to use.",
        type=int,
    )
    simulation_arguments.add_argument(
        "--pv-t-system-size",
        "-pv-t",
        help="The number of PV-T collectors to use.",
        type=float,
    )
    simulation_arguments.add_argument(
        "--simulation",
        "-sim",
        action="store_true",
        default=False,
        help="Run a simulation.",
    )
    simulation_arguments.add_argument(
        "--solar-thermal-system-size",
        "-st",
        help="The number of solar-thermal collectors to use.",
        type=float,
    )

    ###############################
    # Optimisation-only arguments #
    ###############################

    optimisation_arguments.add_argument(
        "--optimisation",
        "-opt",
        action="store_true",
        default=False,
        help="Run an optimisation.",
    )

    ####################
    # Hidden arguments #
    ####################

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )

    return parser.parse_args(args)


def _parse_hpc_args(args: list[Any]) -> argparse.Namespace:
    """
    Parses command-line arguments into a :class:`argparse.NameSpace`.

    Inputs:
        The unparsed command-line arguments.

    Outputs:
        The parsed command-line arguments.

    """

    parser = argparse.ArgumentParser()

    required_arguments = parser.add_argument_group("required arguments")

    ######################
    # Required arguments #
    ######################

    # Location/Weather:
    #   The weather information to use.
    required_arguments.add_argument(
        "--runs",
        "-r",
        help="The path to the runs file to use.",
        type=str,
    )

    ######################
    # Optional arguments #
    ######################

    parser.add_argument(
        "--walltime",
        "-w",
        default=None,
        help="The walltime in hours.",
        type=int,
    )

    return parser.parse_args(args)


def parse_hpc_args_and_runs(
    args: list[Any], logger: Logger
) -> Tuple[str, list[HPCSimulation], int | None]:
    """
    Parse the arguments and runs.

    Inputs:
        - args:
            The unparsed command-line arguments.
        - logger:
            The logger to use for the run.

    Outputs:
        - run_filename:
            The name of the runs file to carry out.
        - runs:
            The runs to carry out.
        - walltime:
            The walltime to use

    """

    # Parse the command-line arguments.
    parsed_args = _parse_hpc_args(args)
    logger.info("Command-line arguments parsed.")

    # Exit if the runs file does not exist.
    if (runs_filename := parsed_args.runs) is None:
        raise MissingParametersError("runs")
    if not os.path.isfile(parsed_args.runs):
        raise FileNotFoundError(f"HPC runs file {runs_filename} could not be found.")

    # Open the runs file and parse the information.
    logger.info("Parsing runs file.")
    with open(runs_filename, "r") as f:
        runs_file_data = json.load(f)

    # Update the walltime if necessary.
    if parsed_args.walltime is not None:
        logger.info("Walltime of %s passed in on the CLI. Updating runs.")
        for entry in runs_file_data:
            entry[WALLTIME] = parsed_args.walltime

    runs = [HPCSimulation(**entry) for entry in runs_file_data]
    logger.info("Runs file parsed: %s runs to carry out.", len(runs))

    return runs_filename, runs, parsed_args.walltime


def validate_args(parsed_args: argparse.Namespace) -> None:
    """
    Validates the command-line arguments.

    Inputs:
        - parsed_args:
            The parsed command-line arguments.

    Raises:
        - Exception:
            Raised if the arguments are invalid.

    """

    if parsed_args.location is None:
        raise Exception("Location must be specified.")
    if parsed_args.scenario is None:
        raise Exception("Scenario must be specified.")
    if parsed_args.profile_types is None:
        raise Exception("Profile types must be specified.")
    if parsed_args.simulation and parsed_args.optimisation:
        raise Exception("Cannot run an optimisation and a simulation.")
    if not parsed_args.simulation and not parsed_args.optimisation:
        raise Exception("Must run either a simulation or an optimisation.")
    if parsed_args.start_hour is None and parsed_args.simulation:
        raise Exception(
            "Start hour for desalination plant operation must be specified if running "
            "a simulation."
        )
    if parsed_args.system_lifetime is None:
        raise Exception("Must provide system lifetime for the operation of the plant.")
