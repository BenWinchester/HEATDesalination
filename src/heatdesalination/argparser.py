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

from typing import Any, List

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


def parse_args(args: List[Any]) -> argparse.Namespace:
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
        "--start",
        help="The start time for the desalination plant to begin operation.",
        type=int,
    )

    #############################
    # Simulation-only arguments #
    #############################

    simulation_arguments.add_argument(
        "--mass-flow-rate",
        "-m",
        help="The mass flow rate through the collector system in kilograms per second.",
        type=float,
    )
    simulation_arguments.add_argument(
        "--output",
        "-o",
        default="simulation_output",
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
        type=int,
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
        type=int,
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

    return parser.parse_args(args)


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
    if parsed_args.simulation and parsed_args.optimisation:
        raise Exception("Cannot run an optimisation and a simulation.")
    if not parsed_args.simulation and not parsed_args.optimisation:
        raise Exception("Must run either a simulation or an optimisation.")
    if parsed_args.start_hour is None and parsed_args.simulation:
        raise Exception(
            "Start hour for desalination plant operation must be specified if running "
            "a simulation."
        )
