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

__all__ = ("parse_args",)


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
        type=int,
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
        "--simulation",
        "-sim",
        action="store_true",
        default=False,
        help="Run a simulation.",
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
        "--solar-thermal-system-size",
        "-st",
        help="The number of solar-thermal collectors to use.",
        type=int,
    )

    return parser.parse_args(args)
