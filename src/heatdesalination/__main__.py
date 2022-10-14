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

def main(args: List[Any]) -> None:
    """
    Main module responsible for the flow of the HEATDesalination program.

    Inputs:
        - args:
            The un-parsed command-line arguments.

    """

    # Parse the command-line arguments.
    # Parse the various input files.

    # IF: Simulation
    #     THEN: Carry out a simulation based on the input conditions and exit.
    # IF: Optimisation
    #     THEN: Carry out an optimisation and exit.
    # Raise an error stating that the code didn't run.
    #

if __name__ == "__main__":
    main(sys.argv[1:])