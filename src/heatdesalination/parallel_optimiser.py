#!/usr/bin/python3.10
########################################################################################
# parallel_optimiser.py - Module for enabling parallel optimisations.                    #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 24/11/2022                                                             #
# License: MIT, Open-source                                                            #
# For more information, contact: benedict.winchester@gmail.com                         #
########################################################################################

"""
parallel_optimiser.py - The simulation module for the HEATDeslination program.

Module for enabling external scripts for carrying out multiple optimisations for a
given location based on some input files.

"""

import dataclasses
import os
import json
import sys

import argparse

from logging import Logger
from multiprocessing import pool
from typing import Any

from tqdm import tqdm

from .__main__ import main as heatdesalination_main
from .__utils__ import (
    CLI_TO_PROFILE_TYPE,
    DONE,
    get_logger,
    ProfileType,
)
from .fileparser import INPUTS_DIRECTORY

# PARALLEL_OPTIMISATIONS_FILEPATH:
#   The file path to the optimisations file.
PARALLEL_OPTIMISATIONS_FILEPATH: str = os.path.join(
    INPUTS_DIRECTORY, "optimisations.json"
)


@dataclasses.dataclass
class Optimisation:
    """
    Represents an optimisation that can be carried out in parallel.

    .. attribute:: location
        The name of the location to use.

    .. attribute:: output
        The name of the output file to use.

    .. attribute:: profile_types
        The list of profile types to be carried out.

    .. attribute:: scenario
        The name of the scenario to use.

    .. attribute:: system_lifetime
        The lifetime of the system in years.

    """

    location: str
    output: str
    profile_types: list[str]
    scenario: str
    system_lifetime: int

    @property
    def profile_type_instances(self) -> list[ProfileType]:
        """
        Return :class:`ProfileType` instances.

        Outputs:
            A `list` of the profile types.

        """

        return [CLI_TO_PROFILE_TYPE[entry] for entry in self.profile_types]


def _parse_args(args: list[Any]) -> argparse.Namespace:
    """
    Parses command-line arguments into a :class:`argparse.NameSpace`.

    Inputs:
        The unparsed command-line arguments.

    Outputs:
        The parsed command-line arguments.

    """

    parser = argparse.ArgumentParser()

    ######################
    # Required arguments #
    ######################

    # Optimisations:
    #   The name of the optimisations input file to use.
    parser.add_argument(
        "--optimisations-file",
        "-o",
        default=None,
        help="The name of the input optimisations file to use.",
        type=str,
    )

    # Output:
    #   The name of the output file to use.
    parser.add_argument(
        "--output-file",
        "--output",
        "-out",
        default="parallel_optimisation_results",
        help="The name of the output file to save the results to.",
        type=str,
    )

    ######################
    # Optional arguments #
    ######################

    # Full results:
    #   Whether to store the full results (True) or not (False).
    parser.add_argument(
        "--partial-results",
        "-p",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )

    return parser.parse_args(args)


def heatdesalination_wrapper(optimisation: Optimisation) -> Any:
    """
    Run a standard optimisation.

    Inputs:
        - optimisation:
            The optimisation to run.

    Outputs:
        The results of the parallel optimisation.

    """

    return heatdesalination_main(
        optimisation.location,
        optimisation.profile_type_instances,
        optimisation.scenario,
        optimisation.system_lifetime,
        None,
        None,
        None,
        True,
        optimisation.output,
        None,
        None,
        False,
        None,
        None,
        disable_tqdm=True,
        hpc=True,
        save_outputs=False,
    )


def main(
    logger: Logger,
    optimisations_file: str,
    output_file: str,
) -> list[Any]:
    """
    Main method for carrying out multiple optimisations.

    Inputs:
        - logger:
            The logger to use if specified.
        - optimisations_file:
            The name of the optimisations file to use.

    Outputs:
        - A `list` containing the outputs of the parallel optimisation.

    """

    # Use the inputted optimisations filepath if provided.
    if (optimisations_filename := optimisations_file) is not None:
        optimisations_filepath: str = os.path.join(
            INPUTS_DIRECTORY, f"{optimisations_filename}.json"
        )
    else:
        optimisations_filepath = PARALLEL_OPTIMISATIONS_FILEPATH

    # Parse the optimisations file.
    with open(optimisations_filepath, "r", encoding="UTF-8") as open_optimisations_file:
        optimisations = [
            Optimisation(**entry) for entry in json.load(open_optimisations_file)
        ]

    print(f"Carrying out parallel optimisation{'.'*35} ", end="")
    logger.info("Carrying out %s parallel simulation(s)", len(optimisations))
    # Carry out the optimisations as necessary.
    with pool.Pool(min(8, len(optimisations))) as worker_pool:
        results = list(
            tqdm(
                worker_pool.imap(
                    heatdesalination_wrapper,
                    optimisations,
                ),
                desc="parallel_optimisations",
                unit="process",
                total=len(optimisations),
            )
        )
    logger.info("Worker pool complete, %s results generated", len(results))
    if len(results) != len(optimisations):
        logger.error(
            "Results and optimisations length mismatch: %s results for %s optimisations",
            len(results),
            len(optimisations),
        )
    print(DONE)

    # Convert to a mapping from simulation information.
    print(f"Prepping results map{'.'*49} ", end="")
    output_data = [
        {
            "optimisation": dataclasses.asdict(optimisations[index]),
            "result": results[index],
        }
        for index in range(len(results))
    ]

    print(DONE)

    print(f"Saving output file{'.'*51} ", end="")
    with open(f"{output_file}.json", "w", encoding="UTF-8") as f:
        json.dump(output_data, f)
    print(DONE)

    print("Exiting")
    return results


if __name__ == "__main__":
    # Parse the command-line arguments.
    parsed_args = _parse_args(sys.argv[1:])

    main(
        get_logger("parallel_optimiser"),
        parsed_args.optimisations_file,
        parsed_args.output_file,
    )
