#!/usr/bin/python3
########################################################################################
# hpc.py - Wrapper script around HEATDesalination when run on Imperial College's HPC.  #
#                                                                                      #
# Authors: Phil Sandwell, Ben Winchester                                               #
# Copyright: Phil Sandwell, 2022                                                       #
# Date created: 29/03/2022                                                             #
# License: MIT                                                                         #
#                                                                                      #
# For more information, please email:                                                  #
#   benedict.winchester@gmail.com                                                      #
#   philip.sandwell@gmail.com                                                          #
########################################################################################
"""
hpc.py - The wrapper script for running HEATDesalination on the HPC.

Imperial College London owns a series of high-performance computers. This module
provides a wrapper around the main functionality of HEATDesalination to enable it to be
run on the HPC.

"""

import os
import sys

from typing import Any

from .__utils__ import get_logger
from .argparser import parse_hpc_args_and_runs
from .parallel_simulator import main as parallel_simulator_main


__all__ = ("main",)

# HPC Job Number:
#   Name of the environment variable for the HPC job number.
HPC_JOB_NUMBER: str = "PBS_ARRAY_INDEX"


# Logger name:
#   The name to use for the logger for this script.
LOGGER_NAME: str = "hpc_run_{}"


def main(args: list[Any]) -> None:
    """
    Wrapper around HEATDesalination when run on the HPC.

    """

    # Determine the run that is to be carried out.
    try:
        hpc_job_number = int(os.getenv(HPC_JOB_NUMBER))  # type: ignore
    except ValueError:
        print(f"HPC environmental variable {HPC_JOB_NUMBER} was not of type int.")
        raise

    # Use a separate logger for each run accordingly.
    logger = get_logger(LOGGER_NAME.format(hpc_job_number), False)
    logger.info("HPC run script executed.")
    logger.info("CLI arguments: %s", ", ".join(args))

    # Call the utility module to parse the HPC run information.
    logger.info("Parsing HPC input file.")
    _, runs, _ = parse_hpc_args_and_runs(args, logger)
    logger.info("HPC input file successfully parsed.")

    # Sanitise the jobn number.
    run_number: int = hpc_job_number - 1

    # Fetch the appropriate run from the list of runs.
    try:
        hpc_run = runs[run_number]
    except IndexError:
        logger.error(
            "Run number %s out of bounds. Only %s runs submitted. Exiting.",
            hpc_job_number,
            len(runs),
        )
        raise
    logger.info("Run successfully determined: %s", str(hpc_run))

    logger.info("Carrying out run.")
    parallel_simulator_main(
        hpc_run.location,
        logger,
        hpc_run.output,
        hpc_run.simulation,
        full_results=False,
        hpc=True,
    )
    logger.info("Run successfully exectued, exiting.")


if __name__ == "__main__":
    main(sys.argv[1:])
