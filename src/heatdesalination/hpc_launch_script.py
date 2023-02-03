#!/usr/bin/python3
########################################################################################
# hpc_launch_script.py - Entry point for running HEATDesalination on Imperial's HPC.   #
#                                                                                      #
# Authors: Ben Winchester                                                              #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 29/11/2022                                                             #
# License: MIT                                                                         #
#                                                                                      #
# For more information, please email:                                                  #
#   benedict.winchester@gmail.com                                                      #
########################################################################################
"""
hpc_launch_script.py - The entry point for running HEATDesalination on the HPC.

Imperial College London owns a series of high-performance computers. This module
provides an entry point for running HEATDesalination across the various HPC computers.

References:
  - This script utilises code from CLOVER, Continuous Lifetime Optimisation of
    Variable Electricity Resources, an open-source model developed by researchers
    at Imperial College London:
        Winchester B., Beath H., Nelson J., and Sandwell P., CLOVER v5.0.5

"""

import math
import os
import subprocess
import sys
import tempfile

from logging import Logger

from .__main__ import __version__
from .__utils__ import (
    AUTO_GENERATED_FILES_DIRECTORY,
    DONE,
    FAILED,
    get_logger,
    HPCSimulation,
)
from .argparser import parse_hpc_args_and_runs

from .fileparser import INPUTS_DIRECTORY

__all__ = ("main",)


# Header string:
#   The ascii text to display when starting HEATDeslination on the HPC.
HEADER_STRING = """


     )                      (
  ( /(        (       *   ) )\\ )                  (                      )
  )\\()) (     )\\    ` )  /((()/(     (         )  )\\ (             )  ( /( (
 ((_)\\  )\\ ((((_)(   ( )(_))/(_))   ))\\ (   ( /( ((_))\\   (     ( /(  )\\()))\\   (    (
  _((_)((_) )\\ _ )\\ (_(_())(_))_   /((_))\\  )(_)) _ ((_)  )\\ )  )(_))(_))/((_)  )\\   )\\ )
 | || || __|(_)_\\(_)|_   _| |   \\ (_)) ((_)((_)_ | | (_) _(_/( ((_)_ | |_  (_) ((_) _(_/(
 | __ || _|  / _ \\    | |   | |) |/ -_)(_-</ _` || | | || ' \\))/ _` ||  _| | |/ _ \\| ' \\))
 |_||_||___|/_/ \\_\\   |_|   |___/ \\___|/__/\\__,_||_| |_||_||_| \\__,_| \\__| |_|\\___/|_||_|

                    ___                     _      _   _  _ ___  ___
                   |_ _|_ __  _ __  ___ _ _(_)__ _| | | || | _ \\/ __|
                    | || '  \\| '_ \\/ -_) '_| / _` | | | __ |  _/ (__
                   |___|_|_|_| .__/\\___|_| |_\\__,_|_| |_||_|_|  \\___|
                             |_|

                        Hybrid Electric and Thermal Desalination
                             Copyright Ben Winchester, 2022
{version_line}

    This version of HEATDesalination has been adapted fgor Imperial College London's
                              High-performance computers.

                             For more information, cont
                     Ben Winchester (benedict.winchester@gmail.com)

"""

# HPC submission command:
#   Command for submitting runs to the HPC>
HPC_SUBMISSION_COMMAND: str = "qsub"

# HPC submission script filename:
#   The name of the HPC script submission file.
HPC_SUBMISSION_SCRIPT_FILENAME: str = "hpc_template.sh"

# HPC array job submission script:
#   The path to the HPC script submission file.
HPC_SUBMISSION_SCRIPT_FILEPATH: str = os.path.join(
    "bin", HPC_SUBMISSION_SCRIPT_FILENAME
)

# Logger name:
#   The name to use for the logger for this script.
LOGGER_NAME: str = "hpc_heatdesalination_launch_script"


def _check_run(logger: Logger, hpc_run: HPCSimulation) -> bool:
    """
    Checks that the HPC run is valid.

    Inputs:
        - logger:
            The logger to use for the run.
        - hpc_run:
            The HPC run to carry out.

    Outputs:
        - Whether the run is valid or not.

    """

    # Check that the location exists.
    if not os.path.isfile(
        os.path.join(AUTO_GENERATED_FILES_DIRECTORY, f"{hpc_run.location}.json")
    ):
        logger.error(
            "Location '%s' does not exist.",
            hpc_run.location,
        )
        return False

    # Check that the simulations file exists.
    if not os.path.isfile(os.path.join(INPUTS_DIRECTORY, f"{hpc_run.simulation}.json")):
        logger.error(
            "Simulations file '%s' not found in inputs folder %s.",
            hpc_run.simulation,
            INPUTS_DIRECTORY,
        )
        return False

    # Check that the output file doesn't already exist.
    if os.path.isfile(
        output_filename := os.path.join(INPUTS_DIRECTORY, f"{hpc_run.output}.json")
    ):
        logger.error("Output file '%s' already exists.", output_filename)
        return False

    # Check that the walltime is between 1 and 72 hours.
    if hpc_run.walltime < 1 or hpc_run.walltime > 72:
        logger.error(
            "Walltime of %s hours is out of bounds. Valid hours between 1 and 72.",
            hpc_run.walltime,
        )
        return False

    return True


def main(args) -> None:
    """
    Wrapper around HEATDesalination when run on the HPC for launching jobs.

    """

    version_string = f"Version {__version__}"
    print(
        HEADER_STRING.format(
            version_line=(
                " " * (44 - math.ceil(len(version_string) / 2))
                + version_string
                + " " * (44 - math.floor(len(version_string) / 2))
            )
        )
    )

    logger = get_logger(LOGGER_NAME, False)
    logger.info("HPC-HEATDesalination script called.")
    logger.info("Arguments: %s", ", ".join(args))

    # Call the utility module to parse the HPC run information.
    runs_filename, runs, walltime = parse_hpc_args_and_runs(args, logger)
    logger.info(
        "Command-line arguments successfully parsed. Run file: %s; Walltime: %s",
        runs_filename,
        walltime,
    )

    # Check that all of the runs are valid.
    print(f"Checking HPC runs{'.'*63} ", end="")
    logger.info("Checking all run files are valid.")
    if not all(_check_run(logger, run) for run in runs):
        print(FAILED)
        logger.error(
            "Not all HPC runs were valid, exiting.",
        )
        raise Exception("Not all HPC runs were valid, see logs for details.")

    print(DONE)
    logger.info("All HPC runs valid.")

    # Parse the default HPC job submission script.
    print(f"Processing HPC job submission script{'.'*44} ", end="")
    logger.info("Parsing base HPC job submission script.")
    try:
        with open(HPC_SUBMISSION_SCRIPT_FILEPATH, "r") as f:
            hpc_submission_script_file_contents = f.read()
    except FileNotFoundError:
        print(FAILED)
        logger.error(
            "HPC job submission file not found. Check that the file, '%s', has not "
            "been removed.",
            HPC_SUBMISSION_SCRIPT_FILEPATH,
        )
        raise

    print(DONE)
    logger.info("HPC job submission file successfully parsed.")

    # Update the template file as a temporary file with the information needed.
    hpc_submission_script_file_contents = hpc_submission_script_file_contents.format(
        NUM_RUNS=len(runs),
        RUNS_FILE=runs_filename,
        WALLTIME=f"{walltime}" if walltime is not None else "",
    )
    logger.info(
        "HPC job submission script updated with %s runs, %s walltime.",
        len(runs_filename),
        walltime,
    )

    # Setup the HPC job submission script.
    with tempfile.TemporaryDirectory() as tmpdirname:
        hpc_submission_script_filepath = os.path.join(
            tmpdirname, HPC_SUBMISSION_SCRIPT_FILENAME
        )

        # Write the submission script file.
        logger.info("Writing temporary HPC submission script.")
        with open(hpc_submission_script_filepath, "w") as f:
            f.write(hpc_submission_script_file_contents)

        logger.info("HPC job submission script successfully submitted.")

        # Update permissions on the file.
        os.chmod(hpc_submission_script_filepath, 0o775)
        logger.info("HPC job submission script permissions successfully updated.")

        # Submit the script to the HPC.
        logger.info("Submitting HEATDeslination jobs to the HPC.")
        print("Sending jobs to the HPC:")
        try:
            subprocess.run(
                [HPC_SUBMISSION_COMMAND, hpc_submission_script_filepath], check=False
            )
        except Exception:  # pylint: disable=broad-except
            logger.error("Failed. See logs for details.")
            print(f"Sending jobs to the HPC{'.'*57} {FAILED}")
            raise
        print(f"Sending jobs to the HPC{'.'*57} {DONE}")
        logger.info("HPC runs submitted. Exiting.")


if __name__ == "__main__":
    main(sys.argv[1:])
