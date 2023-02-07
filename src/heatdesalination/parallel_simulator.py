#!/usr/bin/python3.10
########################################################################################
# parallel_simulator.py - Module for enabling parallel simulations.                    #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 24/11/2022                                                             #
# License: MIT, Open-source                                                            #
# For more information, contact: benedict.winchester@gmail.com                         #
########################################################################################

"""
parallel_simulator.py - The simulation module for the HEATDeslination program.

Module for enabling external scripts for carrying out multiple optimisations for a
given location based on some input files.

"""

import argparse
import dataclasses
import functools
import os
import json
import sys

from logging import Logger
from multiprocessing import pool
from typing import Any

from tqdm import tqdm

from src.heatdesalination.__main__ import main as heatdesalination_main
from src.heatdesalination.__utils__ import (
    CLI_TO_PROFILE_TYPE,
    DONE,
    FlowRateError,
    get_logger,
    ProfileType,
    Solution,
)
from src.heatdesalination.fileparser import INPUTS_DIRECTORY

# SIMULATIONS_FILEPATH:
#   The file path to the simulations file.
SIMULATIONS_FILEPATH: str = os.path.join(INPUTS_DIRECTORY, "simulations.json")


@dataclasses.dataclass
class Simulation:
    """
    Represents a simulation that can be carried out.

    .. attribute:: battery_capacity
        The battery capacity in kWh.

    .. attribute:: buffer_tank_capacity
        The buffer-tank capacity in litres.

    .. attribute:: mass_flow_rate
        The mass flow rate in kg/s.

    .. attribute:: output
        The name of the output file to use.

    .. attribute:: profile_types
        The list of profile types to be carried out.

    .. attribute:: pv_system_size
        The size of the PV system in numbers of collectors.

    .. attribute:: pv_t_system_size
        The size of the PV-T system in numbers of collectors.

    .. attribute:: scenario
        The name of the scenario to use.

    .. attribute:: solar_thermal_system_size
        The size of the solar-thermal system in numbers of collectors.

    .. attribute:: start_hour
        The start hour for the desalination plant.

    .. attribute:: system_lifetime
        The lifetime of the system in years.

    """

    battery_capacity: float
    buffer_tank_capacity: float
    mass_flow_rate: float
    output: str
    profile_types: list[str]
    pv_system_size: float
    pv_t_system_size: float
    scenario: str
    solar_thermal_system_size: float
    start_hour: int
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

    required_arguments = parser.add_argument_group("required arguments")

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

    # Output:
    #   The name of the file to save the output results from the simulations to.
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="The name of the output file to use for the multiple results.",
        type=str,
    )

    # Simulations:
    #   The name of the simulations input file to use.
    parser.add_argument(
        "--simulations-file",
        "-s",
        default=None,
        help="The name of the input simulations file to use.",
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


def heatdesalination_wrapper(
    simulation: Simulation, hpc: bool, location: str
) -> dict[ProfileType, Solution] | None:
    """
    Run a steady-state simulation

    Inputs:
        - simulation:
            The simulation to carry out.
        - hpc:
            Whether the program is being run on the HPC (True) or not (False).
        - location:
            The location for which to run the simulation.

    Outputs:
        - The results of the simulation or `None` if a flow-rate error was
          encountered.

    """

    try:
        return heatdesalination_main(
            location,
            simulation.profile_type_instances,
            simulation.scenario,
            simulation.system_lifetime,
            simulation.battery_capacity,
            simulation.buffer_tank_capacity,
            simulation.mass_flow_rate,
            False,
            simulation.output,
            simulation.pv_t_system_size,
            simulation.pv_system_size,
            True,
            simulation.solar_thermal_system_size,
            simulation.start_hour,
            disable_tqdm=True,
            save_outputs=False,
            hpc=hpc,
        )
    except FlowRateError:
        print("Flow-rate error, skipping results.")
        return None


def main(
    location: str,
    logger: Logger,
    output: str,
    simulations_file: str,
    full_results: bool = True,
    hpc: bool = False,
) -> list[Any]:
    """
    Main method for carrying out multiple simulations.

    Inputs:
        - location:
            The name of the location to use for the weather data.
        - logger:
            The logger to use if specified.
        - output:
            THe mame of the output file to use.
        - simulations_file:
            The name of the simulations file to use.
        - full_results:
            Whether to record the full results (True) or reduced results (False).
        - hpc:
            Whether the script is being run on the HPC (True) or not (False).

    """

    # Use the inputted simulations filepath if provided.
    if (simulations_filename := simulations_file) is not None:
        simulations_filepath: str = os.path.join(
            INPUTS_DIRECTORY, f"{simulations_filename}.json"
        )
    else:
        simulations_filepath = SIMULATIONS_FILEPATH

    # Parse the simulations file.
    with open(simulations_filepath, "r") as simulations_file:
        simulations = [Simulation(**entry) for entry in json.load(simulations_file)]

    print(f"Carrying out parallel simulation{'.'*37} ", end="")
    logger.info("Carrying out %s parallel simulation(s)", len(simulations))
    # Carry out the simulations as necessary.
    with pool.Pool(min(8, len(simulations))) as worker_pool:
        results = list(
            tqdm(
                worker_pool.imap(
                    functools.partial(
                        heatdesalination_wrapper, hpc=hpc, location=location
                    ),
                    simulations,
                ),
                total=len(simulations),
            )
        )
    logger.info("Worker pool complete, %s results generated", len(results))
    if len(results) != len(simulations):
        logger.error(
            "Results and simulations length mismatch: %s results for %s simulations",
            len(results),
            len(simulations),
        )
    print()

    # Convert to a mapping from simulation information.
    print(f"Prepping results map{'.'*49} ", end="")
    if full_results:
        results_map = [
            {
                "simulation": dataclasses.asdict(simulations[index]),
                "results": results[index],
            }
            for index in range(len(results))
        ]
    else:
        results_map = [
            {
                "simulation": dataclasses.asdict(simulations[index]),
                "results": {key: value[1] for key, value in results[index].items()},
            }
            for index in range(len(results))
        ]

    print(DONE)

    print(f"Saving output file{'.'*51} ", end="")
    if output is not None:
        with open(f"{output}.json", "w") as f:
            json.dump(results_map, f)
    print(DONE)

    print("Exiting")
    return results


if __name__ == "__main__":
    # Parse the command-line arguments.
    parsed_args = _parse_args(sys.argv[1:])

    # Setup the logger.
    logger = get_logger(f"{parsed_args.location}_parallel_simulator")

    main(
        parsed_args.location,
        logger,
        parsed_args.output,
        parsed_args.simulations_file,
        not parsed_args.partial_results,
    )
