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

from multiprocessing import pool
from typing import Any, Dict, List

from tqdm import tqdm

from src.heatdesalination.__main__ import main as heatdesalination_main
from src.heatdesalination.__utils__ import (
    CLI_TO_PROFILE_TYPE,
    get_logger,
    ProfileType,
    read_yaml,
    Solution,
)
from src.heatdesalination.fileparser import INPUTS_DIRECTORY
from src.heatdesalination.simulator import determine_steady_state_simulation

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
    profile_types: List[str]
    pv_system_size: float
    pv_t_system_size: float
    scenario: str
    solar_thermal_system_size: float
    start_hour: int
    system_lifetime: int

    @property
    def profile_type_instances(self) -> List[ProfileType]:
        """
        Return :class:`ProfileType` instances.

        Outputs:
            A `list` of the profile types.

        """

        return [CLI_TO_PROFILE_TYPE[entry] for entry in self.profile_types]


def _parse_args(args: List[Any]) -> argparse.Namespace:
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

    return parser.parse_args(args)


def heatdesalination_wrapper(
    simulation: Simulation, location: str
) -> Dict[ProfileType, Solution]:
    """
    Run a steady-state simulation

    """

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
    )


def main(args: List[Any]) -> List[Any]:
    """
    Main method for carrying out multiple simulations..

    Inputs:
        - args:
            The command-line arguments.

    """

    # Parse the command-line arguments.
    parsed_args = _parse_args(args)
    logger = get_logger(f"{parsed_args.location}_parallel_simulator")

    # Parse the simulations file.
    with open(SIMULATIONS_FILEPATH, "r") as simulations_file:
        simulations = [Simulation(**entry) for entry in json.load(simulations_file)]

    print(f"Carrying out parallel simulation{'.'*37} ", end="")
    logger.info("Carrying out %s parallel simulation(s)", len(simulations))
    # Carry out the simulations as necessary.
    with pool.Pool(8) as worker_pool:
        results = worker_pool.map(
            functools.partial(heatdesalination_wrapper, location=parsed_args.location),
            simulations,
        )
    logger.info("Worker pool complete, %s results generated", len(results))
    if len(results) != len(simulations):
        logger.error(
            "Results and simulations length mismatch: %s results for %s simulations",
            len(results),
            len(simulations),
        )
    print("[  DONE  ]")

    # Convert to a mapping from simulation information.
    print(f"Prepping results map{'.'*49} ", end="")
    results_map = [
        {
            "simulation": dataclasses.asdict(simulations[index]),
            "results": results[index],
        }
        for index in range(len(results))
    ]

    print("[  DONE  ]")

    print(f"Saving output file{'.'*51} ", end="")
    if parsed_args.output is not None:
        with open(f"{parsed_args.output}.json", "w") as f:
            json.dump(results_map, f)
    print("[  DONE  ]")

    print("Exiting")
    return results


if __name__ == "__main__":
    main(sys.argv[1:])
