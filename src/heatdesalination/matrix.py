#!/usr/bin/python3.10
########################################################################################
# matrix.py - The matrix construction and solving module                               #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 14/10/2022                                                             #
# License: MIT, Open-source                                                            #
# For more information, contact: benedict.winchester@gmail.com                         #
########################################################################################

"""
matrix.py - The matrix module for the HEATDeslination program.

The matrix module is responsible for constructing and solving matrix equations required
to simulate the performance of the various components of the heat-supplying solar
system which consists of solar-thermal and PV-T collectors depending on the
configuration.

"""

from typing import Dict, Optional, Tuple

from .__utils__ import Scenario
from .solar import HybridPVTPanel, PVPanel, SolarThermalPanel


# Temperature precision:
#   The precision required when solving the matrix equation for the system temperatures.
TEMPERATURE_PRECISION: float = 1.44


def solve_matrix(
    ambient_temperature: Dict[int, float],
    hybrid_pvt_panel: Optional[HybridPVTPanel],
    pv_panel: Optional[PVPanel],
    scenario: Scenario,
    solar_irradiance: Dict[int, float],
    solar_thermal_collector: Optional[SolarThermalPanel],
    *,
    time_index: int,
) -> Tuple[float]:
    """
    Solve the matrix equation for the performance of the solar system.

    Inputs:
        - ambient_temperature:
            The ambient temperature profile.
        - hybrid_pvt_panel:
            The :class:`HybridPVTPanel` to use for the run if appropriate.
        - pv_panel:
            The :class:`PVPanel` to use for the run if appropriate.
        - scenario:
            The :class:`Scenario` to use for the run.
        - solar_irradiance:
            The solar irradiance profile.
        - solar_thermal_collector:
            The :class:`SolarThermalPanel` to use for the run if appropriate.

    Outputs:

    """

    # Set up variables to track for a valid solution being found.

    # Iterate until a valid solution is found within the hard-coded precision.
    while not (solution_found := False):
        # Calculate the various coefficients which go into the matrix.

        # Solve the matrix.

        # Check whether the solution is valid given the hard-coded precision specified.

        solution_found = True
