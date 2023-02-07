#!/usr/bin/python3.10
########################################################################################
# weather.py - The weather module                                                      #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 17/10/2022                                                             #
# License: MIT, Open-source                                                            #
# For more information, contact: benedict.winchester@gmail.com                         #
########################################################################################

"""
weather.py - The weather module for the HEATDeslination program.

This module is responsible for parsing weather information from online interfaces,
producing average results, as well as optimum tilt angles, and determine upper and lower
bounds for the irradiance at a given location.

"""

import argparse
import functools
import os
import sys

from logging import Logger
from typing import Any, Tuple

import json
import numpy as np
import pandas as pd

from geopy import geocoders
from pvlib.iotools import pvgis
from requests import HTTPError

from .__utils__ import (
    AMBIENT_TEMPERATURE,
    AUTO_GENERATED_FILES_DIRECTORY,
    DONE,
    get_logger,
    LATITUDE,
    LONGITUDE,
    ProfileType,
    OPTIMUM_TILT_ANGLE,
    SOLAR_IRRADIANCE,
    WIND_SPEED,
)

# ADDRESS:
#   String used for parsing address information from latitude and longitude.
ADDRESS: str = "address"

# CITY:
#   String used for parsing city name information.
CITY: str = "city"

# COUNTRY:
#   String used for parsing country name information.
COUNTRY: str = "country"

# COUNTY:
#   String used for parsing county name information.
COUNTY: str = "county"

# FIXED:
#   Keyword for the fixed-panel mounting systems.
FIXED: str = "fixed"

# IRRADIANCE_COLUMN_NAME:
#   Column name to use for irradiance data from parsed data set.
IRRADIANCE_COLUMN_NAME: str = "poa_global"

# LANGUAGE:
#   String used to determine language for display city and country name information.
LANGUAGE: str = "en"

# LOGGER_NAME:
#   The name to use for the log file.
LOGGER_NAME: str = "weather"

# MOUNTING_SYSTEM:
#   Keyword for the mounting system of the panel.
MOUNTING_SYSTEM: str = "mounting_system"

# PVGIS_ERROR:
#   The fractional error in PVGIS data.
#   Milosavljević DD, Kevkić TS, Jovanović SJ. Review and validation of photovoltaic
# solar simulation tools/software based on case study. Open Physics 2022;20:431–51.
PVGIS_ERROR: float = 0.181

# SLOPE:
#   Keyword for the slope of the panel mounting system.
SLOPE: str = "slope"

# STATE:
#   String used for parsing state name information.
STATE: str = "state"

# STATE_DISTRICT:
#   String used for parsing state-district name information.
STATE_DISTRICT: str = "state_district"

# VALUE:
#   Keyword for the value of the slope of the panel mounting system.
VALUE: str = "value"

# WEATHER_COLUMN_HEADERS:
#   Column headers to use for the weather profiles.
WEATHER_COLUMN_HEADERS: pd.Index = pd.Index(
    [SOLAR_IRRADIANCE, "solar_elevation", AMBIENT_TEMPERATURE, WIND_SPEED, "Int"]
)


def _get_location_name(
    latitude: float, longitude: float, *, logger: Logger
) -> Tuple[str, str]:
    """
    Determines the name of the location for sanity checking.

    Inputs:
        - latitude:
            The latitude of the location
        - longitude:
            The longitude of the location.
        - logger:
            The logger to use for the run.

    Outputs:
        - location_name:
            A user-readable location name for logging and printing.
        - output_name:
            The name of the output file to use for the location if not specified.

    """

    # Initialize Nominatim API
    geolocator = geocoders.Nominatim(user_agent="geoapiExercises")

    # Determine the location
    address = geolocator.reverse(
        ",".join([str(latitude), str(longitude)]), language=LANGUAGE
    ).raw[ADDRESS]

    # Attempt to determine the city/precise information.
    if CITY in address:
        accurate_name: str = address[CITY]
    elif COUNTY in address:
        accurate_name = address[COUNTY]
    elif STATE_DISTRICT in address and STATE in address:
        accurate_name = ", ".join([address[STATE_DISTRICT], address[STATE]])
    elif STATE in address:
        accurate_name = address[STATE]
    else:
        logger.info("Could not determine precise name of location.")
        accurate_name = None

    # Generate user-readable strings and output names
    location_name = ", ".join(
        [accurate_name if accurate_name is not None else "NaN", address[COUNTRY]]
    )
    output_name = "_".join(
        [
            accurate_name.lower().replace(", ", "_").replace(" ", "_")
            if accurate_name is not None
            else "NaN",
            address[COUNTRY].lower().replace(" ", "_"),
        ]
    )

    return location_name, output_name


def _parse_args(args: list[Any]) -> argparse.Namespace:
    """
    Parses command-line arguments.

    Inputs:
        - args:
            The unparsed command-line arguments used, if any.

    Outputs:
        The parsed command-line arguments.

    """

    parser = argparse.ArgumentParser()

    required_arguments = parser.add_argument_group("required arguments")

    # Latitude:
    #   The latitude of the location.
    required_arguments.add_argument(
        "--latitude",
        "-lat",
        help="The latitude of the location.",
        type=float,
    )

    # Longitude:
    #   The longitude of the location.
    required_arguments.add_argument(
        "--longitude",
        "-lon",
        help="The longitude of the location.",
        type=float,
    )

    # Outputs:
    #   The name of the location being fetched, used to save the output file.
    parser.add_argument(
        "--output",
        "-o",
        help="The name of the location being parsed, used for saving the output.",
        type=str,
    )

    # Longitude:
    #   The longitude of the location.
    required_arguments.add_argument(
        "--timezone",
        "-t",
        help="The timezone in hours relative to UTC for the location.",
        type=float,
    )

    return parser.parse_args(args)


def main(
    latitude: float, longitude: float, timezone: int, output: str | None = None
) -> None:
    """
    The main method for the weather module.

    Inputs:
        - latitude:
            The latitude of the location.
        - longitude:
            The longitude of the location.
        - timezone:
            The UTC timezone offset.
        - output:
            The name of the output file to save the data to.

    """

    # Instantiate a logger.
    logger = get_logger(LOGGER_NAME)
    logger.info(
        "Weather module instantiated for location %sN %sE.", latitude, longitude
    )

    # Determine the location using geopy module.
    location_name, output_name = _get_location_name(latitude, longitude, logger=logger)
    logger.info("Location determined: %s", location_name)
    print(
        f"Fetching weather information for {location_name} ({latitude:.2g}N, {longitude:.2g}E)"
    )

    # Call PVGIS to get the weather data for the location
    try:
        parsed_data = pvgis.get_pvgis_hourly(
            latitude,
            longitude,
            components=False,
            optimal_surface_tilt=True,
        )
    except (TypeError, ValueError):
        logger.error(
            "PVGIS did not return an output. Check your internet connection and that "
            "you supplied valid latitude and longitude parameters."
        )
        raise
    except HTTPError:
        logger.error("Location selected is over the sea.")
        raise
    logger.info(
        "Weather data successfully fetched for %s at %sN %sE.",
        location_name,
        latitude,
        longitude,
    )
    logger.info(
        "Weather data:\n%s\n%s\n%s",
        parsed_data[0],
        json.dumps(parsed_data[1], indent=4),
        json.dumps(parsed_data[2], indent=4),
    )
    print(
        f"Weather information successfully fetched, parsing{'.'*20} ",
        end="",
    )

    # Create a map between day and profiles chunked
    weather_data = parsed_data[0]
    daily_weather_profiles: dict[int, pd.DataFrame] = {
        day: weather_data[day * 24 : (day + 1) * 24]
        for day in range(int(len(weather_data) / 24))
    }

    # Create a map between day and cumulative irradiance for the day
    cumulative_irradiance_to_day: dict[int, float] = {
        np.sum(irradiance_profile[IRRADIANCE_COLUMN_NAME]): day
        for day, irradiance_profile in daily_weather_profiles.items()
    }

    # Reverse the cumulative irradiance map to determine the max and min irradiance days
    # max_day = cumulative_irradiance_to_day[max(cumulative_irradiance_to_day)]
    # max_profile = daily_weather_profiles[max_day]
    #
    # min_day = cumulative_irradiance_to_day[min(cumulative_irradiance_to_day)]
    # min_profile = daily_weather_profiles[min_day]
    #

    # Determine the average profiles.
    unindexed_profiles = [
        profile.reset_index(drop=True) for profile in daily_weather_profiles.values()
    ]
    average_profile = pd.DataFrame(
        functools.reduce(lambda x, y: x.add(y, fill_value=0), unindexed_profiles)
    ) / len(unindexed_profiles)

    # Determine the standard deviation.
    standard_deviation = pd.DataFrame(
        np.sqrt(
            functools.reduce(
                lambda x, y: x.add((y - average_profile) ** 2, fill_value=0),
                unindexed_profiles,
            )
            / len(unindexed_profiles)
        )
    )

    # Determine the maximum value at each hour of the profiles and the minimum value.
    concatenated_frame = pd.concat(
        unindexed_profiles, keys=range(len(unindexed_profiles))
    ).groupby(level=1)
    max_profile = concatenated_frame.max()
    min_profile = concatenated_frame.min()

    upper_std_profile = average_profile + standard_deviation
    lower_std_profile = average_profile - standard_deviation

    # Ensure that no standard deviations are out of bounds.
    upper_std_profile = (
        pd.concat([upper_std_profile, max_profile], keys=[0, 1]).groupby(level=1).min()
    )
    lower_std_profile = (
        pd.concat([lower_std_profile, min_profile], keys=[0, 1]).groupby(level=1).max()
    )

    # Compute weather-based error bars.
    upper_error_bar_profile = average_profile * (1 + PVGIS_ERROR)
    lower_error_bar_profile = average_profile * (1 - PVGIS_ERROR)

    # Set the column headers correctly
    average_profile.columns = WEATHER_COLUMN_HEADERS
    lower_error_bar_profile.columns = WEATHER_COLUMN_HEADERS
    lower_std_profile.columns = WEATHER_COLUMN_HEADERS
    max_profile.columns = WEATHER_COLUMN_HEADERS
    min_profile.columns = WEATHER_COLUMN_HEADERS
    upper_error_bar_profile.columns = WEATHER_COLUMN_HEADERS
    upper_std_profile.columns = WEATHER_COLUMN_HEADERS
    average_profile_dict = average_profile.to_dict()

    print(DONE)
    print(f"Saving weather data output{'.'*43} ", end="")

    # Generate the output data structure.
    output_data = {
        ProfileType.AVERAGE.value: {
            key: {int(time + timezone) % 24: value for time, value in entry.items()}
            for key, entry in average_profile_dict.items()
        },
        LATITUDE: latitude,
        LONGITUDE: longitude,
        ProfileType.LOWER_ERROR_BAR.value: {
            key: {int(time + timezone) % 24: value for time, value in entry.items()}
            for key, entry in lower_error_bar_profile.to_dict().items()
        },
        ProfileType.LOWER_STANDARD_DEVIATION.value: {
            key: {int(time + timezone) % 24: value for time, value in entry.items()}
            for key, entry in lower_std_profile.to_dict().items()
        },
        ProfileType.MAXIMUM.value: {
            key: {int(time + timezone) % 24: value for time, value in entry.items()}
            for key, entry in max_profile.to_dict().items()
        },
        ProfileType.MINIMUM.value: {
            key: {int(time + timezone) % 24: value for time, value in entry.items()}
            for key, entry in min_profile.to_dict().items()
        },
        OPTIMUM_TILT_ANGLE: (
            optimum_tilt_angle := parsed_data[1][MOUNTING_SYSTEM][FIXED][SLOPE][VALUE]
        ),
        ProfileType.UPPER_ERROR_BAR.value: {
            key: {int(time + timezone) % 24: value for time, value in entry.items()}
            for key, entry in upper_error_bar_profile.to_dict().items()
        },
        ProfileType.UPPER_STANDARD_DEVIATION.value: {
            key: {int(time + timezone) % 24: value for time, value in entry.items()}
            for key, entry in upper_std_profile.to_dict().items()
        },
    }

    # Return and save information on the weather conditions for an average day and the
    # max and min irradiance days as well as the optimum tilt angle and the latitude and
    # longitude.
    output_name = output if output is not None else output_name
    os.makedirs(AUTO_GENERATED_FILES_DIRECTORY, exist_ok=True)
    with open(
        filepath := f"{os.path.join(AUTO_GENERATED_FILES_DIRECTORY, output_name)}.json",
        "w",
        encoding="UTF-8",
    ) as output_file:
        json.dump(output_data, output_file, indent=4)

    logger.info("Output data successfully saved to %s.", filepath)
    print(DONE)
    print(f"Output data successfully saved to {filepath}.")
    print(f"Optimum tilt angle for {location_name} is {optimum_tilt_angle} degrees.")


if __name__ == "__main__":
    # Parse the arguments.
    parsed_args = _parse_args(sys.argv[1:])

    # Raise an error if the latitude and longitude weren't provif
    if parsed_args.latitude is None:
        raise Exception("Latitude must be specified.")
    if parsed_args.longitude is None:
        raise Exception("Longitude must be specified.")
    if parsed_args.timezone is None:
        raise Exception("Timezone must be specified.")

    # Call the main function.
    main(
        parsed_args.latitude,
        parsed_args.longitude,
        parsed_args.timezone,
        parsed_args.output,
    )
