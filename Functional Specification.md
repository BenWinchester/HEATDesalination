# :one: Functional Specification

This document contains an overview of the functionality required and expected of the `HEATDesalination` package.

#### Table of contents

* [Overview](#overview)
  * [Modification history](#modification-history)
* [Use cases](#use-cases)
* [Requirements](#requirements)
  * [Interface requirements](#interface-requirements)
  * [Package requirements](#package-requirements)
  * [Logging requirements](#logging-requirements)
* [Configuration](#configuration)
* [Non-functional requirements](#non-functional-requirements)
* [Error reporting](#error-reporting)

## Overview

This functional specification (FS) document aims to outline the functionality which is expected of the `HEATDesalination` package.

### Modification history

Date | Comments
--- | ---
21/03/2023 | Document created

## Use cases

A user should be able to
* Simulate the performance of a given energy system for meeting the needs of a thermally- and electrically-driven desalination plant over its lifetime;
* Optimise the capacity of the energy-generation system over the lifetime of the plant given some target criterion.

## Requirements

The code should be

* Written in [![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/);
* Open-access;
* Use a limited number of external packages where possible to widen the usability of the package;
* Conform to coding standards using
  * :art: `black`, the Python formatter;
  * :green_heart: `mypy`, the Python type-checker;
  * :shirt: `pylint`, the Python linter.

### Interface requirements

The `HEATDesalination` package should expose a command-line interface which is capable of
* Downloading and fetching weather data for a specified location (set of coordinates) for use within the model.
* Simulating a plant, taking in input information required to determine which of the input files are to be used, which scenario to use, which weather-data file to use etc.;
* Optimising the energy-generation system of a plant over its lifetime.

### Package requirements

The Python package should
* Expose all classes and structures where applicable, along with all methods where applicable, so that the model can be best integrated with other models.

### Logging requirements
Sufficient logging should take place throughout to report on
* :white_circle: `DEBUG` calls, which should provide useful information, if requested, on the internal workings of the code;
* :large_blue_circle: `INFO` calls, which should provide useful information on the flow of the code from which the flow of the code, and which sections ran, can be determined;
* :red_circle: `WARNING` calls which inform the user that the software has:
  * utilised a deprecated piece of code,
  * carried out an operation which is not recommended,
  * made an assumption based on incomplete input information;
* :black_circle: `ERROR` calls which inform the user that an error has occurred and that the flow of the code has stopped. They should contain as much information as is deemed necessary to diagnose and rectify the issue.

## Configuration

Configuration information will be needed to determine

* The desalination plants available within the model;
* The heat pumps available for use by the model;
* The scenarios that can be selected;
* The solar collectors, whether solar-thermal, PV-T or PV in nature, which can be selected;
* The storage components, whether electrical or hot-water storage devices, available for selection.
* The optimisations that should be carried out;
* The water pumps available within the model.

For each of these components, performance and technical requirements will be needed, as well as costs and emissions information.

## Non-functional requirements

The package will contain
* Module-unit tests (MUTs) which carry out automated tests of the functionality of the various modules contained within the code;
* Component-unit tests (CUTs) which carry out component-level tests of the functionality of the various components contained within the code;
* Integration tests (ITs) which will test
  * Simulation of the plant over its lifetime;
  * Optimisation of the plant over its lifetime.

## Error reporting

If errors occur in the code, the software will, where possible, continue until the error is critical and will result in the cessation of the code. At this point, an error will be raised containing as much information as is possible and is necessary for the user to determine where the error occurred, which input file/line of code was responsible for the error, and how best to rectify the issue.
