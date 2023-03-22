# :two: Design specification

This document contains an overview of the code design required and expected of the software.

#### Table of contents

* [Overview](#overview)
  * [Reviewers](#reviewers)
  * [Modification history](#modification-history)
* [Introduction](#introduction)
* [Nomenclature](#nomenclature)
* [Problem definition](#problem-definition)
* [Design considerations](#design-considerations)
* [Functional structure](#functional-structure)
* [System flow](#system-flow)
  * [Simulation](#simulation)
  * [Optimisation](#optimisation)
* [Data structures](#data-structures)
  * [Internal data structures](#internal-data-structures)
* [Description of algorithms](#description-of-algorithms)
* [Interface design](#interface-design)
* [Restrictions and considerations](#restrictions-and-considerations)
  * [Software](#software)
  * [High-performance computing](#high-performance-computing)
  * [Hardware](#hardware)
* [Source code](#source-code)
* [Development unit testing](#development-unit-testing)
  * [Module unit testing](#module-unit-testing)
  * [Component unit testing](#component-unit-testing)
  * [Integration testing](#integration-testing)
* [References](#references)


## Overview

The `HEATDesalination` package, installed as `heatdesalination`, provides functionality for modelling the energy systems providing heat and electricity to supply thermally-driven desalination plants over the lifetime. This design specification (DS) document aims to outline the design which is expected of the software and of the models contained within it.

### Reviewers

This DS document will be reviewed by maintainers of the repository.

### Modification history

Date | Comments
--- | ---
21/03/2023 | Document created

## Introduction

Demans for electricity [[1]](#1) and clean-water [\[2,](#2)[3\]](#3) continue to rise globally. There is a need to meet clean-water demands in a sustianable way in-line with the United Nations Sustainable Development Goals (SDGs). As the climate changes and the market share of desalination in producing the world's clean-water supply increases in some places, there will be a growing need to carry out desalination using renewably-generated electricity.

Hybrid photovoltaic-thermal (PV-T) collectors are solar panels which are capable of producing both electricity and heat from a single collector, providing synergistic benefits over stand-alone photovoltaic (PV) and solar-thermal systems. As such, they have the potential to provide performance benefits over PV and solar-thermal collectors in powering desalination plants that require both electricity and heat. Despite this, existing thermally-driven deslination plants tend to be powered by a single renewable technology. Similarly, open-source energy-system models tend to consider only a single technology type, with limited scope for considering different manufacturers' data sets and multiple technologies in tandem [\[4,](#4)[5\]](#5).

There is hence a need to develop models capable of simulating, assessing and optimising systems of renewable energy for meeting the needs of desalination plants. The `HEATDesalination` package attempts to do this in an open-source and extensible way.

## Nomenclature

Abbreviation | Definition
--- | ---
HPC | **high-performance computer**, a computer with much greater computing power than standalone desktop computers
PV | **photovoltaic**, used to describe solar panels that can convert sunlight to electricity
PV-T | **photovoltaic-thermal**, used to describe solar panels that can convert sunlight to both heat and electricity from a single collector
SDG | **sustainable-development goal**, one of a series of goals published by the United Nations.
YAML | **yet another markup language**, refers to a type of input file ending with `.yml` or `.yaml`

## Problem definition

The model needs to be able to consider

* A desalination plant, with user-specified
  * start and end times,
  * electricity and hot-water requirements
  * over its lifetime;
* A variety of solar collectors,
  * including PV, PV-T and solar-thermal collectors,
  * across a range of user-specifiable manufacturer values,
  * over their lifetime;
* Including the degradation of components such as
  * the PV and PV-T collectors, which degrade fractionally as time goes on,
  * the battery storage installed, which degrades, amongst other things, based on the number of full cycles that have been carried out,
  * the solar inverters installed, which degrade based on the number of years for which they have been installed;
* And needs to be able to
  * simulate the system over its lifetime,
  * assess the performance of the system against various metrics, including
    * The total cost of the system,
    * The LCUE of energy consumed,
    * The environmental impact of the system, here measured in kg CO $_2$ eq emissions,
  * and to optimise the system.

## Design considerations

The software needs to be

* **Extensible**, so that future features can be added with ease,
* **Modular**, so that each module and component can be easily exported and altered with little impact on the surrounding system,
* Conform to Python coding standards including
  * :art: `black`, the Python formatter,
  * :green_heart: `mypy`, the Python type-checker,
  * :shirt: `pylint`, the Python linter,
  * :shirt: `yamllint`, a linter for YAML inputs files

## Functional structure

The code will be classified into two types of modules: one set which form the main functional flow, and another which mostly contain data structures and small calculations. The schematic of the code flow is shown in [Figure 1](#figure-1).

<figure>
<img src="https://user-images.githubusercontent.com/8342509/226968653-8cb3d7a3-add6-4694-9a78-009676c63671.png">
<figcaption align = "center"><a id="figure-1"><b>Figure 1.</b></a> Schematic of the modular structure of the code. Modules which carry out functionality and calculations are shown in orange, those which primarilly contain data structures with some minor data structures are shown in yellow, with external APIs shown in green.</figcaption>
</figure>

The heat-pump, plant, solar, storage and water-pump modules will contain functionality required to represent heat pumps, desalination plants, solar collectors (whether PV, PV-T or solar-thermal), storage devices (whether electrical or hot-water) and electircally-powered water pumps respectively. These modules will primarily consist of data structures with some small functionality included for assessing performance characteristics of the various components. E.G., the heat-pump module should be able to take in environmental parameters as well as input parameters and output the performance characterstics of the heat pump.

The remaining code will be structured into

* a `__main__.py` module, which will contain the primary entry point for the code,
* an `argparser.py` module, which will deal with command-line argument parsing,
* a `fileparser.py` module, which will be responsible for parsing the various input files,
* and, for running the computations,
  * an `optimiser.py` module, which will carry out optimisations by calling through to
  * the `simulator.py` module, which will carry out simulations of the plant performance, involving
  * matrix calculations carried out by the `matrix.py` module.

There will also be a second entry point for downloading weather data. This will be contained within the `weather.py` module. As this downloading process doesn't need to happen every time that a user runs the `HEATDesalination` program, it makes sense to have a stand-alone flow.

### HPC launch script

The high-performance computing (HPC) launch script will sit above the `__main__.py` module and will contain functionality needed to correctly run the `HEATDesalination` program on Imperial College London's HPCs.

## System flow

The code will contain two primary flows, depending on whether the user is running an optimisation or a simulation.

### Simulation

If the user specifies that a simulation is being run, the code will

1. Parse all of the input files and command-line arguments;
2. Run a simulation of the performance of the plant over its lifetime, iterating until the same start-of-day conditions are achieved after multiple runs to ensure that a steady-state solution has been achieved;
3. Appraise this system and store the results, returning the results as well as the appraisal.

### Optimisation

If the user specifies that an optimisation is being run, the code will

1. Parse all of the input files and command-line arguments;
2. Run am optimisation. This will consist of
    1. Running simulations and appraising the results,
    2. Determining the value of the criterion being optimised,
    3. Aim to minimise or maximise this criterion depending on the requirements of the user;
3. If necessary, carry out simulations that round the various non-integer parameters which must be integers (e.g., the number of solar-thermal collectors) and carry out simulations in this surrounding space and determine the optimum system from this new set;
4. Appraise this optimum system;
5. Save the results of the optimisation and return the results as well as the appraisal of the optimum system.

## Data structures

### Internal data structures

## Description of algorithms

## Interface design

## Restrictions and considerations

### Software

### High-performance computing

### Hardware

## Source code

## Development unit testing

### Module unit testing

### Component unit testing

### Integration testing

***

## References

<a id="1">[1]</a>
Hasanuzzaman, M., Zubir, U. S., Ilham, N. I. & Seng Che, H. Global electricity demand, generation, grid system, and renewable energy polices: a review. Wiley Interdiscipinary Rev. Energy Environ. 6 (3), e222 (2017). URL [https://onlinelibrary.wiley.com/doi/10.1002/wene.222](https://onlinelibrary.wiley.com/doi/10.1002/wene.222). DOI [https://doi.org/10.1002/wene.222](https://onlinelibrary.wiley.com/doi/10.1002/wene.222).

<a id="2">[2]</a>
United Nations Educational, Scientific and Cultural Organization (UNESCO) World Water Assessment Programme. The United Nations World Water Development Report: Valuing Water (UNESCO Publishing, Paris, France, 2021). URL [https://play.google.com/store/books/details?id=WnMnEAAAQBAJ](https://play.google.com/store/books/details?id=WnMnEAAAQBAJ).

<a id="3">[3]</a>
Burek, P. et al. Water futures and solution-fast track initiative. International Institute for Applied Systems Analysis (2016). URL [http://pure.iiasa.ac.at/id/eprint/13008/](http://pure.iiasa.ac.at/id/eprint/13008/).

<a id="4">[4]</a>
Sandwell P., Winchester B., Beath H., & Nelson J. (2023). CLOVER: A modelling framework for sustainable community-scale energy systems. Journal of Open Source Software, 8(82), 4799, DOI [https://doi.org/10.21105/joss.04799](https://doi.org/10.21105/joss.04799).

<a id="5">[5]</a>
Winchester, B., Beath, H., Nelson, J., & Sandwell, P. (2023). CLOVER (Version v5.0.7) [Computer software]. DOI [https://doi.org/10.5281/zenodo.6925535](https://doi.org/10.5281/zenodo.6925535).
