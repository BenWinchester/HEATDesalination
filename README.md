# HEATDesalination
Simulation and optimisation of hybrid electric and thermal powered desalination systems

The `HEATDesalination` model provides the ability to simulate, and optimise, energy systems for the provision of both heat and electricity to thermally-driven desalination plants. Case-studies have been conducted for multi-effect distillation (MED) plants, but the software is capable of simulating any plants which require both heat and electricity as inputs.

#### Table Of Contents

:link: [Dependencies](Dependencies)

:clap: [Acknowledgements](Acknowledgements)

[Downloading HEATDesalination](#1.-downloading-heatdesalination)

🐍 [Setting up your Python environment](#setting-up-your-python-environment)
  * [Anaconda method](#anaconda-method)
  * [Pip install](#pip-install)

🌦️ [PVGIS](#pvgis)

:memo: [Completing input files](#completing-input-files)
* [Location-based files](#location-based-files)
* [System-based files](#system-based-files)

:fire: [Running HEATDesalination](#running-heatdesalination)
* [Fetching weather data](#fetching-weather-data)
* [Running a simulation](#running-a-simulation)
* [Running an optimisation](#running-an-optimisation)
* [Parallel simulation and optimisation](#parallel-simulation-and-optimisation)

🎓 [Running HEATDesalination on Imperial College London's high-performance computers](#running-heatdesalination-on-imperial-college-londons-high-performance-computers)

:memo: [References](References)

## Dependencies
This module imports libraries from:
* The open-source `pvlib` Python package developed by Holmgren, Hansen and Mikofski [[1]](#1).

This module integrates with the open-source PVGIS framework, developed by Huld et al., [[2]](#2).

## Acknowledgements
This repository uses code developed by the [CLOVER-energy](https://github.com/CLOVER-energy) team by Winchester et al. [(2022)](#3). Thanks to all at the [@CLOVER-energy/clover-development-team](https://github.com/orgs/CLOVER-energy/teams/clover-development-team) for their work.

## 1. Downloading HEATDesalination

`HEATDesalination` is best installed from [pypi](https://pypi.org/project/heat-desalination/):

```bash
python -m pip install heat-desalination
```

should be run from a terminal or powershell window. This will fetch and install the latest version of `HEATDesalination` along with all of its dependencies. `HEATDesalination` runs best in [Python 3.10](https://www.python.org/downloads/release/python-3100/). 

### Working as a developer

If you wish to help develop and work on the project, or, if you have any modifications that you wish to make to the code, the best approach is to run a git clone of the reposiroty. This will ensure that you have an up-to-date copy of the code which you can use to make changes, push commits and open pull requests within the repository:

```bash
git clone https://github.com/BenWinchester/HEATDesalination
```

#### Setting up your Python environment

`HEATDesalination` uses [Python 3.10](https://www.python.org/downloads/release/python-3100/). If you have installed the package `HEATDesalination` following the instructions in the [Downloading HEATDesalination](#downloading-heatdesalination) section, then you should already have everything that you need. Otherwise, you will need to install the required dependencies.

##### Anaconda method

To install using [`conda`](https://www.anaconda.com/), a Python-based virutal-environment manager, from the root of the repository, run:

```bash
conda install --file requirements.txt
```

**Note**, on some systems, Anaconda may be unable to find the `requirements.txt` file. In these cases, it's necessary to use the absolute path to the file, e.g.,

```bash
conda install --file C:\Users\<User>\...\requirements.txt
```

##### Pip install

If you feel more comfortable using [`pip`](https://pypi.org/project/pip/), the Python package manager, you can use this either from within an Anaconda environment or straight from the command-line:

```bash
python -m pip install -r requirements.txt
```

## 2. PVGIS

`HEATDesalination` relies on the package [`pvlib.iotools.pvgis`](https://pvlib-python.readthedocs.io/en/stable/_modules/pvlib/iotools/pvgis.html) [[1]]. This package is responsible for fetching weather data from the [Photovoltaic Geographical Information System](https://joint-research-centre.ec.europa.eu/pvgis-online-tool_en) (PVGIS). This data is used in the internal models to assess the performance of the solar collectors considered in the system. No API tokens or login keys are needed in order to use this API.

In order to download weather related data for your given location, simply run

```bash
python -m heat-desalination-weather -lat <latitude> -lon -<longitude> -t <timezone>
```

or

```bash
heat-desalination-weather -lat <latitude> -lon -<longitude> -t <timezone>
```

from your command-line interface, provided that you have installed the `HEATDesalination` package, where `<latitude>` and `<longitude>` are floating-point (i.e., decimal) numbers that give the latitude and longitude of the location for which you wish to download data respectively and `<timezone>` is the decimal timezone offset, e.g., `5.5` for a 5-and-a-half hour time difference from UTC.

If you have downloaded the code from Github, you will need to run

```bash
python -m src.heatdesalination.weather -lat <latitude> -lon -<longitude> -t <timezone>
```

from your command-line interface.

## 3. Completing input files

There are several input files which the `HEATDesalination` program requires in order to run. Some of these provide information about the configuration of your specific system whilst others are helpful in telling the program which simulations and optimisations you wish to run. These can broadly be grouped into [location-based files](#location-based-files) and [system-based files](#system-based-files).

### Location-based files

### System-based files

## 4. Running HEATDesalination

### Fetching weather data

### Running a simulation

### Running an optimisation

### Parallel simulation and optimisation

---

## Running HEATDesalination on Imperial College London's high-performance computers

---

## References
<a id="1">[1]</a> 
Holmgren, W. F., Hansen, C. W., & Mikofski, M. A. 2018 "pvlib python: a python package for modeling solar energy systems." Journal of Open Source Software, 3(29), 884. [https://doi.org/10.21105/joss.00884](https://doi.org/10.21105/joss.00884)

<a id="2">[2]</a>
Huld, T., Müller, R. & Gambardella, A., 2012 "A new solar radiation database for estimating PV performance in Europe and Africa". Solar Energy, 86, 1803-1815. [http://dx.doi.org/10.1016/j.solener.2012.03.006](http://dx.doi.org/10.1016/j.solener.2012.03.006)

<a id="3">[3]</a>
Winchester, B., Beath, H., Nelson, J., & Sandwell, P. "CLOVER (Version v5.0.5)" 2020. [Computer software]. [https://doi.org/10.5281/zenodo.6925535](https://doi.org/10.5281/zenodo.6925535)

<a id="4">[4]</a>
Sandwell P., Winchester B., Beath H., & Nelson J. "CLOVER: A modelling framework for sustainable community-scale energy systems." Journal of Open Source Software, 8(82), 4799, 2023. [https://doi.org/10.21105/joss.04799](https://doi.org/10.21105/joss.04799)
