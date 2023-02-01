### Description
Describe the pull request:
* Why are you opening this pull request?
  * Does the pull reuqest resolve an outstanding bug? If so, mark the pull request with the **bug tag**.
  * Does the pull request introduce new features to the model? If so, mark the pull reuqest with the **feature tag**.
* What version of HEATDesalination will this be merged into, and what version will it be updated to? **NOTE:** if you are updating the version of HEATDesalination, please update the various metadata files to reflect this. (These include `setup.cfg`, `.zenodo.json`, `CITATION.cff` and `__main__.py`.)

### Linked Issues
This pull request:
* closes issue 1,
* resolves issue 2,

### Unit tests
This pull request:
* modifies the module unit tests for modules X and Y,
* introduces new component unit tests for the Z component.

### Note
Any other information which is useful for the pull request.

## Requirements
### Reviewers
All pull requests must be approved by an administrator of the repository ([@BenWinchester](https://github.com/BenWinchester)). Make sure to request a review or your pull request will not be approved.

### Checks
HEATDesalination runs a series of automated tests. Make sure to run these prior to opening the pull request. You will not be able to merge your pull request unless all of these automated checks are passing on your code base.
**NOTE:** If you are modifying the automated tests, be sure that you justify this.

The automated tests which are currently run include:
* :art: Code formatting, run with `python -m black src`,
* :green_heart: Mypy type checking, run with `python -m mypy src`,
* :memo: Pylint code linting, run with `python -m pylint src`,
* :white_check_mark: Pytest automated testing, run with `python -m pytest`.

### Metadata files
If you are opening a pull request that will update the version of HEATDesalination, i.e., bring in a new release, then you will need to update the various metadata files as part of your pull request:
* `.zenodo.json` - Update the version number, author list, and date of your proposed release. Add any papers which have been released relevant to HEATDesalination since the last release if relevant;
* `CITATION.cff` - Update the version number, author list, and date of your proposed release. **NOTE:** the date will need to reflect the date on which your pull request is approved;
* `setup.cfg` - Update the version number of HEATDesalniation and include any new files or endpoints required in the `heat-desalination` package:
  * The version is updated under the `version` variable,
  * New packages should be added under the `install_requires` list,
  * New endpoints should be added under the `console_scripts` list;
* `src/heatdesalination/__main__.py` - Update the `__version__` variable name to reflect these changes internally within HEATDeslination.
