#!/usr/bin/python3
########################################################################################
# storage_utils.py - Storage utility module.                                           #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 18/01/2022                                                             #
# License: MIT                                                                         #
# Most recent update: 19/10/2022                                                       #
########################################################################################
"""
storage_utils.py - The storage utility module for HEATDesalination, taken from CLOVER.

HEATDesalination considers several storage media for various forms of energy. These are
all contained and considered within this module.

The code for this module is taken from CLOVER-energy/CLOVER and is reproduced with
permission from the authors under the open-source MIT license.

"""

import dataclasses

from typing import Any

from ..__utils__ import (
    AREA,
    COST,
    HEAT_CAPACITY_OF_WATER,
    NAME,
    CostableComponent,
    ResourceType,
)

__all__ = (
    "Battery",
    "CleanWaterTank",
    "HotWaterTank",
)

# C_RATE_CHARGING:
#   Keyword used for parsing the charging c rate.
C_RATE_CHARGING: str = "c_rate_charging"

# C_RATE_DISCHARGING:
#   Keyword used for parsing the discharging c rate.
C_RATE_DISCHARGING: str = "c_rate_discharging"

# CAPACITY:
#   Keyword used for parsing the capacity.
CAPACITY: str = "capacity"

# CONVERSION_IN:
#   Keyword used for parsing the conversion input efficiency.
CONVERSION_IN: str = "conversion_in"

# CONVERSION_OUT:
#   Keyword used for parsing the conversion output efficiency.
CONVERSION_OUT: str = "conversion_out"

# CYCLE_LIFETIME:
#   Keyword used for parsing the cycle lifetime.
CYCLE_LIFETIME: str = "cycle_lifetime"

# HEAT_CAPACITY:
#   Keyword used for parsing heat capacity.
HEAT_CAPACITY: str = "heat_capacity"

# HEAT_LOSS_COEFFICIENT:
#   Keyword used for parsing heat-loss coefficient.
HEAT_LOSS_COEFFICIENT: str = "heat_loss_coefficient"

# LEAKAGE:
#   Keyword used for parsing leakage.
LEAKAGE: str = "leakage"

# LIFETIME_LOSS:
#   Keyword used for parsing lifetime loss from the storage.
LIFETIME_LOSS: str = "lifetime_loss"

# MAXIMUM_CHARGE:
#   Keyword used for parsing maximum charge.
MAXIMUM_CHARGE: str = "maximum_charge"

# MINIMUM_CHARGE:
#   Keyword used for parsing minimum charge.
MINIMUM_CHARGE: str = "minimum_charge"


class _BaseStorage(CostableComponent):
    """
    Repsesents an abstract base storage unit.

    .. attribute:: capacity
        The capacity of the :class:`_BaseStorage` unit.

    .. attribute:: cycle_lifetime
        The number of cycles for which the :class:`_BaseStorage` can perform.

    .. attribute:: label
        The label given to the :class:`_BaseStorage` instance.

    .. attribute:: leakage
        The rate of level leakage from the :class:`_BaseStorage`.

    .. attribute:: maximum_charge
        The maximum level of the :class:`_BaseStorage`, defined between 0 (able to hold
        no charge) and 1 (able to fully charge).

    .. attribute:: minimum_charge
        The minimum level of the :class:`_BaseStorage`, defined between 0 (able to fully
        discharge) and 1 (unable to discharge any amount).

    .. attribute:: name
        A unique name for identifying the :class:`_BaseStorage`.

    .. attribute:: resource_type
        The type of resource being stored by the :class:`_BaseStorage` instance.

    """

    label: str
    resource_type: ResourceType

    def __init__(
        self,
        capacity: float,
        cost: float,
        cycle_lifetime: int,
        leakage: float,
        maximum_charge: float,
        minimum_charge: float,
        name: str,
    ) -> None:
        """
        Instantiate a :class:`Storage` instance.

        Inputs:
            - capacity:
                The capacity which the :class:`_BaseStorage` can hold, measured in the
                appropriate unit for the resource being stored.
            - cost:
                The cost of the :class:`_BaseStorage` instance.
            - cycle_lifetime:
                The number of cycles for which the :class:`_BaseStorage` instance can
                perform.
            - leakage:
                The rate of leakage from the storage.
            - maximum_charge:
                The maximum level that can be held by the :class:`_BaseStorage`.
            - minimum_charge:
                The minimum level to which the :class:`_BaseStorage` instance can
                discharge.
            - name:
                The name to assign to the :class:`Storage` instance.

        """

        self.capacity: float = capacity
        self.cycle_lifetime: int = cycle_lifetime
        self.leakage: float = leakage
        self.maximum_charge: float = maximum_charge
        self.minimum_charge: float = minimum_charge
        self.name: str = name

        super().__init__(cost)

    def __hash__(self) -> int:
        """
        Return a unique hash identifying the :class:`_BaseStorage` instance.

        Outputs:
            - Return a unique hash identifying the :class:`_BaseStorage` instance.

        """

        return hash(self.name)

    def __init_subclass__(cls, label: str, resource_type: ResourceType) -> None:
        """
        Method run when a :class:`_BaseStorage` child is instantiated.

        Inputs:
            - label:
                A `str` that identifies the class type.
            - resource_type:
                The type of load being modelled.

        """

        super().__init_subclass__()
        cls.label = label
        cls.resource_type = resource_type

    def __str__(self) -> str:
        """
        Returns a nice-looking string describing the :class:`_BaseStorage` instance.

        Outputs:
            - A `str` giving information about the :class:`_BaseStorage` instance.

        """

        return (
            "Storage("
            + f"name={self.name}, "
            + f"capacity={self.capacity} (units), "
            + f"cycle_lifetime={self.cycle_lifetime} cycles, "
            + f"leakage={self.leakage}, "
            + f"maximum_charge={self.maximum_charge}, "
            + f"minimum_charge={self.minimum_charge}"
            + ")"
        )

    @classmethod
    def from_dict(cls, storage_data: dict[str, Any]) -> Any:
        """
        Create a :class:`CleanWaterTank` instance based on the file data passed in.

        Inputs:
            - storage_data:
                The tank data, extracted from the relevant input file.

        Outputs:
            - A :class:`CleanWaterTank` instance.

        """

        return cls(
            storage_data[CAPACITY],
            storage_data[COST],
            storage_data[CYCLE_LIFETIME],
            storage_data[LEAKAGE],
            storage_data[MAXIMUM_CHARGE],
            storage_data[MINIMUM_CHARGE],
            storage_data[NAME],
        )


@dataclasses.dataclass
class Battery(_BaseStorage, label="battery", resource_type=ResourceType.ELECTRICITY):
    """
    Represents a battery.

    .. attribute:: c_rate_charging
        The rate of charge of the :class:`Battery`.

    .. attribute:: c_rate_discharging
        The rate of discharge of the :class:`Battery`.

    .. attribute:: conversion_in
        The input conversion efficiency of the :class:`Battery`.

    .. attribute:: conversion_out
        The output conversion efficiency of the :class:`Battery`.

    .. attribute:: lifetime_loss
        The overall loss in capacity of the :class:`Battery` over its lifetime.

    """

    def __init__(
        self,
        capacity: float,
        cost: float,
        cycle_lifetime: int,
        leakage: float,
        maximum_charge: float,
        minimum_charge: float,
        name: str,
        c_rate_charging: float,
        c_rate_discharging: float,
        conversion_in: float,
        conversion_out: float,
        lifetime_loss: float,
    ) -> None:
        """
        Instantiate a :class:`Battery` instance.

        Inputs:
            - capacity:
                The capacity of the battery in kWh.
            - cost:
                The cost of the :class:`Battery` instance.
            - cycle_lifetime:
                The number of cycles for which the :class:`Battery` instance can
                perform.
            - leakage:
                The rate of leakage from the storage.
            - maximum_charge:
                The maximum level that can be held by the :class:`Battery`.
            - minimum_charge:
                The minimum level to which the :class:`Battery` instance can
                discharge.
            - name:
                The name to assign to the :class:`Battery` instance.
            - c_rate_charging:
                The rate of charge of the :class:`Battery`.
            - c_rate_discharging:
                The rate of discharge of the :class:`Battery`.
            - conversion_in:
                The efficiency of conversion of energy into the :class:`Battery`.
            - conversion_out:
                The efficiency of conversion of energy out of the :class:`Battery`.
            - lifetime_loss:
                The loss in capacity of the :class:`Battery` over its lifetime.

        """

        super().__init__(
            capacity,
            cost,
            cycle_lifetime,
            leakage,
            maximum_charge,
            minimum_charge,
            name,
        )
        self.c_rate_charging: float = c_rate_charging
        self.c_rate_discharging: float = c_rate_discharging
        self.conversion_in: float = conversion_in
        self.conversion_out: float = conversion_out
        self.lifetime_loss: float = lifetime_loss

    def __hash__(self) -> int:
        """
        Return a unique hash identifying the :class:`_BaseStorage` instance.

        Outputs:
            - Return a unique hash identifying the :class:`_BaseStorage` instance.

        """

        return hash(self.name)

    def __str__(self) -> str:
        """
        Returns a nice-looking string describing the :class:`_BaseStorage` instance.

        Outputs:
            - A `str` giving information about the :class:`_BaseStorage` instance.

        """

        return (
            "Battery("
            + f"{self.label} storing {self.resource_type.value} loads, "
            + f"name={self.name}, "
            + f"capacity={self.capacity}, "
            + f"cycle_lifetime={self.cycle_lifetime} cycles, "
            + f"leakage={self.leakage}, "
            + f"maximum_charge={self.maximum_charge}, "
            + f"minimum_charge={self.minimum_charge}, "
            + f"c_rate_charging={self.c_rate_charging}, "
            + f"c_rate_discharging={self.c_rate_discharging}, "
            + f"conversion_in={self.conversion_in}, "
            + f"conversion_out={self.conversion_out}, "
            + f"lifetime_loss={self.lifetime_loss}, "
            + ")"
        )

    def __repr__(self) -> str:
        """
        Defines the default representation of the :class:`Battery` instance.

        Outputs:
            - A `str` giving the default representation of the :class:`Battery`
              instance.

        """

        return (
            "Battery("
            + f"{self.label} storing {self.resource_type.value} loads, "
            + f"name={self.name}, capacity={self.capacity}"
            + ")"
        )

    @classmethod
    def from_dict(cls, storage_data: dict[str, Any]) -> Any:
        """
        Create a :class:`Battery` instance based on the file data passed in.

        Inputs:
            - storage_data:
                The battery data, extracted from the relevant input file.

        Outputs:
            - A :class:`Battery` instance.

        """

        return cls(
            storage_data[CAPACITY],
            storage_data[COST],
            storage_data[CYCLE_LIFETIME],
            storage_data[LEAKAGE],
            storage_data[MAXIMUM_CHARGE],
            storage_data[MINIMUM_CHARGE],
            storage_data[NAME],
            storage_data[C_RATE_CHARGING],
            storage_data[C_RATE_DISCHARGING],
            storage_data[CONVERSION_IN],
            storage_data[CONVERSION_OUT],
            storage_data[LIFETIME_LOSS],
        )


@dataclasses.dataclass
class CleanWaterTank(
    _BaseStorage, label="clean_water_tank", resource_type=ResourceType.CLEAN_WATER
):
    """Represents a clean-water tank."""

    def __str__(self) -> str:
        """
        Returns a nice-looking string describing the :class:`CleanWaterTank` instance.

        Outputs:
            - A `str` giving information about the :class:`CleanWaterTank` instance.

        """

        return (
            "CleanWaterTank("
            + f"{self.label} storing {self.resource_type.value} loads, "
            + f"name={self.name}, "
            + f"capacity={self.capacity} litres, "
            + f"cycle_lifetime={self.cycle_lifetime} cycles, "
            + f"leakage={self.leakage}, "
            + f"maximum_charge={self.maximum_charge}, "
            + f"minimum_charge={self.minimum_charge}"
            + ")"
        )

    def __repr__(self) -> str:
        """
        Defines the default representation of the :class:`CleanWaterTank` instance.

        Outputs:
            - A `str` giving the default representation of the :class:`CleanWaterTank`
              instance.

        """

        return (
            "CleanWaterTank("
            + f"{self.label} storing {self.resource_type.value} loads, "
            + f"name={self.name}"
            + ")"
        )


@dataclasses.dataclass
class HotWaterTank(
    _BaseStorage, label="hot_water_tank", resource_type=ResourceType.HOT_WATER
):
    """
    Represents a hot-water tank.

    .. attribute:: area
        The area of the hot-water tank, measured in meters squared.

    .. attribute:: heat_capacity
        The specific heat capacity of the contents of the tank, measured in Joules per
        kilogram Kelvin, defaults to that of water at stp.

    .. attribute:: heat_loss_coefficient
        The heat loss from the tank, measured in Watts per meter squared per Kelvin.

    .. attribute:: heat_transfer_coefficient
        The heat transfer coefficient from the tank to its surroundings, measured in
        Watts per Kelvin.

    """

    def __init__(
        self,
        capacity: int,
        cost: float,
        cycle_lifetime: int,
        leakage: float,
        maximum_charge: float,
        minimum_charge: float,
        name: str,
        area: float,
        heat_capacity: float,
        heat_loss_coefficient: float,
    ) -> None:
        """
        Instantiate a :class:`CleanWaterTank`.

        Inputs:
            - capacity:
                The capacity of the :class:`HotWaterTank`.
            - cost:
                The cost of the :class:`HotWaterTank` instance.
            - cycle_lifetime:
                The number of cycles for which the :class:`HotWaterTank` instance can
                perform.
            - leakage:
                The rate of leakage from the storage.
            - maximum_charge:
                The maximum level that can be held by the :class:`HotWaterTank`.
            - minimum_charge:
                The minimum level to which the :class:`HotWaterTank` instance can
                discharge.
            - name:
                The name to assign to the :class:`HotWaterTank` instance.
            - mass:
                The mass of water that can be held in the clean-water tank.
            - area:
                The surface area of the tank.
            - heat_capacity:
                The specific heat capacity of the contents of the :class:`HotWaterTank`.
            - heta_loss_coefficient:
                The heat-loss coefficient for the :class:`HotWaterTank`.

        """

        super().__init__(
            capacity,
            cost,
            cycle_lifetime,
            leakage,
            maximum_charge,
            minimum_charge,
            name,
        )
        self.area = area
        self.heat_capacity = heat_capacity
        self.heat_loss_coefficient = heat_loss_coefficient

    def __hash__(self) -> int:
        """
        Return a unique hash identifying the :class:`_BaseStorage` instance.

        Outputs:
            - Return a unique hash identifying the :class:`_BaseStorage` instance.

        """

        return hash(self.name)

    @property
    def heat_transfer_coefficient(self) -> float:
        """
        Return the heat-transfer coefficient from the :class:`HotWaterTank`.

        Outputs:
            - The heat-transfer coefficient from the :class:`HotWaterTank` to its
              surroundings, measured in Watts per Kelvin.

        """

        return self.heat_loss_coefficient * self.area

    def __str__(self) -> str:
        """
        Returns a nice-looking string describing the :class:`HotWaterTank` instance.

        Outputs:
            - A `str` giving information about the :class:`HotWaterTank` instance.

        """

        return (
            "HotWaterTank("
            + f"{self.label} storing {self.resource_type.value} loads, "
            + f"name={self.name}, "
            + f"area={self.area} m^2, "
            + f"capacity={self.capacity} litres, "
            + f"cost={self.cost} $/unit, "
            + f"cycle_lifetime={self.cycle_lifetime} cycles, "
            + f"heat_capacity={self.heat_capacity} J/kg*K, "
            + f"heat_loss_coefficient={self.heat_loss_coefficient} W/m^2K, "
            + f"heat_transfer_coefficient={self.heat_transfer_coefficient} W/K, "
            + f"leakage={self.leakage}, "
            + f"maximum_charge={self.maximum_charge}, "
            + f"minimum_charge={self.minimum_charge}"
            + ")"
        )

    def __repr__(self) -> str:
        """
        Defines the default representation of the :class:`HotWaterTank` instance.

        Outputs:
            - A `str` giving the default representation of the :class:`HotWaterTank`
              instance.

        """

        return (
            "HotWaterTank("
            + f"{self.label} storing {self.resource_type.value} loads, "
            + f"name={self.name}"
            + ")"
        )

    @classmethod
    def from_dict(cls, storage_data: dict[str, Any]) -> Any:
        """
        Create a :class:`HotWaterTank` instance based on the file data passed in.

        Inputs:
            - storage_data:
                The tank data, extracted from the relevant input file.

        Outputs:
            - A :class:`HotWaterTank` instance.

        """

        return cls(
            storage_data[CAPACITY],
            storage_data[COST],
            storage_data[CYCLE_LIFETIME],
            storage_data[LEAKAGE],
            storage_data[MAXIMUM_CHARGE],
            storage_data[MINIMUM_CHARGE],
            storage_data[NAME],
            storage_data[AREA],
            storage_data[HEAT_CAPACITY]
            if HEAT_CAPACITY in storage_data
            else HEAT_CAPACITY_OF_WATER,
            storage_data[HEAT_LOSS_COEFFICIENT],
        )

    @property
    def mass(self) -> float:
        """
        Return the mass of the tank.

        Outputs:
            The mass of the hot-water tank.

        """

        return self.capacity
