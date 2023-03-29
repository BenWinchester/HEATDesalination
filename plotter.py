interact
import matplotlib.pyplot as plt

ALPHA = 0.3
fig, ax = plt.subplots()

solution = simulation_outputs

# Plot envelopes
ax.plot(
    solution.electricity_demands.keys(),
    solution.electricity_demands.values(),
    "--",
    label="demand",
    color="C3",
)
ax.plot(
    solution.total_collector_electrical_output_power["degraded"].keys(),
    solution.total_collector_electrical_output_power["degraded"].values(),
    "--",
    label="solar production",
    color="C1",
)
ax.plot(
    solution.battery_storage_profile.keys(),
    solution.battery_storage_profile.values(),
    "--",
    label="battery state of charge",
    color="C0",
)

###################
# Plot solar bars #
###################
ax.bar(
    solution.solar_power_supplied.keys(),
    solution.solar_power_supplied.values(),
    alpha=ALPHA,
    label="solar power supplied",
    color="C1",
)
bottom = list(solution.solar_power_supplied.values())

ax.bar(
    solution.battery_power_input_profile.keys(),
    solution.battery_power_input_profile.values(),
    alpha=0.6,
    label="power to batteries",
    bottom=bottom,
    color="C1",
)
bottom = [
    entry + solution.battery_power_input_profile[index]
    for index, entry in enumerate(bottom)
]

ax.bar(
    solution.dumped_solar.keys(),
    solution.dumped_solar.values(),
    alpha=0.3,
    label="dumped solar",
    bottom=bottom,
    color="C1",
)
bottom = [entry + solution.dumped_solar[index] for index, entry in enumerate(bottom)]

#####################
# Plot battery bars #
#####################
ax.bar(
    solution.battery_electricity_suppy_profile.keys(),
    solution.battery_electricity_suppy_profile.values(),
    alpha=ALPHA,
    label="storage power supplied",
    bottom=bottom,
    color="C0",
)
bottom = [
    entry + solution.battery_electricity_suppy_profile[index]
    for index, entry in enumerate(bottom)
]

##################
# Plot grid bars #
##################

ax.bar(
    solution.grid_electricity_supply_profile.keys(),
    solution.grid_electricity_supply_profile.values(),
    alpha=ALPHA,
    label="grid power supplied",
    bottom=bottom,
    color="C2",
)

plt.xlabel("Hour of day")
plt.ylabel("Average hourly power flow / kWh")

ax.legend(bbox_to_anchor=(1.0, 1.0))
plt.show()

###################
# Ad-hoc plotting #
###################

import json
import matplotlib.pyplot as plt
import seaborn as sns
from src.heatdesalination.__utils__ import ProfileType

ALPHA = 1.0
fig, ax = plt.subplots()

solution = simulation_outputs

# Plot envelopes
ax.plot(
    solution["Electricity demand / kWh"].keys(),
    solution["Electricity demand / kWh"].values(),
    "--",
    label="demand",
    color="C3",
)
ax.plot(
    solution["Total collector electrical output power / kW"].keys(),
    solution["Total collector electrical output power / kW"].values(),
    "--",
    label="solar production",
    color="C1",
)
ax.plot(
    solution["Battery storage profile / kWh"].keys(),
    solution["Battery storage profile / kWh"].values(),
    "--",
    label="battery state of charge",
    color="C0",
)

###################
# Plot solar bars #
###################
ax.bar(
    solution["Electricity demand met through solar collectors / kWh"].keys(),
    solution["Electricity demand met through solar collectors / kWh"].values(),
    alpha=ALPHA,
    label="solar power supplied",
    color="C1",
)
bottom = list(
    solution["Electricity demand met through solar collectors / kWh"].values()
)

ax.bar(
    solution["Battery power inflow profile / kWh"].keys(),
    solution["Battery power inflow profile / kWh"].values(),
    alpha=0.6,
    label="power to batteries",
    bottom=bottom,
    color="C1",
)
bottom = [
    entry + solution["Battery power inflow profile / kWh"][index]
    for index, entry in enumerate(bottom)
]

ax.bar(
    solution["Dumped electricity / kWh"].keys(),
    solution["Dumped electricity / kWh"].values(),
    alpha=0.3,
    label="dumped solar",
    bottom=bottom,
    color="C1",
)
bottom = [
    entry + solution["Dumped electricity / kWh"][index]
    for index, entry in enumerate(bottom)
]

#####################
# Plot battery bars #
#####################
ax.bar(
    solution["Electricity demand met through storage / kWh"].keys(),
    solution["Electricity demand met through storage / kWh"].values(),
    alpha=ALPHA,
    label="storage power supplied",
    bottom=bottom,
    color="C0",
)
bottom = [
    entry + solution["Electricity demand met through storage / kWh"][index]
    for index, entry in enumerate(bottom)
]

##################
# Plot grid bars #
##################

ax.bar(
    solution["Electricity demand met through the grid / kWh"].keys(),
    solution["Electricity demand met through the grid / kWh"].values(),
    alpha=ALPHA,
    label="grid power supplied",
    bottom=bottom,
    color="C2",
)

plt.xlabel("Hour of day")
plt.ylabel("Average hourly power flow / kWh")

ax.legend(bbox_to_anchor=(1.0, 1.0))
plt.show()


######################################################
# Plotting weather profiles with standard deviations #
######################################################

import json
import matplotlib.pyplot as plt
import os
import seaborn as sns
from src.heatdesalination.__utils__ import ProfileType

sns.set_palette("colorblind")

with open(
    os.path.join("auto_generated", "fujairah_emirate_united_arab_emirates.json"),
    "r",
    encoding="UTF-8",
) as f:
    data = json.load(f)

keywords_to_plot = ["irradiance", "ambient_temperature", "wind_speed"]
ylabels = [
    "Solar irradiance / W/m^2",
    "Ambient temperature / degrees Celcius",
    "Wind speed / m/s",
]
map_colours = ["C1", "C3", "C0"]

for index, keyword in enumerate(keywords_to_plot):
    mapping = {
        int(key): value
        for key, value in data[ProfileType.AVERAGE.value][keyword].items()
    }
    average_profile = {key: mapping[key] for key in sorted(mapping)}
    mapping = {
        int(key): value
        for key, value in data[ProfileType.LOWER_ERROR_BAR.value][keyword].items()
    }
    lower_profile = {key: mapping[key] for key in sorted(mapping)}
    mapping = {
        int(key): value
        for key, value in data[ProfileType.UPPER_ERROR_BAR.value][keyword].items()
    }
    upper_profile = {key: mapping[key] for key in sorted(mapping)}
    mapping = {
        int(key): value
        for key, value in data[ProfileType.MAXIMUM.value][keyword].items()
    }
    max_profile = {key: mapping[key] for key in sorted(mapping)}
    mapping = {
        int(key): value
        for key, value in data[ProfileType.MINIMUM.value][keyword].items()
    }
    min_profile = {key: mapping[key] for key in sorted(mapping)}
    plt.plot(
        average_profile.keys(),
        average_profile.values(),
        label=f"average {keyword}",
        color=map_colours[index],
    )
    plt.fill_between(
        list(lower_profile.keys()),
        list(lower_profile.values()),
        list(upper_profile.values()),
        color=map_colours[index],
        alpha=0.5,
        label="PVGIS error",
    )
    plt.plot(
        max_profile.keys(),
        max_profile.values(),
        "--",
        label=f"maximum {keyword}",
        color=map_colours[index],
    )
    plt.plot(
        min_profile.keys(),
        min_profile.values(),
        "--",
        label=f"maximum {keyword}",
        color=map_colours[index],
    )
    plt.xlabel("Hour of day")
    plt.ylabel(ylabels[index])
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.show()

#####################################
# Post-simulation analysis plotting #
#####################################

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

output_name: str = "8_mar_debug"

ALPHA = 0.9
sns.set_style("whitegrid")
sns.set_context("notebook")

colorblind_palette = sns.color_palette(
    [
        "#E04606",  # Orange
        "#F09F52",  # Pale orange
        "#52C0AD",  # Pale green
        "#006264",  # Green
        "#D8247C",  # Pink
        "#EDEDED",  # Pale pink
        "#E7DFBE",  # Pale yellow
        "#FBBB2C",  # Yellow
    ]
)

sns.set_palette(colorblind_palette)

with open(
    os.path.join("simulation_outputs", f"{output_name}_average_weather_conditions.csv"),
    "r",
    encoding="UTF-8",
) as f:
    average_data = pd.read_csv(f, index_col=0)


with open(
    os.path.join(
        "simulation_outputs", f"{output_name}_lower_error_bar_weather_conditions.csv"
    ),
    "r",
    encoding="UTF-8",
) as f:
    lower_error_data = pd.read_csv(f, index_col=0)


with open(
    os.path.join(
        "simulation_outputs", f"{output_name}_upper_error_bar_weather_conditions.csv"
    ),
    "r",
    encoding="UTF-8",
) as f:
    upper_error_data = pd.read_csv(f, index_col=0)

# Temperature plot
x: list[int] = list(range(len(average_data)))
keys: list[str] = [
    "Ambient temperature / degC",
    # "Collector system input temperature / degC",
    # "PV-T collector output temperature / degC",
    "Collector system output temperature / degC",
    "Tank temperature / degC",
]

for index, key in enumerate(keys):
    plt.plot(x, average_data[key], c=f"C{index}", label=key.replace("/ degC", ""))
    # plt.plot(x, list(lower_error_data[key]), "--", c=f"C{index}")
    # plt.plot(x, list(upper_error_data[key]), "--", c=f"C{index}")
    # plt.fill_between(x, list(lower_error_data[key]), list(upper_error_data[key]), color=f"C{index}", alpha=0.1)

plt.legend()
plt.xlabel("Hour of day")
plt.ylabel("Temperature / degC")
plt.show()

# Heating energy plot
x: list[int] = list(range(len(average_data)))

# Compute the hot-water demand in heat
fig, ax = plt.subplots()
hot_water_heat_demand = (
    (average_data["Hot-water demand temperature / degC"] - 65)
    * average_data["Hot-water demand volume / kg/s"]
    * 4.184
)
tank_heat_supply = (
    (average_data["Tank temperature / degC"] - 65)
    * average_data["Hot-water demand volume / kg/s"]
    * 4.184
)

ax.plot(x, hot_water_heat_demand, "--", c="C2")
# ax.plot(x, tank_heat_supply, c="C1", label="Hot-water tanks")
ax.fill_between(
    x,
    [0] * len(x),
    tank_heat_supply,
    color="C0",
    alpha=ALPHA,
    label="Heat supplied from tank(s)",
)
# ax.plot(x, tank_heat_supply + average_data['Auxiliary heating demand / kWh(th)'], c="C2", label="Auxiliary heating")
ax.fill_between(
    x,
    tank_heat_supply,
    tank_heat_supply + average_data["Auxiliary heating demand / kWh(th)"],
    color="C1",
    alpha=ALPHA,
    label="Heat supplied from heat pump(s)",
)

ax2 = ax.twinx()
ax2.plot(
    x, average_data["Tank temperature / degC"], "--", c="C3", label="Tank temperature"
)

ax.legend()
ax2.legend()
ax.set_xlabel("Hour of day")
ax.set_ylabel("Thermal Energy Supplied / kWh(th)")
ax2.set_ylabel("Tank temperature / degC")
plt.show()

# Print the total amount of auxiliary heating vs in-house heating that took place
print(f"Total collector heating: {np.sum(tank_heat_supply)} kWh(th)")
print(
    f"Total auxiliary heating: {np.sum(average_data['Auxiliary heating demand / kWh(th)'])} kWh(th)"
)

# Plot the electrical sources
plt.plot(
    x,
    average_data["Electricity demand / kWh"],
    "--",
    c="C0",
    label="Total electricity demand",
)
plt.fill_between(
    x,
    average_data["Base electricity dewmand / kWh"],
    average_data["Base electricity dewmand / kWh"]
    + average_data["Electrical auxiliary heating demand / kWh(el)"],
    color="C2",
    label="Plant auxiliary heating requirements",
    alpha=ALPHA,
)
plt.fill_between(
    x,
    [0] * len(x),
    average_data["Base electricity dewmand / kWh"],
    color="C1",
    label="Plant electrical requirements",
    alpha=ALPHA,
)
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.xlabel("Hour of day")
plt.ylabel("Electrical demand / kWh")
plt.show()


sns.set_palette("PuBu_r", n_colors=5)
plt.fill_between(
    x,
    average_data["Electricity demand met through storage / kWh"]
    + average_data["Electricity demand met through solar collectors / kWh"]
    + average_data["Electricity demand met through the grid / kWh"]
    + average_data["Battery power inflow profile / kWh"],
    average_data["Electricity demand met through storage / kWh"]
    + average_data["Electricity demand met through solar collectors / kWh"]
    + average_data["Electricity demand met through the grid / kWh"]
    + average_data["Battery power inflow profile / kWh"]
    + average_data["Dumped electricity / kWh"],
    color="C4",
    label="Dumped electricity",
    alpha=ALPHA,
)
plt.fill_between(
    x,
    average_data["Electricity demand met through storage / kWh"]
    + average_data["Electricity demand met through solar collectors / kWh"]
    + average_data["Electricity demand met through the grid / kWh"],
    average_data["Electricity demand met through storage / kWh"]
    + average_data["Electricity demand met through solar collectors / kWh"]
    + average_data["Electricity demand met through the grid / kWh"]
    + average_data["Battery power inflow profile / kWh"],
    color="C3",
    label="Power to storage",
    alpha=ALPHA,
)
plt.fill_between(
    x,
    average_data["Electricity demand met through storage / kWh"]
    + average_data["Electricity demand met through solar collectors / kWh"],
    average_data["Electricity demand met through storage / kWh"]
    + average_data["Electricity demand met through solar collectors / kWh"]
    + average_data["Electricity demand met through the grid / kWh"],
    color="C2",
    label="Grid",
    alpha=ALPHA,
)
plt.fill_between(
    x,
    average_data["Electricity demand met through storage / kWh"],
    average_data["Electricity demand met through storage / kWh"]
    + average_data["Electricity demand met through solar collectors / kWh"],
    color="C1",
    label="Solar",
    alpha=ALPHA,
)
plt.fill_between(
    x,
    [0] * len(x),
    average_data["Electricity demand met through storage / kWh"],
    color="C0",
    label="Storage",
    alpha=ALPHA,
)
plt.plot(
    x,
    average_data["Electricity demand / kWh"],
    "--",
    c="C0",
    label="Electricity demand / kWh",
)
plt.legend()
plt.xlabel("Hour of day")
plt.ylabel("Source of electrical power / kWh")
plt.show()


##########################################
# Manual post-simulation cost comparison #
##########################################

interact

from src.heatdesalination.optimiser import TotalCost
from src.heatdesalination.__utils__ import OptimisableComponent

result

battery_capacity = result.x[0]
pv_system_size = result.x[1]
# pv_t_system_size = result.x[2]
# solar_thermal_system_size = result.x[3]

buffer_tank.cacpacity = optimisation_parameters.fixed_buffer_tank_capacitiy_value

# Run a simulation based on this.
solution = determine_steady_state_simulation(
    ambient_temperatures[profile_type],
    battery,
    battery_capacity,
    buffer_tank,
    desalination_plant,
    optimisation_parameters.fixed_mass_flow_rate_value,
    hybrid_pv_t_panel,
    logger,
    pv_panel,
    pv_system_size,
    optimisation_parameters.fixed_pv_t_value,
    # pv_t_system_size,
    scenario,
    solar_irradiances[profile_type],
    solar_thermal_collector,
    optimisation_parameters.fixed_st_value,
    # solar_thermal_system_size,
    system_lifetime,
    disable_tqdm=disable_tqdm,
)

print(
    "Total cost = {} MUSD".format(
        TotalCost.calculate_value(
            {
                battery: battery_capacity,
                buffer_tank: buffer_tank.capacity,
                pv_panel: pv_system_size,
                # hybrid_pv_t_panel: pv_t_system_size,
                hybrid_pv_t_panel: optimisation_parameters.fixed_pv_t_value,
                # solar_thermal_collector: solar_thermal_system_size,
                solar_thermal_collector: optimisation_parameters.fixed_st_value,
            },
            scenario,
            solution,
            system_lifetime,
        )
        / 10**6
    )
)

###########################################
# Plotting costs of the surrounding areas #
###########################################

import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.heatdesalination.__utils__ import ProfileType
from src.heatdesalination.optimiser import TotalCost

with open("25_by_25_pv_t_st_square.json", "r", encoding="UTF-8") as f:
    data = json.load(f)

costs = [
    (entry["results"][ProfileType.AVERAGE.value][TotalCost.name] / 10**6)
    for entry in data
]
# palette = sns.color_palette("blend:#0173B2,#64B5CD", as_cmap=True)
palette = sns.color_palette("rocket", as_cmap=True)

pv_sizes = [entry["simulation"]["pv_system_size"] for entry in data]
battery_capacities = [entry["simulation"]["battery_capacity"] for entry in data]

frame = pd.DataFrame(
    {
        "Number of PV panels": pv_sizes,
        "Storage capacity / kWh": battery_capacities,
        "Cost / MUSD": costs,
    }
).pivot(
    index="Number of PV panels",
    columns="Storage capacity / kWh",
    values="Cost / MUSD",
)
# sns.heatmap(frame, cmap=palette, annot=True)
sns.heatmap(frame, cmap=palette, annot=False)
plt.show()

# adapt the colormaps such that the "under" or "over" color is "none"
# cmap1 = sns.color_palette("blend:#FAECD9,#AD5506", as_cmap=True)
# cmap1 = sns.dark_palette("502382", as_cmap=True)
# cmap1 = sns.color_palette("PuOr_r", as_cmap=True)
cmap1 = sns.cubehelix_palette(
    start=0.2, rot=-0.3, dark=0.5, light=1, as_cmap=True, reverse=True
)
cmap1.set_over("none")

# cmap2 = sns.color_palette("blend:#502382,#E9E9F1", as_cmap=True)
# cmap2 = sns.light_palette("502382", as_cmap=True)
# cmap2 = sns.color_palette("PuBu_r", as_cmap=True)
cmap2 = sns.cubehelix_palette(
    start=0.2, rot=-0.3, dark=0, light=0.5, as_cmap=True, reverse=True
)
cmap2.set_over("none")

ax1 = sns.heatmap(frame, vmin=2, vmax=max(costs), cmap=cmap1, cbar_kws={"pad": 0.02})
ax2 = sns.heatmap(frame, vmin=min(costs), vmax=2, cmap=cmap2, ax=ax1)

# Invert the axes
min_cost_index = {cost: index for index, cost in enumerate(costs)}[min(costs)]
plt.scatter(
    [battery_capacities[min_cost_index]],
    [pv_sizes[min_cost_index]],
    marker="x",
    color="red",
    s=10000,
    zorder=-1,
)

ax3 = sns.heatmap(
    frame,
    mask=frame > min(costs),
    cmap=cmap2,
    cbar=False,
    annot=True,
    annot_kws={"weight": "bold"},
)

ax1.invert_yaxis()
ax2.invert_yaxis()
ax3.invert_yaxis()

plt.show()

################################
# Sub-data version for squares #
################################

sub_data = [entry for entry in data if entry["simulation"]["pv_system_size"] > 3000]

costs = [
    (entry["results"][ProfileType.AVERAGE.value][1][TotalCost.name] / 10**6)
    for entry in sub_data
]
# palette = sns.color_palette("blend:#0173B2,#64B5CD", as_cmap=True)
palette = sns.color_palette("rocket", as_cmap=True)

pv_sizes = [entry["simulation"]["pv_system_size"] for entry in sub_data]
battery_capacities = [entry["simulation"]["battery_capacity"] for entry in sub_data]

frame = pd.DataFrame(
    {
        "Number of PV panels": pv_sizes,
        "Storage capacity / kWh": battery_capacities,
        "Cost / MUSD": costs,
    }
).pivot(
    index="Number of PV panels",
    columns="Storage capacity / kWh",
    values="Cost / MUSD",
)
sns.heatmap(frame, cmap=palette, annot=True)
# sns.heatmap(frame, cmap=palette, annot=False)
plt.show()

################
# Contour plot #
################


import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.heatdesalination.__utils__ import ProfileType
from src.heatdesalination.optimiser import TotalCost


with open("200_x_200_pv_batt_square.json", "r", encoding="UTF-8") as f:
    data = json.load(f)

palette = sns.color_palette("rocket", as_cmap=True)
pv_sizes = [entry["simulation"]["pv_system_size"] for entry in data]
battery_capacities = [entry["simulation"]["battery_capacity"] for entry in data]
costs = [
    (entry["results"][ProfileType.AVERAGE.value][1][TotalCost.name] / 10**6)
    for entry in data
]

batt_uniq, batt_index = np.unique(battery_capacities, return_inverse=True)
pv_uniq, pv_index = np.unique(pv_sizes, return_inverse=True)
batt_mesh, pv_mesh = np.meshgrid(batt_uniq, pv_uniq)
cost_mesh = np.array(costs).reshape(batt_mesh.shape)

fig, ax = plt.subplots()

contours = ax.contourf(batt_mesh, pv_mesh, cost_mesh, 100, cmap=cmap2)
fig.colorbar(contours, ax=ax)

plt.show()


# adapt the colormaps such that the "under" or "over" color is "none"
cmap1 = sns.color_palette("blend:#0173b2,#5AC4FE", as_cmap=True)
# cmap1 = sns.cubehelix_palette(
#     start=0.5, rot=-0.5, dark=0.5, light=1, as_cmap=True, reverse=True
# )
cmap1.set_over("none")
cmap2 = sns.color_palette("blend:#5AC4FE,#FFFFFF", as_cmap=True)
# cmap2 = sns.cubehelix_palette(
#     start=0.5, rot=-0.5, dark=0, light=0.5, as_cmap=True, reverse=True
# )
cmap2.set_over("none")

# cmap1 = sns.color_palette("blend:#00548F,#84BB4E", as_cmap=True)
# cmap1.set_over("none")
# cmap2 = sns.color_palette("blend:#84BB4E,#00548F", as_cmap=True)
# cmap2.set_over("none")


fig, ax = plt.subplots()

contours = ax.contour(
    batt_mesh,
    pv_mesh,
    cost_mesh,
    300,
    cmap=cmap2,
    extent=[
        min(battery_capacities),
        max(battery_capacities),
        min(pv_sizes),
        max(pv_sizes),
    ],
)
fig.colorbar(contours, ax=ax)

plt.xlabel("Battery capacity / kWh")
plt.ylabel("PV system size / collectors")

min_cost_index = {cost: index for index, cost in enumerate(costs)}[min(costs)]
plt.scatter(
    [battery_capacities[min_cost_index]],
    [pv_sizes[min_cost_index]],
    marker="x",
    color="red",
    s=1000,
    zorder=10,
)

plt.show()

##################################
# Second attempt at contour plot #
##################################

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.heatdesalination.__utils__ import ProfileType
from src.heatdesalination.optimiser import (
    AuxiliaryHeatingFraction,
    GridElectricityFraction,
    SolarElectricityFraction,
    StorageElectricityFraction,
    TotalCost,
)

sns.set_palette("PuBu")
sns.set_context("paper")
sns.set_style("whitegrid")

with open("pv_t_1262_st_318_tank_49_output.json", "r", encoding="UTF-8") as f:
    data = json.load(f)

with open("pv_pv_t_square_25_x_25.json", "r", encoding="UTF-8") as f:
    data = json.load(f)

try:
    auxiliary_heating_fraction = [
        (
            (entry["results"][ProfileType.AVERAGE.value][AuxiliaryHeatingFraction.name])
            if entry["results"] is not None
            else None
        )
        for entry in data
    ]
except TypeError:
    auxiliary_heating_fraction = [
        (
            (
                entry["results"][ProfileType.AVERAGE.value][1][
                    AuxiliaryHeatingFraction.name
                ]
            )
            if entry["results"] is not None
            else None
        )
        for entry in data
    ]

try:
    grid_fraction = [
        (
            (entry["results"][ProfileType.AVERAGE.value][GridElectricityFraction.name])
            if entry["results"] is not None
            else None
        )
        for entry in data
    ]
except TypeError:
    grid_fraction = [
        (
            (
                entry["results"][ProfileType.AVERAGE.value][1][
                    GridElectricityFraction.name
                ]
            )
            if entry["results"] is not None
            else None
        )
        for entry in data
    ]

try:
    solar_fraction = [
        (
            (entry["results"][ProfileType.AVERAGE.value][SolarElectricityFraction.name])
            if entry["results"] is not None
            else None
        )
        for entry in data
    ]
except TypeError:
    solar_fraction = [
        (
            (
                entry["results"][ProfileType.AVERAGE.value][1][
                    SolarElectricityFraction.name
                ]
            )
            if entry["results"] is not None
            else None
        )
        for entry in data
    ]

try:
    storage_fraction = [
        (
            (
                entry["results"][ProfileType.AVERAGE.value][
                    StorageElectricityFraction.name
                ]
            )
            if entry["results"] is not None
            else None
        )
        for entry in data
    ]
except TypeError:
    storage_fraction = [
        (
            (
                entry["results"][ProfileType.AVERAGE.value][1][
                    StorageElectricityFraction.name
                ]
            )
            if entry["results"] is not None
            else None
        )
        for entry in data
    ]

try:
    costs = [
        (
            (entry["results"][ProfileType.AVERAGE.value][TotalCost.name] / 10**6)
            if entry["results"] is not None
            else None
        )
        for entry in data
    ]
except TypeError:
    costs = [
        (
            (entry["results"][ProfileType.AVERAGE.value][1][TotalCost.name] / 10**6)
            if entry["results"] is not None
            else None
        )
        for entry in data
    ]

# palette = sns.color_palette("blend:#0173B2,#64B5CD", as_cmap=True)
# palette = sns.color_palette("rocket", as_cmap=True)

pv_sizes = [entry["simulation"]["pv_system_size"] for entry in data]
battery_capacities = [entry["simulation"]["battery_capacity"] for entry in data]
pv_t_sizes = [entry["simulation"]["pv_t_system_size"] for entry in data]
solar_thermal_sizes = [
    entry["simulation"]["solar_thermal_system_size"] for entry in data
]

# Generate the frame
frame = pd.DataFrame(
    {
        "Storage capacity / kWh": battery_capacities,
        # "Number of PV-T collectors": pv_t_sizes,
        "Number of PV panels": pv_sizes,
        "Cost / MUSD": costs,
    }
)
pivotted_frame = frame.pivot(
    index="Number of PV panels",
    columns="Storage capacity / kWh",
    values="Cost / MUSD"
    # index="Number of PV panels", columns="Number of PV-T collectors", values="Cost / MUSD"
)

# PV-T and solar-thermal
frame = pd.DataFrame(
    {
        "Solar-thermal capacity / collectors": solar_thermal_sizes,
        "PV-T capacity / collectors": pv_t_sizes,
        # "Cost / MUSD": costs,
        "Auxiliary heating fraction": auxiliary_heating_fraction
        # "Grid fraction": grid_fraction
        # "Solar fraction": solar_fraction
        # "Storage fraction": storage_fraction
    }
)
pivotted_frame = frame.pivot(
    index="PV-T capacity / collectors",
    columns="Solar-thermal capacity / collectors",
    # values="Cost / MUSD"
    values="Auxiliary heating fraction"
    # values="Grid fraction"
    # values="Solar fraction"
    # values="Storage fraction"
)

# Extract the arrays.
Z = pivotted_frame.values

# X_unique = np.sort(frame["Storage capacity / kWh"].unique())
# Y_unique = np.sort(frame["Number of PV panels"].unique())

X_unique = np.sort(frame["Solar-thermal capacity / collectors"].unique())
Y_unique = np.sort(frame["PV-T capacity / collectors"].unique())

# X_unique = np.sort(frame["Number of PV-T collectors"].unique())
# Y_unique = np.sort(frame["Number of PV panels"].unique())

X, Y = np.meshgrid(X_unique, Y_unique)

# Define levels in z-axis where we want lines to appear
levels = np.array(
    [
        0.0,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.1,
        0.11,
        0.12,
        0.13,
        0.14,
        0.15,
        0.16,
        0.17,
        0.18,
        0.19,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.45,
        1.5,
        1.55,
        1.6,
        1.65,
        1.7,
        1.75,
        1.8,
        1.85,
        1.9,
        2,
        2.1,
        2.2,
        2.3,
        2.4,
        2.5,
        2.6,
        2.7,
        2.8,
        2.9,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ]
)

# Generate a color mapping of the levels we've specified
fig, ax = plt.subplots()
cmap = sns.color_palette("PuBu", len(levels), as_cmap=True)
cpf = ax.contourf(X, Y, Z, len(levels), cmap=cmap)

# Set all level lines to black
line_colors = ["black" for l in cpf.levels]

# Make plot and customize axes
contours = ax.contour(X, Y, Z, levels=levels, colors=line_colors)
ax.clabel(contours, fontsize=10, colors=line_colors)

# ax.set_xlabel("Storage size / kWh")
# _ = ax.set_ylabel("PV size / number of collectors")

# ax.set_xlabel("PV-T size / number of collectors")
# _ = ax.set_ylabel("PV size / number of collectors")

ax.set_xlabel("Solar-thermal capacity / collectors")
ax.set_ylabel("PV-T capacity / collectors")

fig.colorbar(cpf, ax=ax, label="Total lifetime cost / MUSD")
# fig.colorbar(cpf, ax=ax, label="Auxiliary heating fraction")
# fig.colorbar(cpf, ax=ax, label="Grid fraction")
# fig.colorbar(cpf, ax=ax, label="Solar fraction")
# fig.colorbar(cpf, ax=ax, label="Storage fraction")

plt.show()

min_cost_index = {cost: index for index, cost in enumerate(costs)}[min(costs)]

sns.set_palette("colorblind")

# Open all vec files and scatter their journeys
plt.scatter(
    battery_capacities[min_cost_index],
    pv_sizes[min_cost_index],
    marker="x",
    color="#A40000",
    label="optimum point",
    linewidths=2.5,
    s=150,
    zorder=1,
)

# Nelder-Mead
with open("pv_t_1262_st_318_tank_49_nelder_mead_vecs.json", "r", encoding="UTF-8") as f:
    nm_vecs = json.load(f)

plt.scatter(
    nm_vecs[-1][0],
    nm_vecs[-1][1],
    marker="x",
    color="C0",
    label="Nelder-Mead",
    linewidths=2.5,
    s=150,
    zorder=1,
)
plt.plot(
    [entry[0] for entry in nm_vecs],
    [entry[1] for entry in nm_vecs],
    color="C0",
    marker="x",
)

# Powell
with open("pv_t_1262_st_318_tank_49_powell.json", "r", encoding="UTF-8") as f:
    powell_vecs = json.load(f)

plt.scatter(
    powell_vecs[-1][0],
    powell_vecs[-1][1],
    marker="x",
    color="C1",
    label="Powell",
    linewidths=2.5,
    s=150,
    zorder=1,
)
plt.plot(
    [entry[0] for entry in powell_vecs],
    [entry[1] for entry in powell_vecs],
    color="C1",
    marker="x",
)

# CG
with open("pv_t_1262_st_318_tank_49_cg.json", "r", encoding="UTF-8") as f:
    cg_vecs = json.load(f)

plt.scatter(
    cg_vecs[-1][0],
    cg_vecs[-1][1],
    marker="x",
    color="C2",
    label="CG",
    linewidths=2.5,
    s=150,
    zorder=1,
)
plt.plot(
    [entry[0] for entry in cg_vecs],
    [entry[1] for entry in cg_vecs],
    color="C2",
    marker="x",
)

# BFGS
with open("pv_t_1262_st_318_tank_49_bfgs.json", "r", encoding="UTF-8") as f:
    bfgs_vecs = json.load(f)

plt.scatter(
    bfgs_vecs[-1][0],
    bfgs_vecs[-1][1],
    marker="x",
    color="C3",
    label="BFGS",
    linewidths=2.5,
    s=150,
    zorder=1,
)
plt.plot(
    [entry[0] for entry in bfgs_vecs],
    [entry[1] for entry in bfgs_vecs],
    color="C3",
    marker="x",
)

# L-BFGS-B
with open("pv_t_1262_st_318_tank_49_l_bfgs_g.json", "r", encoding="UTF-8") as f:
    l_bfgs_g_vecs = json.load(f)

plt.scatter(
    l_bfgs_g_vecs[-1][0],
    l_bfgs_g_vecs[-1][1],
    marker="x",
    color="C4",
    label="L-BFGS-B",
    linewidths=2.5,
    s=150,
    zorder=1,
)
plt.plot(
    [entry[0] for entry in l_bfgs_g_vecs],
    [entry[1] for entry in l_bfgs_g_vecs],
    color="C4",
    marker="x",
)

# TNC
with open("pv_t_1262_st_318_tank_49_tnc.json", "r", encoding="UTF-8") as f:
    tnc_vecs = json.load(f)

plt.scatter(
    tnc_vecs[-1][0],
    tnc_vecs[-1][1],
    marker="x",
    color="C5",
    label="TNC",
    linewidths=2.5,
    s=150,
    zorder=1,
)
plt.plot(
    [entry[0] for entry in tnc_vecs],
    [entry[1] for entry in tnc_vecs],
    color="C5",
    marker="x",
)

plt.legend(bbox_to_anchor=(1.0, 1.0))

plt.xlabel("Storage capacity / kWh")
plt.ylabel("Number of PV panels")

plt.xlabel("Solar-thermal collector capacity / collectors")
plt.ylabel("PV-T collector capacity / collectors")

plt.xlim(min(battery_capacities), max(battery_capacities))
plt.ylim(min(pv_sizes), max(pv_sizes))

plt.show()

plt.savefig(
    "pv_batt_parameter_optimisation_heatmap.png",
    dpi=1200,
    transparent=True,
    bbox_inches="tight",
)


#########################
# Post-HPC contour plot #
#########################

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.heatdesalination.__utils__ import ProfileType
from src.heatdesalination.optimiser import TotalCost

with open("25_by_25_pv_t_st_square.json", "r", encoding="UTF-8") as f:
    min_cost_list = json.load(f)

costs = [
    (entry["results"][ProfileType.AVERAGE.value][TotalCost.name] / 10**6)
    for entry in min_cost_list
    if entry[(simulation_key := "simulation")][(tank_key := "buffer_tank_capacity")]
    == 32
]
# palette = sns.color_palette("blend:#0173B2,#64B5CD", as_cmap=True)
# palette = sns.color_palette("rocket", as_cmap=True)

pv_sizes = [
    entry[simulation_key]["pv_system_size"]
    for entry in min_cost_list
    if entry[simulation_key][tank_key] == 32
]
battery_capacities = [
    entry[simulation_key]["battery_capacity"]
    for entry in min_cost_list
    if entry[simulation_key][tank_key] == 32
]
pv_t_sizes = [
    entry[simulation_key]["pv_t_system_size"]
    for entry in min_cost_list
    if entry[simulation_key][tank_key] == 32
]
st_sizes = [
    entry[simulation_key]["solar_thermal_system_size"]
    for entry in min_cost_list
    if entry[simulation_key][tank_key] == 32
]

# Generate the frame in PV-batt space
pv_sizes = [
    entry[simulation_key]["pv_system_size"]
    for entry in min_cost_list
    if entry[simulation_key][tank_key] == 32
    and entry[simulation_key]["pv_t_system_size"] == 72
    and entry[simulation_key]["solar_thermal_system_size"] == 218
]
battery_capacities = [
    entry[simulation_key]["battery_capacity"]
    for entry in min_cost_list
    if entry[simulation_key][tank_key] == 32
    and entry[simulation_key]["pv_t_system_size"] == 72
    and entry[simulation_key]["solar_thermal_system_size"] == 218
]
costs = [
    (entry["results"][ProfileType.AVERAGE.value][TotalCost.name] / 10**6)
    for entry in min_cost_list
    if entry[simulation_key][tank_key] == 32
    and entry[simulation_key]["pv_t_system_size"] == 72
    and entry[simulation_key]["solar_thermal_system_size"] == 218
]
frame = pd.DataFrame(
    {
        "Storage capacity / kWh": battery_capacities,
        "Number of PV panels": pv_sizes,
        "Cost / MUSD": costs,
    }
)
pivotted_frame = frame.pivot(
    index="Number of PV panels",
    columns="Storage capacity / kWh",
    values="Cost / MUSD",
)

# Generate the frame
frame = pd.DataFrame(
    {
        "Number of PV-T collectors": pv_t_sizes,
        "Number of solar-thermal collectors": st_sizes,
        "Cost / MUSD": costs,
    }
)
pivotted_frame = frame.pivot(
    index="Number of PV-T collectors",
    columns="Number of solar-thermal collectors",
    values="Cost / MUSD",
)

# # Generate the frame

# Extract the arrays.
Z = pivotted_frame.values
# X_unique = np.sort(frame["Storage capacity / kWh"].unique())
# Y_unique = np.sort(frame["Number of PV panels"].unique())
X_unique = np.sort(frame["Number of PV-T collectors"].unique())
Y_unique = np.sort(frame["Number of solar-thermal collectors"].unique())
X, Y = np.meshgrid(X_unique, Y_unique)

# Define levels in z-axis where we want lines to appear
levels = np.array(
    [
        1.4,
        1.45,
        1.5,
        1.55,
        1.6,
        1.65,
        1.7,
        1.75,
        1.8,
        1.85,
        1.9,
        2,
        2.5,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ]
)

# Generate a color mapping of the levels we've specified
fig, ax = plt.subplots()
cmap = sns.color_palette("PuBu", len(levels), as_cmap=True)
cpf = ax.contourf(X, Y, Z, len(levels), cmap=cmap)

# Set all level lines to black
line_colors = ["black" for l in cpf.levels]

# Make plot and customize axes
contours = ax.contour(X, Y, Z, levels=levels, colors=line_colors)
ax.clabel(contours, fontsize=10, colors=line_colors)
ax.set_xlabel("Storage size / kWh")
_ = ax.set_ylabel("PV size / number of collectors")

fig.colorbar(cpf, ax=ax, label="Total lifetime cost / MUSD")
#############
# Subsquare #
#############

for start_index in range(27):
    sub_data = data[start_index::(repeat_index := 27)]
    pv_sizes = [entry["simulation"]["pv_system_size"] for entry in sub_data]
    battery_capacities = [entry["simulation"]["battery_capacity"] for entry in sub_data]
    frame = pd.DataFrame(
        {
            "Number of PV panels": pv_sizes,
            "Storage capacity / kWh": battery_capacities,
            "Cost / MUSD": costs[start_index::repeat_index],
        }
    ).pivot(
        index="Number of PV panels",
        columns="Storage capacity / kWh",
        values="Cost / MUSD",
    )
    sns.heatmap(frame, cmap=palette, annot=True)
    plt.show()


##############################
# Setting up simulation runs #
##############################

import json
import os

from tqdm import tqdm

default_entry = {
    (batt_key := "battery_capacity"): 1400,
    (tank_key := "buffer_tank_capacity"): 30000,
    "mass_flow_rate": 20,
    (pv_key := "pv_system_size"): 0,
    (pv_t_key := "pv_t_system_size"): 300,
    (st_key := "solar_thermal_system_size"): 72,
    "scenario": "default_uae",
    "start_hour": 8,
    "system_lifetime": 25,
    "output": None,
    "profile_types": ["avr", "uer", "ler"],
}

battery_capacities = range(0, 1001, 100)
pv_sizes = range(0, 6001, 250)
pv_t_sizes = range(0, 3600, 145)
solar_thermal_sizes = range(0, 12830, 514)
tank_capacities = range(15, 100, 80)

runs = []
for batt in battery_capacities:
    for pv in pv_sizes:
        entry = default_entry.copy()
        entry[batt_key] = batt
        entry[pv_key] = pv
        runs.append(entry)

with open(
    os.path.join("inputs", "pv_batt_square_200_x_200.json"), "w", encoding="UTF-8"
) as f:
    json.dump(runs, f)

runs = []
for batt in battery_capacities:
    for pv_t in pv_t_sizes:
        entry = default_entry.copy()
        entry[batt_key] = batt
        entry[pv_t_key] = pv_t
        runs.append(entry)

with open(
    os.path.join("inputs", "pv_t_batt_square_simulations.json"), "w", encoding="UTF-8"
) as f:
    json.dump(runs, f)

runs = []
for batt in battery_capacities:
    for st in solar_thermal_sizes:
        entry = default_entry.copy()
        entry[batt_key] = batt
        entry[st_key] = st
        runs.append(entry)


with open(
    os.path.join("inputs", "st_batt_square_simulations.json"), "w", encoding="UTF-8"
) as f:
    json.dump(runs, f)

runs = []
for pv in pv_sizes:
    for pv_t in pv_t_sizes:
        entry = default_entry.copy()
        entry[pv_key] = pv
        entry[pv_t_key] = pv_t
        runs.append(entry)

with open(
    os.path.join("inputs", "pv_pv_t_square_simulations.json"), "w", encoding="UTF-8"
) as f:
    json.dump(runs, f)

runs = []
for pv in pv_sizes:
    for st in solar_thermal_sizes:
        entry = default_entry.copy()
        entry[pv_key] = pv
        entry[st_key] = st
        runs.append(entry)

with open(
    os.path.join("inputs", "pv_st_square_simulations.json"), "w", encoding="UTF-8"
) as f:
    json.dump(runs, f)

runs = []
for pv_t in pv_t_sizes:
    for st in solar_thermal_sizes:
        entry = default_entry.copy()
        entry[pv_t_key] = pv_t
        entry[st_key] = st
        runs.append(entry)

runs.pop(0)

with open(
    os.path.join("inputs", "pv_t_st_square_simulations.json"), "w", encoding="UTF-8"
) as f:
    json.dump(runs, f)

runs = []

for batt in tqdm(battery_capacities, desc="batt"):
    for pv in tqdm(pv_sizes, desc="pv", leave=False):
        for pv_t in pv_t_sizes:
            for st in solar_thermal_sizes:
                for tank in tank_capacities:
                    entry = default_entry.copy()
                    entry[batt_key] = batt
                    entry[pv_key] = pv
                    entry[pv_t_key] = pv_t
                    entry[st_key] = st
                    runs.append(entry)

with open(
    os.path.join("inputs", "fifty_by_fifth_simulations.json"), "w", encoding="UTF-8"
) as f:
    json.dump(runs, f)

import shutil

shutil.copy2(
    os.path.join("inputs", "pv_t_st_square_simulations.json"),
    os.path.join("inputs", "simulations.json"),
)

# Runs for the HPC
basename = os.path.join("inputs", "pv_t_{pv_t}_st_{st}_tank_{tank}_runs.json")
for pv_t in tqdm(pv_t_sizes, desc="pv_t_sizes"):
    for st in tqdm(solar_thermal_sizes, desc="st_sizes", leave=False):
        for tank in tqdm(tank_capacities, desc="tank capacities", leave=False):
            # Setup runs at this resolution
            runs = []
            for batt in tqdm(battery_capacities, desc="batt", leave=False):
                for pv in tqdm(pv_sizes, desc="pv", leave=False):
                    entry = default_entry.copy()
                    entry[batt_key] = batt
                    entry[pv_key] = pv
                    entry[pv_t_key] = pv_t
                    entry[st_key] = st
                    entry[tank_key] = tank
                    runs.append(entry)
            # Save these runs to the file.
            with open(
                os.path.join(basename.format(pv_t=pv_t, st=st, tank=tank)),
                "w",
                encoding="UTF-8",
            ) as f:
                json.dump(runs, f)

##############################
# Assemble the HPC runs file #
##############################

import re

import json

default_entry = {
    "location": "fujairah_united_arab_emirates",
    (simulation_key := "simulation"): None,
    (output_key := "output"): None,
}

regex = re.compile(r"(P?pv_t_\d*_st_\d*_tank_\d*_runs)")
entries = [entry for entry in os.listdir("inputs") if regex.match(entry) is not None]

hpc_simulations = []
for entry in entries:
    temp = default_entry.copy()
    temp[simulation_key] = regex.match(entry).group(0)
    temp[output_key] = f"{regex.match(entry).group(0)}_output"
    hpc_simulations.append(temp)

#######################
# HPC file processing #
#######################

import os
import re

import json

from tqdm import tqdm

from src.heatdesalination.__utils__ import ProfileType
from src.heatdesalination.optimiser import TotalCost

os.chdir("hpc_parallel_simulations")

regex = re.compile(r"pv_t_(?P<pv_t>\d*)_st_(?P<st>\d*)_tank_(?P<tank>\d*)_runs_output")
output_filenames = [
    entry for entry in os.listdir(".") if regex.match(entry) is not None
]

# Cycle through the file names, compute the costs, and, if the file has a lower cost,
# save it as the lowest-cost filename.

min_cost: float = 10**10
min_cost_filename: str | None = None
min_cost_overflow: dict[str, float] = {}
output_to_display: list[str] = []

for filename in tqdm(output_filenames, desc="files", unit="file"):
    with open(filename, "r", encoding="UTF-8") as f:
        data = json.load(f)
    # Calculate the costs
    costs = [
        (entry["results"][ProfileType.AVERAGE.value][TotalCost.name] / 10**6)
        for entry in data
    ]
    # If the lowest cost is lower than the lowest value encountered so far, use this.
    if (current_minimum_cost := min(costs)) < min_cost:
        min_cost_filename = filename
        min_cost = current_minimum_cost
        print((output := f"New min cost in {min_cost_filename}: {min_cost:.3g}"))
        output_to_display.append(output)
        continue
    # If the lowest cost is equal to the lowest value encountered so far, save this.
    if current_minimum_cost == min_cost:
        min_cost_overflow[filename] = current_minimum_cost
        print((output := "Equal min cost found, saving"))
        output_to_display.append(output)

print("Min cost file {}".format(min_cost_filename))
output_to_display.append(f"Min cost of {min_cost} in {min_cost_filename}")

with open("../min_cost_analysis.txt", "w", encoding="UTF-8") as f:
    f.writelines(output_to_display)

##########################################
# HPC min-cost search in parallel planes #
##########################################

regex = re.compile(r"pv_t_(?P<pv_t>\d*)_st_(?P<st>\d*)_tank_(?P<tank>\d*)_runs_output")
output_filenames = [
    entry for entry in os.listdir(".") if regex.match(entry) is not None
]


# Assemble a list containing the cost of the various points matching PV and batt.
# min_cost_batt: float = 170
# min_cost_pv: float = 6400
min_cost_list = []

batt_key: str = "battery_capacity"
pv_key: str = "pv_system_size"
pv_t_key: str = "pv_t_system_size"
st_key: str = "solar_thermal_system_size"
tank_key: str = "buffer_tank_capacitiy"

simulation_key: str = "simulation"

# Cycle through the filenames
for filename in tqdm(output_filenames, desc="files", unit="file"):
    with open(filename, "r", encoding="UTF-8") as f:
        data = json.load(f)
    # Find the point with the matching PV and batt
    min_cost_list.extend(
        [
            entry
            for entry in data
            if entry[simulation_key][batt_key] == min_cost_batt
            and entry[simulation_key][pv_key] == min_cost_pv
        ]
    )

####################
# Dump all vectors #
####################

interact

import json

algorithm = "cobyla"
vecs = [list(entry) for entry in result.allvecs]

with open(f"pv_t_1262_st_318_tank_49_{algorithm}.json", "w", encoding="UTF-8") as f:
    json.dump(vecs, f)

##########################
# Creating optimisations #
##########################

default_optimisation = {
    "location": "fujairah_united_arab_emirates",
    (output_key := "output"): "parallel_optimisation_output_1",
    "profile_types": ["avr", "ler", "uer"],
    (scenario_key := "scenario"): "default",
    (system_lifetime_key := "system_lifetime"): 25,
}

output = "fujairah_uae_{}"
optimisations = []

for discount_rate in range(-20, 21, 1):
    scenario = "uae_dr_{}".format(
        f"{'m_' if discount_rate < 0 else ''}{f'{round(discount_rate/10, 2)}'.replace('.', '').replace('-','')}"
    )
    optimisation = default_optimisation.copy()
    optimisation[output_key] = output.format(scenario)
    optimisation[scenario_key] = scenario
    optimisations.append(optimisation)

with open(os.path.join("inputs", "optimisations.json"), "w", encoding="UTF-8") as f:
    json.dump(optimisations, f)

##########################################
# Post-processing parallel optimisations #
##########################################

import json
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.signal import lfilter
from src.heatdesalination.__utils__ import ProfileType

# Noise-filtering parameters
n = 5  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1

ALPHA = 0.5
DPI = 1200
sns.set_style("whitegrid")
sns.set_context((context := "notebook"))
sns.set_palette("colorblind")

# !! Update both name and fig identifier
with open("hpc_nm_low_tol_grid_optimisations_probe.json", "r", encoding="UTF-8") as f:
    data = json.load(f)

fig_identifier: str = f"nm_low_tol_grid_optimisations_{context}"
FIGURE_DIMENSIONS = (8.27, 8.27)

keys = [
    "total_cost",
    "auxiliary_heating_fraction",
    "dumped_electricity",
    "grid_electricity_fraction",
    "solar_electricity_fraction",
    "storage_electricity_fraction",
]

optimisation_titles = [
    "Four parameter",
    "No-PV optimisation (PV-T, ST, Storage)",
    "No-PV-T optimisation (PV, ST, Storage)",
    "No-ST optimisation (PV, PV-T, Storage)",
]

x = [entry["optimisation"]["scenario"] for entry in data]

# Grid discount processing
x = [
    entry_2.replace("m_", "-")
    for entry_2 in [
        entry_1.split("_dr_")[-1] for entry_1 in [entry.split("uae")[1] for entry in x]
    ]
]
x = [entry.replace("--", "-").replace("_", ".") for entry in x]
x = [float(entry) for entry in x]
# x = [(-(200 + entry) if entry < 0 else entry) for entry in x]

# Heat-exchanger efficiency processing
# x = [float(entry.split("_")[-1].split("%")[0]) for entry in x]

# PV degradation efficiency processing
# x = [float(".".join(entry.split("deg_")[1:]).replace("%", "")) / 100 for entry in x]

# Plot the various component sizes for each of the runs.
# Result 0 - four-parameter optimisation
plt.figure()  # figsize=FIGURE_DIMENSIONS)
sns.set_palette("colorblind")
batt = [entry["result"][0][1][ProfileType.AVERAGE.value][1][0] for entry in data]
ler_batt = [
    entry["result"][0][1][ProfileType.LOWER_ERROR_BAR.value][1][0] for entry in data
]
uer_batt = [
    entry["result"][0][1][ProfileType.UPPER_ERROR_BAR.value][1][0] for entry in data
]
pv = [entry["result"][0][1][ProfileType.AVERAGE.value][1][1] for entry in data]
ler_pv = [
    entry["result"][0][1][ProfileType.LOWER_ERROR_BAR.value][1][1] for entry in data
]
uer_pv = [
    entry["result"][0][1][ProfileType.UPPER_ERROR_BAR.value][1][1] for entry in data
]
pv_t = [entry["result"][0][1][ProfileType.AVERAGE.value][1][2] for entry in data]
ler_pv_t = [
    entry["result"][0][1][ProfileType.LOWER_ERROR_BAR.value][1][2] for entry in data
]
uer_pv_t = [
    entry["result"][0][1][ProfileType.UPPER_ERROR_BAR.value][1][2] for entry in data
]
st = [entry["result"][0][1][ProfileType.AVERAGE.value][1][3] for entry in data]
ler_st = [
    entry["result"][0][1][ProfileType.LOWER_ERROR_BAR.value][1][3] for entry in data
]
uer_st = [
    entry["result"][0][1][ProfileType.UPPER_ERROR_BAR.value][1][3] for entry in data
]
plt.plot(x, batt, color="C0", label="Battery capacity / kWh")
plt.fill_between(x, ler_batt, uer_batt, color="C0", alpha=ALPHA)
plt.plot(x, pv, color="C1", label="Num. PV collectors")
plt.fill_between(x, ler_pv, uer_pv, color="C1", alpha=ALPHA)
plt.plot(x, pv_t, color="C2", label="Num. PV-T collectors")
plt.fill_between(x, ler_pv_t, uer_pv_t, color="C2", alpha=ALPHA)
plt.plot(x, st, color="C3", label="Num. solar-thermal collectors")
plt.fill_between(x, ler_st, uer_st, color="C3", alpha=ALPHA)
plt.xlabel("Mean grid discount rate / %/year")
plt.xlim(min(x), max(x))
plt.ylabel("Component size")
plt.ylim(0, 1.1 * max(ler_batt + ler_pv + ler_pv_t + ler_st + batt + pv + pv_t + st))
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.savefig(
    f"{fig_identifier}_four_param_component_sizes.png",
    dpi=1200,
    transparent=True,
    bbox_inches="tight",
)
plt.close()

plt.figure()  # figsize=FIGURE_DIMENSIONS)

# Result 1 - no-PV optimisation
sns.set_palette("colorblind")
batt = [entry["result"][1][1][ProfileType.AVERAGE.value][1][0] for entry in data]
ler_batt = [
    entry["result"][1][1][ProfileType.LOWER_ERROR_BAR.value][1][0] for entry in data
]
uer_batt = [
    entry["result"][1][1][ProfileType.UPPER_ERROR_BAR.value][1][0] for entry in data
]
pv_t = [entry["result"][1][1][ProfileType.AVERAGE.value][1][1] for entry in data]
ler_pv_t = [
    entry["result"][1][1][ProfileType.LOWER_ERROR_BAR.value][1][1] for entry in data
]
uer_pv_t = [
    entry["result"][1][1][ProfileType.UPPER_ERROR_BAR.value][1][1] for entry in data
]
st = [entry["result"][1][1][ProfileType.AVERAGE.value][1][2] for entry in data]
ler_st = [
    entry["result"][1][1][ProfileType.LOWER_ERROR_BAR.value][1][2] for entry in data
]
uer_st = [
    entry["result"][1][1][ProfileType.UPPER_ERROR_BAR.value][1][2] for entry in data
]
plt.plot(x, batt, color="C0", label="Battery capacity / kWh")
plt.fill_between(x, ler_batt, uer_batt, color="C0", alpha=ALPHA)
plt.plot(x, pv_t, color="C2", label="Num. PV-T collectors")
plt.fill_between(x, ler_pv_t, uer_pv_t, color="C2", alpha=ALPHA)
plt.plot(x, st, color="C3", label="Num. solar-thermal collectors")
plt.fill_between(x, ler_st, uer_st, color="C3", alpha=ALPHA)
plt.xlabel("Mean grid discount rate / %/year")
plt.xlim(-25, 25)
plt.ylabel("Component size")
plt.ylim(0, 1.25 * max(uer_batt + uer_pv_t + uer_st + batt + pv_t + st))
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.savefig(
    f"{fig_identifier}_no_pv_component_sizes.png",
    dpi=DPI,
    transparent=True,
    bbox_inches="tight",
)
plt.close()

plt.figure()  # figsize=FIGURE_DIMENSIONS)

# Result 2 - no-PV-T optimisation
sns.set_palette("colorblind")
batt = [entry["result"][2][1][ProfileType.AVERAGE.value][1][0] for entry in data]
ler_batt = [
    entry["result"][2][1][ProfileType.LOWER_ERROR_BAR.value][1][0] for entry in data
]
uer_batt = [
    entry["result"][2][1][ProfileType.UPPER_ERROR_BAR.value][1][0] for entry in data
]
pv = [entry["result"][2][1][ProfileType.AVERAGE.value][1][1] for entry in data]
ler_pv = [
    entry["result"][2][1][ProfileType.LOWER_ERROR_BAR.value][1][1] for entry in data
]
uer_pv = [
    entry["result"][2][1][ProfileType.UPPER_ERROR_BAR.value][1][1] for entry in data
]
st = [entry["result"][2][1][ProfileType.AVERAGE.value][1][2] for entry in data]
ler_st = [
    entry["result"][2][1][ProfileType.LOWER_ERROR_BAR.value][1][2] for entry in data
]
uer_st = [
    entry["result"][2][1][ProfileType.UPPER_ERROR_BAR.value][1][2] for entry in data
]
plt.plot(x, batt, color="C0", label="Battery capacity / kWh")
plt.fill_between(x, ler_batt, uer_batt, color="C0", alpha=ALPHA)
plt.plot(x, pv, color="C1", label="Num. PV collectors")
plt.fill_between(x, ler_pv, uer_pv, color="C1", alpha=ALPHA)
plt.plot(x, st, color="C3", label="Num. solar-thermal collectors")
plt.fill_between(x, ler_st, uer_st, color="C3", alpha=ALPHA)
plt.xlabel("Mean grid discount rate / %/year")
plt.xlim(-25, 25)
plt.ylabel("Component size")
plt.ylim(0, 1.25 * max(uer_batt + uer_pv + uer_st + batt + pv + st))
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.savefig(
    f"{fig_identifier}_no_pv_t_component_sizes.png",
    dpi=DPI,
    transparent=True,
    bbox_inches="tight",
)
plt.close()

plt.figure()  # figsize=FIGURE_DIMENSIONS)

# Result 3 - no-st-parameter optimisation
sns.set_palette("colorblind")
batt = [entry["result"][3][1][ProfileType.AVERAGE.value][1][0] for entry in data]
ler_batt = [
    entry["result"][3][1][ProfileType.LOWER_ERROR_BAR.value][1][0] for entry in data
]
uer_batt = [
    entry["result"][3][1][ProfileType.UPPER_ERROR_BAR.value][1][0] for entry in data
]
pv = [entry["result"][3][1][ProfileType.AVERAGE.value][1][1] for entry in data]
ler_pv = [
    entry["result"][3][1][ProfileType.LOWER_ERROR_BAR.value][1][1] for entry in data
]
uer_pv = [
    entry["result"][3][1][ProfileType.UPPER_ERROR_BAR.value][1][1] for entry in data
]
pv_t = [entry["result"][3][1][ProfileType.AVERAGE.value][1][2] for entry in data]
ler_pv_t = [
    entry["result"][3][1][ProfileType.LOWER_ERROR_BAR.value][1][2] for entry in data
]
uer_pv_t = [
    entry["result"][3][1][ProfileType.UPPER_ERROR_BAR.value][1][2] for entry in data
]
plt.plot(x, batt, color="C0", label="Battery capacity / kWh")
plt.fill_between(x, ler_batt, uer_batt, color="C0", alpha=ALPHA)
plt.plot(x, pv, color="C1", label="Num. PV collectors")
plt.fill_between(x, ler_pv, uer_pv, color="C1", alpha=ALPHA)
plt.plot(x, pv_t, color="C2", label="Num. PV-T collectors")
plt.fill_between(x, ler_pv_t, uer_pv_t, color="C2", alpha=ALPHA)
plt.xlabel("Mean grid discount rate / %/year")
plt.xlim(-25, 25)
plt.ylabel("Component size")
plt.ylim(0, 1.25 * max(uer_batt + uer_pv + uer_pv_t + batt + pv + pv_t))
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.savefig(
    f"{fig_identifier}_no_st_component_sizes.png",
    dpi=DPI,
    transparent=True,
    bbox_inches="tight",
)
plt.close()

plt.figure()  # figsize=FIGURE_DIMENSIONS)

# Plot the various keys
for plot_index, title in enumerate(optimisation_titles[:1]):
    for index, key in enumerate(keys):
        y = [
            entry["result"][plot_index][1]["average_weather_conditions"][0][key]
            for entry in data
        ]
        # y_lsd = [
        #     entry["result"][plot_index][1][
        #         "lower_standard_deviation_weather_conditions"
        #     ][0][key]
        #     for entry in data
        # ]
        # y_usd = [
        #     entry["result"][plot_index][1][
        #         "upper_standard_deviation_weather_conditions"
        #     ][0][key]
        #     for entry in data
        # ]
        y_ler = [
            entry["result"][plot_index][1]["lower_error_bar_weather_conditions"][0][key]
            for entry in data
        ]
        y_uer = [
            entry["result"][plot_index][1]["upper_error_bar_weather_conditions"][0][key]
            for entry in data
        ]
        # y_max = [
        #     entry["result"][plot_index][1]["maximum_irradiance_weather_conditions"][0][
        #         key
        #     ]
        #     for entry in data
        # ]
        # y_min = [
        #     min(y_lsd[index], y_usd[index], y_max[index]) for index in range(len(y_max))
        # ]
        # y_max = [
        #     max(y_lsd[index], y_usd[index], y_max[index]) for index in range(len(y_max))
        # ]
        # Determine the x range
        # Plot
        plt.plot(
            x,
            y,
            color=f"C{index}",
            label=f"{key.replace('_', ' ').capitalize()} (unsmoothed)",
        )
        plt.fill_between(x, y_ler, y, color=f"C{index}", alpha=0.5)
        plt.fill_between(x, y, y_uer, color=f"C{index}", alpha=0.5)
        # plt.plot(x, y_min, "--", color=f"C{index}")
        # plt.plot(x, y_max, "--", color=f"C{index}")
        plt.plot(x, y_uer, "--", color=f"C{index}")
        plt.plot(x, y_ler, "--", color=f"C{index}")
        plt.ylabel(key.replace("_", " ").capitalize())
        plt.xlabel("Mean grid discount rate / %/year")
        plt.xlim(-25, 25)
        plt.ylim(0, 1.5 * max(y))
        plt.title(
            f"{key.replace('_', ' ').capitalize()} (unsmoothed) for {title.capitalize()}"
        )
        plt.legend(bbox_to_anchor=(1.0, 1.0))
        plt.savefig(
            f"{fig_identifier}_{key}_unsmoothed_{title}.png",
            dpi=DPI,
            transparent=True,
            bbox_inches="tight",
        )
        plt.close()
        plt.figure()  # figsize=FIGURE_DIMENSIONS)
        plt.plot(
            x,
            lfilter(b, a, y),
            color=f"C{index}",
            label=f"{key.replace('_', ' ').capitalize()} (smoothed)",
        )
        plt.fill_between(
            x, lfilter(b, a, y_ler), lfilter(b, a, y), color=f"C{index}", alpha=0.5
        )
        plt.fill_between(
            x, lfilter(b, a, y), lfilter(b, a, y_uer), color=f"C{index}", alpha=0.5
        )
        # plt.plot(x, y_min, "--", color=f"C{index}")
        # plt.plot(x, y_max, "--", color=f"C{index}")
        plt.plot(x, lfilter(b, a, y_uer), "--", color=f"C{index}")
        plt.plot(x, lfilter(b, a, y_ler), "--", color=f"C{index}")
        plt.ylabel(key.replace("_", " ").capitalize())
        plt.xlabel("Mean grid discount rate / %/year")
        plt.xlim(-25, 25)
        plt.ylim(0, 1.5 * max(y))
        plt.title(
            f"{key.replace('_', ' ').capitalize()} (smoothed) for {title.capitalize()}"
        )
        plt.legend(bbox_to_anchor=(1.0, 1.0))
        plt.savefig(
            f"{fig_identifier}_{key}_smoothed_{title}.png",
            dpi=DPI,
            transparent=True,
            bbox_inches="tight",
        )
        plt.close()
        plt.figure()  # figsize=FIGURE_DIMENSIONS)
        with open(
            f"grid_high_res_weather_error_{key}.json", "w", encoding="UTF-8"
        ) as f:
            json.dump(
                {
                    "x": x,
                    key: y,
                    f"{key}_uer": y_uer,
                    f"{key}_ler": y_ler,
                    # f"{key}_usd": y_usd,
                    # f"{key}_lsd": y_lsd,
                    # f"{key}_max": y_max,
                    # f"{key}_min": y_min,
                },
                f,
            )
    # Plot the fractions of power from storage, pv, and the grid.
    sns.set_palette("PuBu_r", n_colors=3)
    storage_fraction = [
        entry["result"][plot_index][1]["average_weather_conditions"][0][
            "storage_electricity_fraction"
        ]
        for entry in data
    ]
    solar_fraction = [
        entry["result"][plot_index][1]["average_weather_conditions"][0][
            "solar_electricity_fraction"
        ]
        for entry in data
    ]
    grid_fraction = [
        entry["result"][plot_index][1]["average_weather_conditions"][0][
            "grid_electricity_fraction"
        ]
        for entry in data
    ]
    plt.plot(
        x,
        (
            grid_line := [
                solar_fraction[index] + grid_fraction[index] + storage_fraction[index]
                for index in range(len(storage_fraction))
            ]
        ),
        color=f"C2",
        label="grid fraction",
    )
    plt.plot(
        x,
        (
            solar_line := [
                storage_fraction[index] + solar_fraction[index]
                for index in range(len(storage_fraction))
            ]
        ),
        color=f"C1",
        label="solar fraction",
    )
    plt.plot(x, storage_fraction, color=f"C0", label="storage fraction")
    plt.fill_between(
        x, [0] * len(storage_fraction), storage_fraction, color="C0", alpha=0.7
    )
    plt.fill_between(x, storage_fraction, solar_line, color="C1", alpha=0.7)
    plt.fill_between(x, solar_line, grid_line, color="C2", alpha=0.7)
    plt.xlabel("Mean grid discount rate / %/year")
    plt.ylabel("Fractional generation of electricity demand")
    plt.title(f"Fractional electricity sources for {title.capitalize()}")
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.xlim(-25, 25)
    plt.savefig(
        f"{fig_identifier}_{key}_unsmoothed_electricity_sources.png",
        dpi=DPI,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close()
    plt.figure()  # figsize=FIGURE_DIMENSIONS)
    # Plot the smoothed fractions of power from storage, pv, and the grid.
    sns.set_palette("PuBu_r", n_colors=3)
    storage_fraction = lfilter(
        b,
        a,
        [
            entry["result"][plot_index][1]["average_weather_conditions"][0][
                "storage_electricity_fraction"
            ]
            for entry in data
        ],
    )
    solar_fraction = lfilter(
        b,
        a,
        [
            entry["result"][plot_index][1]["average_weather_conditions"][0][
                "solar_electricity_fraction"
            ]
            for entry in data
        ],
    )
    grid_fraction = lfilter(
        b,
        a,
        [
            entry["result"][plot_index][1]["average_weather_conditions"][0][
                "grid_electricity_fraction"
            ]
            for entry in data
        ],
    )
    plt.plot(
        x,
        (
            grid_line := [
                solar_fraction[index] + grid_fraction[index] + storage_fraction[index]
                for index in range(len(storage_fraction))
            ]
        ),
        color=f"C2",
        label="grid fraction",
    )
    plt.plot(
        x,
        (
            solar_line := [
                storage_fraction[index] + solar_fraction[index]
                for index in range(len(storage_fraction))
            ]
        ),
        color=f"C1",
        label="solar fraction",
    )
    plt.plot(x, storage_fraction, color=f"C0", label="storage fraction")
    plt.fill_between(
        x, [0] * len(storage_fraction), storage_fraction, color="C0", alpha=0.7
    )
    plt.fill_between(x, storage_fraction, solar_line, color="C1", alpha=0.7)
    plt.fill_between(x, solar_line, grid_line, color="C2", alpha=0.7)
    plt.xlabel("Mean grid discount rate / %/year")
    plt.ylabel("Fractional generation of electricity demand")
    plt.title(f"Fractional electricity sources for {title.capitalize()}")
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.xlim(-25, 25)
    plt.savefig(
        f"{fig_identifier}_{key}_smoothed_electricity_sources.png",
        dpi=DPI,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close()
    plt.figure()  # figsize=FIGURE_DIMENSIONS)
    storage_fraction = [
        entry["result"][plot_index][1]["average_weather_conditions"][0][
            "storage_electricity_fraction"
        ]
        for entry in data
    ]
    solar_fraction = [
        entry["result"][plot_index][1]["average_weather_conditions"][0][
            "solar_electricity_fraction"
        ]
        for entry in data
    ]
    grid_fraction = [
        entry["result"][plot_index][1]["average_weather_conditions"][0][
            "grid_electricity_fraction"
        ]
        for entry in data
    ]
    plt.plot(x, (grid_line := grid_fraction), color=f"C2", label="grid fraction")
    plt.plot(x, (solar_line := solar_fraction), color=f"C1", label="solar fraction")
    plt.plot(x, storage_fraction, color=f"C0", label="storage fraction")
    plt.fill_between(
        x, [0] * len(storage_fraction), storage_fraction, color="C0", alpha=0.7
    )
    plt.fill_between(x, storage_fraction, solar_line, color="C1", alpha=0.7)
    plt.fill_between(x, solar_line, grid_line, color="C2", alpha=0.7)
    plt.xlabel("Mean grid discount rate / %/year")
    plt.ylabel("Fractional generation of electricity demand")
    plt.title(f"Fractional electricity sources for {title.capitalize()}")
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.xlim(-25, 25)
    plt.savefig(
        f"{fig_identifier}_{key}_overlapping_electricity_sources.png",
        dpi=DPI,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close()
    plt.figure()  # figsize=FIGURE_DIMENSIONS)

################################
# PV degradation probe results #
################################

import json
import matplotlib.pyplot as plt
import os
import seaborn as sns

sns.set_palette("colorblind")

with open("hpc_pv_degradation_optimisations_probe.json", "r", encoding="UTF-8") as f:
    data = json.load(f)

keys = [
    "total_cost",
    "auxiliary_heating_fraction",
    "dumped_electricity",
    "grid_electricity_fraction",
    "solar_electricity_fraction",
    "storage_electricity_fraction",
]

optimisation_titles = [
    "Four parameter",
    "No-PV optimisation (PV-T, ST, Storage)",
    "No-PV-T optimisation (PV, ST, Storage)",
    "No-ST optimisation (PV, PV-T, Storage)",
]

x = [entry["optimisation"]["scenario"] for entry in data]
x = [entry.split("deg_")[1].split("%")[0] for entry in x]
x = [float(entry) for entry in x]

# Plot the various keys
for plot_index, title in enumerate(optimisation_titles):
    sns.set_palette("colorblind")
    for index, key in enumerate(keys):
        y = [
            entry["result"][plot_index][1]["average_weather_conditions"][0][key]
            for entry in data
        ]
        # y_lsd = [
        #     entry["result"][plot_index][1][
        #         "lower_standard_deviation_weather_conditions"
        #     ][0][key]
        #     for entry in data
        # ]
        # y_usd = [
        #     entry["result"][plot_index][1][
        #         "upper_standard_deviation_weather_conditions"
        #     ][0][key]
        #     for entry in data
        # ]
        y_ler = [
            entry["result"][plot_index][1]["lower_error_bar_weather_conditions"][0][key]
            for entry in data
        ]
        y_uer = [
            entry["result"][plot_index][1]["upper_error_bar_weather_conditions"][0][key]
            for entry in data
        ]
        # y_max = [
        #     entry["result"][plot_index][1]["maximum_irradiance_weather_conditions"][0][
        #         key
        #     ]
        #     for entry in data
        # ]
        # y_min = [
        #     min(y_lsd[index], y_usd[index], y_max[index]) for index in range(len(y_max))
        # ]
        # y_max = [
        #     max(y_lsd[index], y_usd[index], y_max[index]) for index in range(len(y_max))
        # ]
        # Determine the x range
        # Plot
        # Sort the lists
        sorted_x, sorted_y, sorted_y_ler, sorted_y_uer = zip(
            *sorted(zip(x, y, y_ler, y_uer))
        )
        plt.plot(sorted_x, sorted_y, color=f"C{index}")
        plt.fill_between(sorted_x, sorted_y_ler, sorted_y, color=f"C{index}", alpha=0.5)
        plt.fill_between(sorted_x, sorted_y, sorted_y_uer, color=f"C{index}", alpha=0.5)
        # plt.plot(x, y_min, "--", color=f"C{index}")
        # plt.plot(x, y_max, "--", color=f"C{index}")
        plt.plot(sorted_x, sorted_y_uer, "--", color=f"C{index}")
        plt.plot(sorted_x, sorted_y_ler, "--", color=f"C{index}")
        plt.ylabel(key.replace("_", " ").capitalize())
        plt.xlabel("PV degradation rate / %/year")
        plt.title(f"{key.replace('_', ' ').capitalize()} for {title.capitalize()}")
        plt.show()
        with open(f"pv_degradation_rate_{key}.json", "w", encoding="UTF-8") as f:
            json.dump(
                {
                    "x": sorted_x,
                    key: sorted_y,
                    f"{key}_uer": sorted_y_uer,
                    f"{key}_ler": sorted_y_ler,
                    # f"{key}_usd": y_usd,
                    # f"{key}_lsd": y_lsd,
                    # f"{key}_max": y_max,
                    # f"{key}_min": y_min,
                },
                f,
            )
    # Plot the fractions of power from storage, pv, and the grid.
    sns.set_palette("PuBu_r", n_colors=3)
    storage_fraction = [
        entry["result"][plot_index][1]["average_weather_conditions"][0][
            "storage_electricity_fraction"
        ]
        for entry in data
    ]
    solar_fraction = [
        entry["result"][plot_index][1]["average_weather_conditions"][0][
            "solar_electricity_fraction"
        ]
        for entry in data
    ]
    grid_fraction = [
        entry["result"][plot_index][1]["average_weather_conditions"][0][
            "grid_electricity_fraction"
        ]
        for entry in data
    ]
    sorted_x, storage_fraction, solar_fraction, grid_fraction = zip(
        *sorted(zip(x, storage_fraction, solar_fraction, grid_fraction))
    )
    plt.plot(
        sorted_x,
        (
            grid_line := [
                solar_fraction[index] + grid_fraction[index] + storage_fraction[index]
                for index in range(len(storage_fraction))
            ]
        ),
        color=f"C2",
        label="grid fraction",
    )
    plt.plot(
        sorted_x,
        (
            solar_line := [
                storage_fraction[index] + solar_fraction[index]
                for index in range(len(storage_fraction))
            ]
        ),
        color=f"C1",
        label="solar fraction",
    )
    plt.plot(sorted_x, storage_fraction, color=f"C0", label="storage fraction")
    plt.fill_between(
        sorted_x, [0] * len(storage_fraction), storage_fraction, color="C0", alpha=0.7
    )
    plt.fill_between(sorted_x, storage_fraction, solar_line, color="C1", alpha=0.7)
    plt.fill_between(sorted_x, solar_line, grid_line, color="C2", alpha=0.7)
    plt.xlabel("Mean grid discount rate / %/year")
    plt.ylabel("Fractional generation of electricity demand")
    plt.title(f"Fractional electricity sources for {title.capitalize()}")
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.show()
    # Plot without raised minima
    plt.plot(sorted_x, (grid_line := grid_fraction), color=f"C2", label="grid fraction")
    plt.plot(
        sorted_x, (solar_line := solar_fraction), color=f"C1", label="solar fraction"
    )
    plt.plot(sorted_x, storage_fraction, color=f"C0", label="storage fraction")
    plt.fill_between(
        sorted_x, [0] * len(storage_fraction), storage_fraction, color="C0", alpha=0.7
    )
    plt.fill_between(sorted_x, storage_fraction, solar_line, color="C1", alpha=0.7)
    plt.fill_between(sorted_x, solar_line, grid_line, color="C2", alpha=0.7)
    plt.xlabel("PV degradation rate / %/year")
    plt.ylabel("Fractional generation of electricity demand")
    plt.title(f"Fractional electricity sources for {title.capitalize()}")
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.show()


#######################
# Scenario-generation #
#######################

import json
import numpy as np
import os
import yaml

from tqdm import tqdm

from src.heatdesalination.__utils__ import GridScheme

with open(
    (scenarios_filepath := os.path.join("inputs", "scenarios.json")),
    "r",
    encoding="UTF-8",
) as f:
    scenarios = yaml.safe_load(f)[(scenarios_key := "scenarios")]

DEFAULT_SCENARIO = scenarios[0]

DEFAULT_OPTIMISATION = {
    (location := "location"): "fujairah_united_arab_emirates",
    (output := "output"): "parallel_optimisation_output",
    "profile_types": ["avr", "ler", "uer"],
    "run_type": "optimisation",
    (scenario := "scenario"): "default",
    "system_lifetime": 30,
}

# Grid cost schemes
GRID_COST_SCHEMES = {
    (abu_dhabi := "abu_dhabi"): GridScheme.ABU_DHABI_UAE.value,
    (gran_canaria := "gran_canaria"): GridScheme.GRAN_CANARIA_SPAIN.value,
    (la_paz := "la_paz"): GridScheme.LA_PAZ_MEXICO.value,
    (tijuana := "tijuana"): GridScheme.TIJUANA_MEXICO.value,
}

# The mapping between location name and the corresponding filename
LOCATIONS = {
    abu_dhabi: "sas_al_nakhl_united_arab_emirates",
    gran_canaria: "gando_gran_canaria_spain",
    la_paz: "la_paz_mexico",
    tijuana: "municipio_de_tijuana_mexico",
}

# The list of plants
PLANTS: list[str] = ["joo_med_24_hour", "el_nashar_24_hour", "rahimi_24_hour"]

# lists of various collectors
PV_PANELS: list[str] = ["rec_325_n_peak_mono", "sharp_nd_af"]
PV_T_PANELS: list[str] = [
    "dualsun_spring_300m_insulated",
    "dualsun_spring_400_insulated",
    "solimpeks_powervolt",
]
ST_COLLECTORS: list[str] = [
    "sti_fkf_240_cucu",
    "eurotherm_solar_pro_20r",
    "augusta_solar_as_100_df_6",
]

# Set up a list of new scenarios for each location
new_scenarios: list[dict[str, str]] = []
new_optimisations: list[dict[str, str]] = []

# Keyword arguments for changing parameters in the scenarios.
fractional_battery_cost_change = "fractional_battery_cost_change"
fractional_grid_cost_change = "fractional_grid_cost_change"
fractional_heat_pump_cost_change = "fractional_heat_pump_cost_change"
fractional_hw_tank_cost_change = "fractional_hw_tank_cost_change"
fractional_inverter_cost_change = "fractional_inverter_cost_change"
fractional_pv_cost_change = "fractional_pv_cost_change"
fractional_pvt_cost_change = "fractional_pvt_cost_change"
fractional_st_cost_change = "fractional_st_cost_change"
grid_cost_scheme = "grid_cost_scheme"
heat_exchanger_efficiency = "heat_exchanger_efficiency"
heat_pump = "heat_pump"
hot_water_tank = "hot_water_tank"
htf_heat_capacity = "htf_heat_capacity"
inverter_cost = "inverter_cost"
inverter_lifetime = "inverter_lifetime"
name = "name"
plant_kwarg = "plant"
pv = "pv"
pv_degradation_rate = "pv_degradation_rate"
pv_t = "pv_t"
solar_thermal = "solar_thermal"

for location_name, location_filename in tqdm(LOCATIONS.items(), desc="locations"):
    # Cycle through the plants
    for plant in tqdm(PLANTS, desc="plants", leave=False):
        # Cycle through the PV options available
        for pv_panel in tqdm(PV_PANELS, desc="pv panels", leave=False):
            # Cycle through the PV-T options available
            for pv_t_panel in PV_T_PANELS:
                for st_panel in ST_COLLECTORS:
                    scenario = DEFAULT_SCENARIO.copy()
                    # Change the grid-cost scheme to match the scenario.
                    scenario[grid_cost_scheme] = GRID_COST_SCHEMES[location_name]
                    scenario[solar_thermal] = st_panel
                    scenario[plant_kwarg] = plant
                    scenario[pv] = pv_panel
                    scenario[pv_t] = pv_t_panel
                    scenario_name = (
                        f"{location_name}_"
                        f"{plant.split('_')[0]}_"
                        f"{pv_panel.split('_')[0]}_"
                        f"{pv_t_panel.split('m')[0].split('_')[-1]}_"
                        f"{st_panel.split('_')[0]}"
                    )
                    scenario[name] = scenario_name
                    new_scenarios.append(scenario)
                    optimisation = DEFAULT_OPTIMISATION.copy()
                    optimisation["scenario"] = scenario_name
                    optimisation["location"] = LOCATIONS[location_name]
                    optimisation["output"] = scenario_name
                    new_optimisations.append(optimisation)

with open(
    os.path.join("inputs", "ecos_optimisations_07_mar_23.json"), "w", encoding="UTF-8"
) as f:
    json.dump(new_optimisations, f)

with open(os.path.join("inputs", "scenarios.json"), "w", encoding="UTF-8") as f:
    json.dump({"scenarios": new_scenarios}, f)


# Code for generating ECOS sensitivity anslysis
import numpy as np

new_scenarios = []
new_optimisations = []

FRACTIONAL_CHANGES = [
    "fractional_battery_cost_change",
    "fractional_grid_cost_change",
    "fractional_heat_pump_cost_change",
    "fractional_pv_cost_change",
    "fractional_pvt_cost_change",
    "fractional_st_cost_change",
]

RANGE: float = 0.4

DEFAULT_SCENARIO = {
    "battery": "renogy_12v_100ah",
    "fractional_battery_cost_change": 0.0,
    "fractional_grid_cost_change": 0.0,
    "fractional_heat_pump_cost_change": 0.0,
    "fractional_hw_tank_cost_change": 0.0,
    "fractional_inverter_cost_change": 0.0,
    "fractional_pv_cost_change": 0.0,
    "fractional_pvt_cost_change": 0.0,
    "fractional_st_cost_change": 0.0,
    "grid_cost_scheme": "abu_dhabi_uae",
    "heat_exchanger_efficiency": 0.4,
    "heat_pump": "ammonia",
    "hot_water_tank": "grant_hpmono_ind150",
    "htf_heat_capacity": 4182.0,
    "inverter_cost": 148.5,
    "inverter_lifetime": 13,
    "name": "abu_dhabi_joo_sharp_300_sti",
    "plant": "joo_med_24_hour",
    "pv": "sharp_nd_af",
    "pv_degradation_rate": 0.011,
    "pv_t": "dualsun_spring_400_insulated",
    "solar_thermal": "sti_fkf_240_cucu",
    "water_pump": "crnf_15_1_a_ca_a_e_hqqe",
    "fractional_water_pump_cost_change": 0.0,
}

# Iterate through these variables and define values from -0.2 to 0.2.
for fractional_change_name in tqdm(FRACTIONAL_CHANGES, desc="fractional_changes"):
    for change_value in np.linspace(-RANGE, RANGE, 81):
        for location_name, location_filename in tqdm(
            LOCATIONS.items(), desc="locations", leave=False
        ):
            # Cycle through the plants
            for plant in tqdm(PLANTS, desc="plants", leave=False):
                # Cycle through the PV-T options available
                scenario = DEFAULT_SCENARIO.copy()
                # Change the grid-cost scheme to match the scenario.
                scenario[grid_cost_scheme] = GRID_COST_SCHEMES[location_name]
                scenario[plant_kwarg] = plant
                scenario_name = (
                    f"{location_name}_"
                    f"{plant.split('_')[0]}_"
                    f"sharp_400_sti_"
                    f"frac_{fractional_change_name.split('_')[1]}_"
                    f"{round(change_value, 3)}"
                )
                scenario[name] = scenario_name
                scenario[fractional_change_name] = change_value
                new_scenarios.append(scenario)
                optimisation = DEFAULT_OPTIMISATION.copy()
                optimisation["scenario"] = scenario_name
                optimisation["location"] = LOCATIONS[location_name]
                optimisation["output"] = scenario_name
                new_optimisations.append(optimisation)

INVERTER_LIFETIMES = range(1, 31, 2)
BATTERY_NAMES = {
    f"renogy_12v_100ah_{cycles}_cycles" for cycles in range(100, 3100, 100)
}

for inverter_lifetime in tqdm(
    INVERTER_LIFETIMES, desc="inverter lifetimes", leave=True
):
    for cycles, battery_name in tqdm(
        enumerate(BATTERY_NAMES), desc="battery lifetimes", leave=False
    ):
        for location_name, location_filename in tqdm(
            LOCATIONS.items(), desc="locations", leave=False
        ):
            # Cycle through the plants
            for plant in tqdm(PLANTS, desc="plants", leave=False):
                scenario = DEFAULT_SCENARIO.copy()
                scenario["battery"] = battery_name
                scenario["inverter_lifetime"] = inverter_lifetime
                scenario_name = (
                    f"{location_name}_"
                    f"{plant.split('_')[0]}_"
                    f"sharp_400_sti_"
                    f"{cycles*100}_batt_cycles_"
                    f"{inverter_lifetime}_inverter_years"
                )
                scenario[name] = scenario_name
                new_scenarios.append(scenario)
                optimisation = DEFAULT_OPTIMISATION.copy()
                optimisation["scenario"] = scenario_name
                optimisation["location"] = LOCATIONS[location_name]
                optimisation["output"] = scenario_name
                new_optimisations.append(optimisation)


grid_optimisations = []

for discount_rate in np.linspace(-25, 25, 250):
    scenario = default_scenario.copy()
    scenario["discount_rate"] = float(discount_rate / 100)
    scenario["name"] = (
        f"uae_dr_{'m_' if discount_rate < 0 else ''}"
        f"{int(round(abs(discount_rate) // 1, 0))}_"
        f"{int(round((abs(discount_rate) % 1) // 0.1, 0))}"
        f"{int(round((abs(discount_rate) % 1) % 0.1, 0))}"
    )
    new_scenarios.append(scenario)
    optimisation = default_optimisation.copy()
    optimisation["scenario"] = scenario["name"]
    grid_optimisations.append(optimisation)

cheap_pv_t_grid_optimisations = []

for discount_rate in np.linspace(-25, 25, 250):
    scenario = default_scenario.copy()
    scenario["discount_rate"] = float(discount_rate / 100)
    scenario["pv_t"] = "dualsun_spring_insulated_discounted"
    scenario["name"] = (
        f"uae_cheap_pv_t_dr_{'m_' if discount_rate < 0 else ''}"
        f"{int(round(abs(discount_rate) // 1, 0))}_"
        f"{int(round((abs(discount_rate) % 1) // 0.1, 0))}"
        f"{int(round((abs(discount_rate) % 1) % 0.1, 0))}"
    )
    new_scenarios.append(scenario)
    optimisation = default_optimisation.copy()
    optimisation["scenario"] = scenario["name"]
    cheap_pv_t_grid_optimisations.append(optimisation)


pv_degradation_optimisations = []

for pv_degradation in np.linspace(0.007, 0.039, 250):
    scenario = default_scenario.copy()
    scenario["pv_degradation_rate"] = float(pv_degradation)
    scenario[
        "name"
    ] = f"uae_pv_deg_{int(round(100*pv_degradation//1,0))}_{int(round(100*round(100*pv_degradation%1,3), 0))}%"
    new_scenarios.append(scenario)
    optimisation = default_optimisation.copy()
    optimisation["scenario"] = scenario["name"]
    pv_degradation_optimisations.append(optimisation)


heat_exchanger_efficiency_optimisations = []

for heat_exchanger_efficiency in np.linspace(0.01, 1, 100):
    scenario = default_scenario.copy()
    scenario["heat_exchanger_efficiency"] = float(heat_exchanger_efficiency)
    scenario[
        "name"
    ] = f"uae_heat_ex_eff_{int(round(100*heat_exchanger_efficiency//1,0))}.{int(round(100*round(100*heat_exchanger_efficiency%1,3), 0))}%"
    new_scenarios.append(scenario)
    optimisation = default_optimisation.copy()
    optimisation["scenario"] = scenario["name"]
    heat_exchanger_efficiency_optimisations.append(optimisation)

heat_pump_efficiency_optimisations = []

for heat_pump_efficiency in np.linspace(0.01, 1, 100):
    scenario = default_scenario.copy()
    scenario["heat_pump_efficiency"] = float(heat_pump_efficiency)
    scenario[
        "name"
    ] = f"uae_heat_pump_eff_{int(round(100*heat_pump_efficiency//1,0))}.{int(round(100*round(100*heat_pump_efficiency%1,3), 0))}%"
    new_scenarios.append(scenario)
    optimisation = default_optimisation.copy()
    optimisation["scenario"] = scenario["name"]
    heat_pump_efficiency_optimisations.append(optimisation)


with open(os.path.join("inputs", "scenarios.json"), "w", encoding="UTF-8") as f:
    json.dump({"scenarios": new_scenarios}, f)

with open(
    os.path.join("inputs", "optimisations_pv_degradation.json"), "w", encoding="UTF-8"
) as f:
    json.dump(pv_degradation_optimisations, f)

with open(
    os.path.join("inputs", "optimisations_heat_exchanger_efficiency.json"),
    "w",
    encoding="UTF-8",
) as f:
    json.dump(heat_exchanger_efficiency_optimisations, f)

with open(
    os.path.join("inputs", "optimisations_heat_pump_efficiency.json"),
    "w",
    encoding="UTF-8",
) as f:
    json.dump(heat_pump_efficiency_optimisations, f)

with open(
    os.path.join("inputs", "grid_optimisations.json"), "w", encoding="UTF-8"
) as f:
    json.dump(grid_optimisations, f)

with open(
    os.path.join("inputs", "cheap_pv_t_grid_optimisations.json"),
    "w",
    encoding="UTF-8",
) as f:
    json.dump(cheap_pv_t_grid_optimisations, f)

##############################################
# Plotting collector performance information #
##############################################

plt.scatter(
    [
        entry[(THERMAL_PERFORMANCE_CURVE := "thermal_performance_curve")][
            (ZEROTH_ORDER := "zeroth_order")
        ]
        for entry in fpc_data
    ],
    [
        entry[THERMAL_PERFORMANCE_CURVE][(FIRST_ORDER := "first_order")]
        for entry in fpc_data
    ],
    marker="x",
    label="flat-plate",
)
plt.scatter(
    [
        entry[(THERMAL_PERFORMANCE_CURVE := "thermal_performance_curve")][
            (ZEROTH_ORDER := "zeroth_order")
        ]
        for entry in etc_data
    ],
    [
        entry[THERMAL_PERFORMANCE_CURVE][(FIRST_ORDER := "first_order")]
        for entry in etc_data
    ],
    marker="x",
    label="evacuated-tube",
)
plt.scatter(
    mean_fpc[THERMAL_PERFORMANCE_CURVE][ZEROTH_ORDER],
    mean_fpc[THERMAL_PERFORMANCE_CURVE][FIRST_ORDER],
    marker="o",
    color="C0",
    label="mean flat-plate",
)
plt.scatter(
    mean_etc[THERMAL_PERFORMANCE_CURVE][ZEROTH_ORDER],
    mean_etc[THERMAL_PERFORMANCE_CURVE][FIRST_ORDER],
    marker="o",
    color="C1",
    label="mean evacuated-tube",
)
plt.legend()
plt.xlabel("Zeroth-order performance curve coefficient")
plt.ylabel("First-order performance curve coefficient")
plt.show()
plt.savefig(
    "solar_thermal_parameter_comparison.png",
    dpi=1200,
    transparent=True,
    bbox_inches="tight",
)

################################
# NPJ Clean Water HPC Analysis #
################################

import enum
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns

from brokenaxes import brokenaxes
from matplotlib import rc
from matplotlib import ticker
from typing import Any


rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"]})
sns.set_context("paper")
# sns.color_palette("colorblind")
sns.set_style("whitegrid")

# Set custom color-blind colormap

R = [224, 240, 231, 82, 0, 237]
G = [70, 159, 223, 192, 98, 237]
B = [6, 82, 190, 173, 100, 237]

colorblind_palette = sns.color_palette(
    [
        "#E04606",  # Orange
        "#F09F52",  # Pale orange
        "#52C0AD",  # Pale green
        "#006264",  # Green
        "#D8247C",  # Pink
        "#EDEDED",  # Pale pink
        "#E7DFBE",  # Pale yellow
        "#FBBB2C",  # Yellow
    ]
)

sns.set_palette(colorblind_palette)

# Read input data
with open("28_mar_23.json", "r", encoding="UTF-8") as f:
    full_data = json.load(f)

data = {
    key: value
    for key, value in full_data.items()
    if "lg" not in key and "frac" not in key and "batt" not in key
}


# Define helper functions
def _process_data(
    data_to_process: pd.DataFrame,
    key_number: int,
    weather_type: str = "average_weather_conditions",
) -> list[float]:
    processed_data: list[float] = []
    for entry in data_to_process.values():
        processed_data.extend(
            [sub_entry[1][weather_type][1][key_number] for sub_entry in entry]
        )
    return processed_data


def battery_capacities(
    data_to_process, weather_type: str = "average_weather_conditions"
):
    return _process_data(data_to_process, 0, weather_type)


def tank_capacities(
    data_to_process: dict[str, Any], weather_type: str = "average_weather_conditions"
):
    return _process_data(data_to_process, 1, weather_type)


def mass_flow_rates(
    data_to_process: dict[str, Any], weather_type: str = "average_weather_conditions"
):
    return _process_data(data_to_process, 2, weather_type)


def pv_sizes(
    data_to_process: dict[str, Any], weather_type: str = "average_weather_conditions"
):
    return _process_data(data_to_process, 3, weather_type)


def pv_t_sizes(
    data_to_process: dict[str, Any], weather_type: str = "average_weather_conditions"
):
    return _process_data(data_to_process, 4, weather_type)


def st_sizes(
    data_to_process: dict[str, Any], weather_type: str = "average_weather_conditions"
):
    return _process_data(data_to_process, 5, weather_type)


def _dual_300_data(data_to_process: dict[str, Any]):
    return {
        key: (value if "300" in key else None) for key, value in data_to_process.items()
    }


def _dual_400_data(data_to_process: dict[str, Any]):
    return {
        key: (value if "400" in key else None) for key, value in data_to_process.items()
    }


def _soli_data(data_to_process: dict[str, Any]):
    return {
        key: (value if "soli" in key else None)
        for key, value in data_to_process.items()
    }


def _fpc_data(data_to_process: dict[str, Any]):
    return {
        key: (value if "sti" in key else None) for key, value in data_to_process.items()
    }


def _etc_euro_data(data_to_process: dict[str, Any]):
    return {
        key: (value if "eurotherm" in key else None)
        for key, value in data_to_process.items()
    }


def _etc_aug_data(data_to_process: dict[str, Any]):
    return {
        key: (value if "augusta" in key else None)
        for key, value in data_to_process.items()
    }


# def _m_si_data(data_to_process: dict[str, Any]):
#     return {key: (value if "lg" in key else None) for key, value in data_to_process.items()}


def _m_si_data(data_to_process: dict[str, Any]):
    return {
        key: (value if "rec" in key else None) for key, value in data_to_process.items()
    }


def _p_si_data(data_to_process: dict[str, Any]):
    return {
        key: (value if "sharp" in key else None)
        for key, value in data_to_process.items()
    }


def scenarios(data_to_process: dict[str, Any]) -> list[str]:
    scenarios = []
    for entry in data_to_process.keys():
        scenarios.extend(3 * [entry.split(".json")[0]])
    return scenarios


def hist_plot(data_to_plot: Any, label: str, legend_label: str | None = None):
    sns.histplot(
        data_to_plot,
        x=label,
        label=(legend_label if legend_label is not None else label).capitalize(),
    )
    plt.xlabel("Number of components installed")
    plt.ylabel("Optimisation scenarios")


# Bubble plot
def frame(data_to_frame):
    return pd.DataFrame(
        {
            (scenario_key := "scenario"): scenarios(data_to_frame),
            "battery": battery_capacities(data_to_frame),
            "tank": tank_capacities(data_to_frame),
            "mass_flow_rate": mass_flow_rates(data_to_frame),
            "pv": pv_sizes(data_to_frame),
            "pv_t": pv_t_sizes(data_to_frame),
            "st": st_sizes(data_to_frame),
        },
    ).set_index(scenario_key)


abu_dhabi_data = {key: value for key, value in data.items() if "abu_dhabi" in key}
abu_dhabi_joo = {key: value for key, value in abu_dhabi_data.items() if "_joo_" in key}
abu_dhabi_rahimi = {
    key: value for key, value in abu_dhabi_data.items() if "_rahimi_" in key
}
abu_dhabi_el = {key: value for key, value in abu_dhabi_data.items() if "_el_" in key}

gran_canaria_data = {key: value for key, value in data.items() if "gran_canaria" in key}
gran_canaria_joo = {
    key: value for key, value in gran_canaria_data.items() if "_joo_" in key
}
gran_canaria_rahimi = {
    key: value for key, value in gran_canaria_data.items() if "_rahimi_" in key
}
gran_canaria_el = {
    key: value for key, value in gran_canaria_data.items() if "_el_" in key
}

la_paz_data = {key: value for key, value in data.items() if "la_paz" in key}
la_paz_joo = {key: value for key, value in la_paz_data.items() if "_joo_" in key}
la_paz_rahimi = {key: value for key, value in la_paz_data.items() if "_rahimi_" in key}
la_paz_el = {key: value for key, value in la_paz_data.items() if "_el_" in key}

tijuana_data = {key: value for key, value in data.items() if "tijuana" in key}
tijuana_joo = {key: value for key, value in tijuana_data.items() if "_joo_" in key}
tijuana_rahimi = {
    key: value for key, value in tijuana_data.items() if "_rahimi_" in key
}
tijuana_el = {key: value for key, value in tijuana_data.items() if "_el_" in key}

joo_data = {key: value for key, value in data.items() if "_joo_" in key}
el_data = {key: value for key, value in data.items() if "_el_" in key}
rahimi_data = {key: value for key, value in data.items() if "_rahimi_" in key}


def solar_hist(data_to_plot):
    _, ax = plt.subplots()
    hist_plot(frame(data_to_plot), "pv", "PV")
    hist_plot(frame(data_to_plot), "pv_t", "PV-T")
    hist_plot(frame(data_to_plot), "st", "Solar-thermal")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[1::2], labels[1::2], bbox_to_anchor=(1.0, 1.0))


# Boxen plot
def _scenario_match(key: str, tank_index: int) -> bool:
    """Responsible for determining whether a result has the correct tank index."""
    if "joo" in key:
        return tank_index == 0
    if "rahimi" in key:
        return tank_index == 2
    return tank_index == 1


def _helper_value(
    data_to_process,
    tank_index: int | None,
    type_number: int,
    variable: str,
    weather_type: str,
):
    if tank_index is not None:
        return [
            (
                entry[tank_index][1][weather_type][type_number][variable]
                if entry is not None
                else None
            )
            for key, entry in data_to_process.items()
            if _scenario_match(key, tank_index)
        ]
    data_to_return = []
    for index in range(3):
        data_to_return.extend(
            [
                (
                    entry[index][1][weather_type][type_number][variable]
                    if entry is not None
                    else None
                )
                for sub_key, entry in data_to_process.items()
                if _scenario_match(sub_key, index)
            ]
        )
    return data_to_return


def _results_value(
    data_to_boxen, tank_index: int | None, variable: str, weather_type: str
):
    return _helper_value(data_to_boxen, tank_index, 0, variable, weather_type)


def _component_value(
    data_to_boxen, tank_index: int | None, variable: str, weather_type: str
):
    return _helper_value(data_to_boxen, tank_index, 1, variable, weather_type)


KEY_TITLES: dict[str, str] = {
    "storage_electricity_fraction": "Storage",
    "solar_electricity_fraction": "Solar",
    "grid_electricity_fraction": "Grid",
    "auxiliary_heating_fraction": "Aux. heating",
}

COMPONENT_TITLES: dict[int, str] = {
    0: "Batteries",
    1: "Buffer tank capacity",
    2: "Mass flow rate",
    3: "PV",
    4: "PV-T",
    5: "Solar-thermal",
}


def boxen_frame(
    data_to_boxen,
    tank_index: int | None = None,
    weather_type: str = "average_weather_conditions",
):
    scenarios_map = {
        (scenario_key := "scenario"): scenarios(data_to_boxen)[::3][
            :: (3 if tank_index is not None else 1)
        ]
    }
    scenarios_map.update(
        {
            KEY_TITLES[key]: _results_value(
                data_to_boxen, tank_index, key, weather_type
            )
            for key in [
                "storage_electricity_fraction",
                "solar_electricity_fraction",
                "grid_electricity_fraction",
                "auxiliary_heating_fraction",
            ]
        }
    )
    return pd.DataFrame(scenarios_map).set_index(scenario_key)


def components_boxen_frame(
    data_to_boxen,
    tank_index: int | None = None,
    weather_type: str = "average_weather_conditions",
):
    scenarios_map = {
        (scenario_key := "scenario"): scenarios(data_to_boxen)[::3][
            :: (3 if tank_index is not None else 1)
        ]
    }
    scenarios_map.update(
        {
            COMPONENT_TITLES[key]: _component_value(
                data_to_boxen, tank_index, key, weather_type
            )
            for key in [0, 3, 4, 5]
        }
    )
    return pd.DataFrame(scenarios_map).set_index(scenario_key)


COST_KEY_TITLES: dict[str, str] = {
    "total_cost": "Total",
    "components_cost": "Components",
    "grid_costs": "Grid",
    "heat_pump_cost": "Heat-pump",
    "inverters_cost": "Inverter(s)",
}


def costs_boxen_frame(
    data_to_boxen,
    specific_rescale: float = 1,
    tank_index: int | None = None,
    unit: float = 10**6,
    weather_type: str = "average_weather_conditions",
):
    """
    Used for calculating the costs.
    Inputs:
        - specific_rescale:
            The specific cost factor used to rescale the costs.
        - tank_index:
            Which tank index to use.
        - unit:
            The cost unit, usually MUSD corresponding to 10^6.
        - weather_type:
            Which type of weather data to use.
    """
    scenarios_map = {
        (scenario_key := "scenario"): scenarios(data_to_boxen)[::3][
            :: (3 if tank_index is not None else 1)
        ]
    }
    scenarios_map.update(
        {
            COST_KEY_TITLES[key]: [
                (entry / (specific_rescale * unit) if entry is not None else entry)
                for entry in _results_value(
                    data_to_boxen, tank_index, key, weather_type
                )
            ]
            for key in [
                "total_cost",
                "components_cost",
                "grid_costs",
                "heat_pump_cost",
                "inverters_cost",
            ]
        }
    )
    return pd.DataFrame(scenarios_map).set_index(scenario_key)


class Plant(enum.Enum):
    """Specifies which plant is being considered."""

    JOO: str = "joo"
    EL_NASHAR: str = "el"
    RAHIMI: str = "rahimi"


def specific_costs_boxen_frame(
    data_to_boxen,
    tank_index: int | None = None,
    unit: float = 1,
    weather_type: str = "average_weather_conditions",
    *,
    plant: Plant,
):
    if plant == Plant.JOO:
        return costs_boxen_frame(data_to_boxen, 32872.5, tank_index, unit, weather_type)
    if plant == Plant.EL_NASHAR:
        return costs_boxen_frame(data_to_boxen, 1314900, tank_index, unit, weather_type)
    if plant == Plant.RAHIMI:
        return costs_boxen_frame(
            data_to_boxen, 18562005, tank_index, unit, weather_type
        )
    raise Exception("Unsupported plant type, %s, specified.", plant)


# Plots of the total cost based on the technology type
def cost_by_tech_frame(
    data_to_cost_by_tech: dict[str, Any], *, cost_key: str = "Total"
):
    """Return a dataframe with columns containing the cost of each technology type"""
    return pd.DataFrame(
        {
            "D.S. 300": costs_boxen_frame(_dual_300_data(data_to_cost_by_tech))[
                cost_key
            ],
            "D.S. 400": costs_boxen_frame(_dual_400_data(data_to_cost_by_tech))[
                cost_key
            ],
            "Solimp.": costs_boxen_frame(_soli_data(data_to_cost_by_tech))[cost_key],
            "m-Si PV": costs_boxen_frame(_m_si_data(data_to_cost_by_tech))[cost_key],
            "p-Si PV": costs_boxen_frame(_p_si_data(data_to_cost_by_tech))[cost_key],
            "FPC": costs_boxen_frame(_fpc_data(data_to_cost_by_tech))[cost_key],
            "Aug.": costs_boxen_frame(_etc_aug_data(data_to_cost_by_tech))[cost_key],
            "Euro.": costs_boxen_frame(_etc_euro_data(data_to_cost_by_tech))[cost_key],
        }
    )


def specific_cost_by_tech_frame(
    data_to_cost_by_tech: dict[str, Any], *, plant: Plant, cost_key: str = "Total"
):
    """Return a dataframe with columns containing the specific cost of each technology type"""
    return pd.DataFrame(
        {
            "D.S. 300": specific_costs_boxen_frame(
                _dual_300_data(data_to_cost_by_tech), plant=plant
            )[cost_key],
            "D.S. 400": specific_costs_boxen_frame(
                _dual_400_data(data_to_cost_by_tech), plant=plant
            )[cost_key],
            "Solimp.": specific_costs_boxen_frame(
                _soli_data(data_to_cost_by_tech), plant=plant
            )[cost_key],
            "m-Si PV": specific_costs_boxen_frame(
                _m_si_data(data_to_cost_by_tech), plant=plant
            )[cost_key],
            "p-Si PV": specific_costs_boxen_frame(
                _p_si_data(data_to_cost_by_tech), plant=plant
            )[cost_key],
            "FPC": specific_costs_boxen_frame(
                _fpc_data(data_to_cost_by_tech), plant=plant
            )[cost_key],
            "Aug.": specific_costs_boxen_frame(
                _etc_aug_data(data_to_cost_by_tech), plant=plant
            )[cost_key],
            "Euro.": specific_costs_boxen_frame(
                _etc_euro_data(data_to_cost_by_tech), plant=plant
            )[cost_key],
        }
    )


# Outputs boxen plot by location

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
K_DEPTH: int = 4
WEATHER_TYPE: str = "average_weather_conditions"
# WEATHER_TYPE: str = "upper_error_bar_weather_conditions"
# WEATHER_TYPE: str = "lower_error_bar_weather_conditions"

TANK_INDEX: int | None = None

sns.boxenplot(
    boxen_frame(abu_dhabi_data, tank_index=TANK_INDEX, weather_type=WEATHER_TYPE),
    ax=(axis := axes[0, 0]),
    k_depth=K_DEPTH,
)
axis.set_title("Abu Dhabi, UAE")
axis.set_ylim(
    max(min(min(boxen_frame(data)["Aux. heating"]) - 0.05, -0.05), -0.8),
    1.05,
)
axis.set_ylabel("Fraction")
axis.text(
    -0.08,
    1.1,
    "a.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)

sns.boxenplot(
    boxen_frame(gran_canaria_data, tank_index=TANK_INDEX, weather_type=WEATHER_TYPE),
    ax=(axis := axes[0, 1]),
    k_depth=K_DEPTH,
)
axis.set_title("Gando, Gran Canaria")
axis.set_ylim(
    max(min(min(boxen_frame(data)["Aux. heating"]) - 0.05, -0.05), -0.8),
    1.05,
)
axis.set_ylabel("Fraction")
axis.text(
    -0.08,
    1.1,
    "b.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)

sns.boxenplot(
    boxen_frame(tijuana_data, tank_index=TANK_INDEX, weather_type=WEATHER_TYPE),
    ax=(axis := axes[1, 0]),
    k_depth=K_DEPTH,
)
axis.set_title("Tijuana, Mexico")
axis.set_ylim(
    max(min(min(boxen_frame(data)["Aux. heating"]) - 0.05, -0.05), -0.8), 1.05
)
axis.set_ylabel("Fraction")
axis.text(
    -0.08,
    1.1,
    "c.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)

sns.boxenplot(
    boxen_frame(la_paz_data, tank_index=TANK_INDEX, weather_type=WEATHER_TYPE),
    ax=(axis := axes[1, 1]),
    k_depth=K_DEPTH,
)
axis.set_title("La Paz, Mexico")
axis.set_ylim(
    max(min(min(boxen_frame(data)["Aux. heating"]) - 0.05, -0.05), -0.8), 1.05
)
axis.set_ylabel("Fraction")
axis.text(
    -0.08,
    1.1,
    "d.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)


plt.savefig(
    "fractions_8.png",
    transparent=True,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

# Components boxen plot by location


def _post_process_split_axes(ax1, ax2):
    """
    Function to post-process the joining of axes.
    Adapted from:
        https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
    """
    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.tick_params(
        labeltop=False, labelbottom=False
    )  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    # Now, let's turn towards the cut-out slanted lines.
    # We create line objects in axes coordinates, in which (0,0), (0,1),
    # (1,0), and (1,1) are the four corners of the axes.
    # The slanted lines themselves are markers at those locations, such that the
    # lines keep their angle and position, independent of the axes size or scale
    # Finally, we need to disable clipping.
    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=12,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    ax1.plot([0], [0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0], [1], transform=ax2.transAxes, **kwargs)


# Joo Plot
gridspec = {"hspace": 0.1, "height_ratios": [1, 1, 0.4, 1, 1]}
fig, axes = plt.subplots(5, 2, figsize=(12, 8), gridspec_kw=gridspec)
fig.subplots_adjust(hspace=0, wspace=0.25)

axes[2, 0].set_visible(False)
axes[2, 1].set_visible(False)
y_label_coord: int = int(-850)

axes[0, 0].get_shared_x_axes().join(axes[0, 0], axes[1, 0])
axes[3, 0].get_shared_x_axes().join(axes[3, 0], axes[4, 0])
axes[3, 1].get_shared_x_axes().join(axes[3, 1], axes[4, 1])
axes[0, 1].get_shared_x_axes().join(axes[0, 1], axes[1, 1])

K_DEPTH: int = 4
WEATHER_TYPE: str = "average_weather_conditions"
# WEATHER_TYPE: str = "upper_error_bar_weather_conditions"
# WEATHER_TYPE: str = "lower_error_bar_weather_conditions"

TANK_INDEX: int | None = None

# Determine the upper y limit

max_y_lim = max(
    max(
        (
            max_frame := components_boxen_frame(
                la_paz_joo, tank_index=TANK_INDEX, weather_type=WEATHER_TYPE
            )
        )["PV"]
    ),
    max(max_frame["PV-T"]),
    max(max_frame["Solar-thermal"]),
)

# Determine the break y limit
break_y_lim = 500
y_text_position: float = 2.25
y_x_label_coord: float = -1.25

lower_y_limits = (0, break_y_lim)
upper_y_limits = (break_y_lim + 1, 1.05 * max_y_lim)

# Abu Dhabi Plot
upper_axis = axes[0, 0]
lower_axis = axes[1, 0]

upper_axis.set_ylim(*upper_y_limits)
lower_axis.set_ylim(*lower_y_limits)

for axis in (upper_axis, lower_axis):
    sns.boxenplot(
        components_boxen_frame(
            abu_dhabi_joo, tank_index=TANK_INDEX, weather_type=WEATHER_TYPE
        ),
        ax=axis,
        k_depth=K_DEPTH,
    )

upper_axis.set_title("Abu Dhabi, UAE")
upper_axis.text(
    y_x_label_coord, y_label_coord, "Number of components", rotation="vertical"
)
lower_axis.set_xlabel("Component")
upper_axis.text(
    -0.1,
    y_text_position,
    "a.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
sns.despine(top=True)
_post_process_split_axes(upper_axis, lower_axis)

# Gran Canaria Plot
upper_axis = axes[0, 1]
lower_axis = axes[1, 1]

upper_axis.set_ylim(*upper_y_limits)
lower_axis.set_ylim(*lower_y_limits)

for axis in (upper_axis, lower_axis):
    sns.boxenplot(
        components_boxen_frame(
            gran_canaria_joo, tank_index=TANK_INDEX, weather_type=WEATHER_TYPE
        ),
        ax=axis,
        k_depth=K_DEPTH,
    )

upper_axis.set_title("Gando, Gran Canaria")
upper_axis.text(
    y_x_label_coord, y_label_coord, "Number of components", rotation="vertical"
)
lower_axis.set_xlabel("Component")
upper_axis.text(
    -0.1,
    y_text_position,
    "b.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
sns.despine(top=True)
_post_process_split_axes(upper_axis, lower_axis)

# Tijuana Plot
upper_axis = axes[3, 0]
lower_axis = axes[4, 0]

upper_axis.set_ylim(*upper_y_limits)
lower_axis.set_ylim(*lower_y_limits)

for axis in (upper_axis, lower_axis):
    sns.boxenplot(
        components_boxen_frame(
            tijuana_joo, tank_index=TANK_INDEX, weather_type=WEATHER_TYPE
        ),
        ax=axis,
        k_depth=K_DEPTH,
    )

upper_axis.set_title("Tijuana, Mexico")
upper_axis.text(
    y_x_label_coord, y_label_coord, "Number of components", rotation="vertical"
)
lower_axis.set_xlabel("Component")
upper_axis.text(
    -0.1,
    y_text_position,
    "c.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
sns.despine(top=True)
_post_process_split_axes(upper_axis, lower_axis)

# La Paz Plot
upper_axis = axes[3, 1]
lower_axis = axes[4, 1]

upper_axis.set_ylim(*upper_y_limits)
lower_axis.set_ylim(*lower_y_limits)

for axis in (upper_axis, lower_axis):
    sns.boxenplot(
        components_boxen_frame(
            la_paz_joo, tank_index=TANK_INDEX, weather_type=WEATHER_TYPE
        ),
        ax=axis,
        k_depth=K_DEPTH,
    )

upper_axis.set_title("La Paz, Mexico")
lower_axis.set_xlabel("Component")
upper_axis.text(
    y_x_label_coord, y_label_coord, "Number of components", rotation="vertical"
)
upper_axis.text(
    -0.1,
    y_text_position,
    "d.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
sns.despine(top=True)
_post_process_split_axes(upper_axis, lower_axis)

plt.savefig(
    "joo_component_sizes_7.png",
    transparent=True,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# El-Nashar Plot
gridspec = {"hspace": 0.1, "height_ratios": [1, 1, 0.4, 1, 1]}
fig, axes = plt.subplots(5, 2, figsize=(12, 8), gridspec_kw=gridspec)
fig.subplots_adjust(hspace=0, wspace=0.25)

axes[2, 0].set_visible(False)
axes[2, 1].set_visible(False)
y_label_coord: int = int(-6.8 * 10**4)

axes[0, 0].get_shared_x_axes().join(axes[0, 0], axes[1, 0])
axes[3, 0].get_shared_x_axes().join(axes[3, 0], axes[4, 0])
axes[3, 1].get_shared_x_axes().join(axes[3, 1], axes[4, 1])
axes[0, 1].get_shared_x_axes().join(axes[0, 1], axes[1, 1])

K_DEPTH: int = 4
WEATHER_TYPE: str = "average_weather_conditions"
# WEATHER_TYPE: str = "upper_error_bar_weather_conditions"
# WEATHER_TYPE: str = "lower_error_bar_weather_conditions"

TANK_INDEX: int | None = None

# Determine the upper y limit

max_y_lim = max(
    max(
        (
            max_frame := components_boxen_frame(
                la_paz_el, tank_index=TANK_INDEX, weather_type=WEATHER_TYPE
            )
        )["PV"]
    ),
    max(max_frame["PV-T"]),
    max(max_frame["Solar-thermal"]),
)

# Determine the break y limit
break_y_lim = 5000
y_text_position: float = 2.25
y_x_label_coord: float = -1.35

lower_y_limits = (0, break_y_lim)
upper_y_limits = (break_y_lim + 1, 1.05 * max_y_lim)

# Abu Dhabi Plot
upper_axis = axes[0, 0]
lower_axis = axes[1, 0]

upper_axis.set_ylim(*upper_y_limits)
lower_axis.set_ylim(*lower_y_limits)

for axis in (upper_axis, lower_axis):
    sns.boxenplot(
        components_boxen_frame(
            abu_dhabi_el, tank_index=TANK_INDEX, weather_type=WEATHER_TYPE
        ),
        ax=axis,
        k_depth=K_DEPTH,
    )

upper_axis.set_title("Abu Dhabi, UAE")
upper_axis.text(
    y_x_label_coord, y_label_coord, "Number of components", rotation="vertical"
)
lower_axis.set_xlabel("Component")
upper_axis.text(
    -0.1,
    y_text_position,
    "a.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
sns.despine(top=True)
_post_process_split_axes(upper_axis, lower_axis)

# Gran Canaria Plot
upper_axis = axes[0, 1]
lower_axis = axes[1, 1]

upper_axis.set_ylim(*upper_y_limits)
lower_axis.set_ylim(*lower_y_limits)

# for axis in (upper_axis, lower_axis):
for axis in (upper_axis, lower_axis):
    if axis == upper_axis:
        axis.xaxis.grid(False)
        continue
    sns.boxenplot(
        components_boxen_frame(
            gran_canaria_el, tank_index=TANK_INDEX, weather_type=WEATHER_TYPE
        ),
        ax=axis,
        k_depth=K_DEPTH,
    )

upper_axis.set_title("Gando, Gran Canaria")
upper_axis.text(
    y_x_label_coord, y_label_coord, "Number of components", rotation="vertical"
)
lower_axis.set_xlabel("Component")
upper_axis.text(
    -0.1,
    y_text_position,
    "b.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
sns.despine(top=True)
_post_process_split_axes(upper_axis, lower_axis)

# Tijuana Plot
upper_axis = axes[3, 0]
lower_axis = axes[4, 0]

upper_axis.set_ylim(*upper_y_limits)
lower_axis.set_ylim(*lower_y_limits)


for axis in (upper_axis, lower_axis):
    if axis == upper_axis:
        tijuana_components_boxen_frame = components_boxen_frame(
            tijuana_el, tank_index=TANK_INDEX, weather_type=WEATHER_TYPE
        )
        tijuana_components_boxen_frame["Batteries"] = [0] * len(
            tijuana_components_boxen_frame
        )
        tijuana_components_boxen_frame["PV-T"] = [0] * len(
            tijuana_components_boxen_frame
        )
        sns.boxenplot(
            tijuana_components_boxen_frame,
            ax=axis,
            k_depth=K_DEPTH,
        )
        continue
    sns.boxenplot(
        components_boxen_frame(
            tijuana_el, tank_index=TANK_INDEX, weather_type=WEATHER_TYPE
        ),
        ax=axis,
        k_depth=K_DEPTH,
    )

upper_axis.set_title("Tijuana, Mexico")
upper_axis.text(
    y_x_label_coord, y_label_coord, "Number of components", rotation="vertical"
)
lower_axis.set_xlabel("Component")
upper_axis.text(
    -0.1,
    y_text_position,
    "c.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
sns.despine(top=True)
_post_process_split_axes(upper_axis, lower_axis)

# La Paz Plot
upper_axis = axes[3, 1]
lower_axis = axes[4, 1]

upper_axis.set_ylim(*upper_y_limits)
lower_axis.set_ylim(*lower_y_limits)

for axis in (upper_axis, lower_axis):
    sns.boxenplot(
        components_boxen_frame(
            la_paz_el, tank_index=TANK_INDEX, weather_type=WEATHER_TYPE
        ),
        ax=axis,
        k_depth=K_DEPTH,
    )

upper_axis.set_title("La Paz, Mexico")
lower_axis.set_xlabel("Component")
upper_axis.text(
    y_x_label_coord, y_label_coord, "Number of components", rotation="vertical"
)
upper_axis.text(
    -0.1,
    y_text_position,
    "d.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
sns.despine(top=True)
_post_process_split_axes(upper_axis, lower_axis)

plt.savefig(
    "el_nashar_component_sizes_7.png",
    transparent=True,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# Rahimi Plot

from matplotlib.ticker import FuncFormatter


def millions(x, pos):
    """
    The two args are the value and tick position
    Function adapted from:
        https://stackoverflow.com/questions/61330427/set-y-axis-in-millions
    """
    return "%1.2f M" % (x * 1e-6)


formatter = FuncFormatter(millions)


gridspec = {"hspace": 0.1, "height_ratios": [1, 1, 0.4, 1, 1]}
fig, axes = plt.subplots(5, 2, figsize=(12, 8), gridspec_kw=gridspec)
fig.subplots_adjust(hspace=0, wspace=0.25)

axes[2, 0].set_visible(False)
axes[2, 1].set_visible(False)
y_label_coord: int = int(-5.5 * 10**5)

axes[0, 0].get_shared_x_axes().join(axes[0, 0], axes[1, 0])
axes[3, 0].get_shared_x_axes().join(axes[3, 0], axes[4, 0])
axes[3, 1].get_shared_x_axes().join(axes[3, 1], axes[4, 1])
axes[0, 1].get_shared_x_axes().join(axes[0, 1], axes[1, 1])

axes[0, 0].yaxis.set_major_formatter(formatter)
axes[1, 0].yaxis.set_major_formatter(formatter)
axes[3, 0].yaxis.set_major_formatter(formatter)
axes[4, 0].yaxis.set_major_formatter(formatter)
axes[0, 1].yaxis.set_major_formatter(formatter)
axes[1, 1].yaxis.set_major_formatter(formatter)
axes[3, 1].yaxis.set_major_formatter(formatter)
axes[4, 1].yaxis.set_major_formatter(formatter)


K_DEPTH: int = 4
WEATHER_TYPE: str = "average_weather_conditions"
# WEATHER_TYPE: str = "upper_error_bar_weather_conditions"
# WEATHER_TYPE: str = "lower_error_bar_weather_conditions"

TANK_INDEX: int | None = None

# Determine the upper y limit

max_y_lim = max(
    max(
        (
            max_frame := components_boxen_frame(
                la_paz_rahimi, tank_index=TANK_INDEX, weather_type=WEATHER_TYPE
            )
        )["PV"]
    ),
    max(max_frame["PV-T"]),
    max(max_frame["Solar-thermal"]),
)

# Determine the break y limit
break_y_lim = 2 * 10**5
y_text_position: float = 2.25
y_x_label_coord: float = -1.35

lower_y_limits = (0, break_y_lim)
upper_y_limits = (break_y_lim + 1, 1.05 * max_y_lim)

# Abu Dhabi Plot
upper_axis = axes[0, 0]
lower_axis = axes[1, 0]

upper_axis.set_ylim(*upper_y_limits)
lower_axis.set_ylim(*lower_y_limits)

for axis in (upper_axis, lower_axis):
    sns.boxenplot(
        components_boxen_frame(
            abu_dhabi_rahimi, tank_index=TANK_INDEX, weather_type=WEATHER_TYPE
        ),
        ax=axis,
        k_depth=K_DEPTH,
    )

upper_axis.set_title("Abu Dhabi, UAE")
upper_axis.text(
    y_x_label_coord, y_label_coord, "Number of components", rotation="vertical"
)
lower_axis.set_xlabel("Component")
upper_axis.text(
    -0.12,
    y_text_position,
    "a.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
sns.despine(top=True)
_post_process_split_axes(upper_axis, lower_axis)

# Gran Canaria Plot
upper_axis = axes[0, 1]
lower_axis = axes[1, 1]

upper_axis.set_ylim(*upper_y_limits)
lower_axis.set_ylim(*lower_y_limits)

# for axis in (upper_axis, lower_axis):
for axis in (upper_axis, lower_axis):
    # if axis == upper_axis:
    #     axis.xaxis.grid(False)
    #     continue
    sns.boxenplot(
        components_boxen_frame(
            gran_canaria_rahimi, tank_index=TANK_INDEX, weather_type=WEATHER_TYPE
        ),
        ax=axis,
        k_depth=K_DEPTH,
    )

upper_axis.set_title("Gando, Gran Canaria")
upper_axis.text(
    y_x_label_coord, y_label_coord, "Number of components", rotation="vertical"
)
lower_axis.set_xlabel("Component")
upper_axis.text(
    -0.12,
    y_text_position,
    "b.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
sns.despine(top=True)
_post_process_split_axes(upper_axis, lower_axis)

# Tijuana Plot
upper_axis = axes[3, 0]
lower_axis = axes[4, 0]

upper_axis.set_ylim(*upper_y_limits)
lower_axis.set_ylim(*lower_y_limits)


for axis in (upper_axis, lower_axis):
    # if axis == upper_axis:
    #     tijuana_components_boxen_frame = components_boxen_frame(
    #         tijuana_el, tank_index=TANK_INDEX, weather_type=WEATHER_TYPE
    #     )
    #     tijuana_components_boxen_frame["Batteries"] = [0] * len(tijuana_components_boxen_frame)
    #     tijuana_components_boxen_frame["PV-T"] = [0] * len(tijuana_components_boxen_frame)
    #     sns.boxenplot(
    #         tijuana_components_boxen_frame,
    #         ax=axis,
    #         k_depth=K_DEPTH,
    #     )
    #     continue
    sns.boxenplot(
        components_boxen_frame(
            tijuana_rahimi, tank_index=TANK_INDEX, weather_type=WEATHER_TYPE
        ),
        ax=axis,
        k_depth=K_DEPTH,
    )

upper_axis.set_title("Tijuana, Mexico")
upper_axis.text(
    y_x_label_coord, y_label_coord, "Number of components", rotation="vertical"
)
lower_axis.set_xlabel("Component")
upper_axis.text(
    -0.12,
    y_text_position,
    "c.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
sns.despine(top=True)
_post_process_split_axes(upper_axis, lower_axis)

# La Paz Plot
upper_axis = axes[3, 1]
lower_axis = axes[4, 1]

upper_axis.set_ylim(*upper_y_limits)
lower_axis.set_ylim(*lower_y_limits)

for axis in (upper_axis, lower_axis):
    sns.boxenplot(
        components_boxen_frame(
            la_paz_rahimi, tank_index=TANK_INDEX, weather_type=WEATHER_TYPE
        ),
        ax=axis,
        k_depth=K_DEPTH,
    )

upper_axis.set_title("La Paz, Mexico")
lower_axis.set_xlabel("Component")
upper_axis.text(
    y_x_label_coord, y_label_coord, "Number of components", rotation="vertical"
)
upper_axis.text(
    -0.12,
    y_text_position,
    "d.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
sns.despine(top=True)
_post_process_split_axes(upper_axis, lower_axis)

# # sns.boxenplot(
# #     components_boxen_frame(
# #         tijuana_rahimi, tank_index=TANK_INDEX, weather_type=WEATHER_TYPE
# #     ),
# #     ax=(axis := axes[1, 0]),
# #     k_depth=K_DEPTH,
# # )
# # axis.set_title("Tijuana, Mexico")
# # axis.set_ylim(0, max_y_lim)
# # # axis.set_yscale("log")
# # axis.set_ylabel("Number of components")
# # axis.text(
# #     -0.08,
# #     1.1,
# #     "c.",
# #     transform=axis.transAxes,
# #     fontsize=16,
# #     fontweight="bold",
# #     va="top",
# #     ha="right",
# # )

# sns.boxenplot(
#     components_boxen_frame(
#         la_paz_rahimi, tank_index=TANK_INDEX, weather_type=WEATHER_TYPE
#     ),
#     ax=(axis := axes[1, 1]),
#     k_depth=K_DEPTH,
# )
# axis.set_title("La Paz, Mexico")
# axis.set_ylim(0, max_y_lim)
# # axis.set_yscale("log")
# axis.set_ylabel("Number of components")
# axis.text(
#     -0.08,
#     1.1,
#     "d.",
#     transform=axis.transAxes,
#     fontsize=16,
#     fontweight="bold",
#     va="top",
#     ha="right",
# )

plt.savefig(
    "rahimi_component_sizes_7.png",
    transparent=True,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# Plotting the variations

fig, axes = plt.subplots(3, 1, figsize=(8, 12))
fig.subplots_adjust(hspace=0.25)

K_DEPTH = 4
TANK_INDEX = None
WEATHER_TYPE = "average_weather_conditions"
Y_SCALE = "linear"

data_to_variation_plot = tijuana_data.copy()

sns.boxenplot(
    cost_by_tech_frame(
        {key: value for key, value in data_to_variation_plot.items() if "joo" in key},
        cost_key="Total",
    ),
    ax=(axis := axes[0]),
    k_depth=K_DEPTH,
)
axis.set_title("Joo et al., 3 tonnes / day")
axis.set_yscale(Y_SCALE)
axis.set_ylabel("Total lifetime system cost / MUSD")
axis.set_xlabel("Technology type specified")
axis.text(
    -0.08,
    1.1,
    "a.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

sns.boxenplot(
    cost_by_tech_frame(
        {key: value for key, value in data_to_variation_plot.items() if "el" in key},
        cost_key="Total",
    ),
    ax=(axis := axes[1]),
    k_depth=K_DEPTH,
)
axis.set_title("El-Nashar et al., 120 tonnes / day")
axis.set_yscale(Y_SCALE)
axis.set_ylabel("Total lifetime system cost / MUSD")
axis.set_xlabel("Technology type specified")
axis.text(
    -0.08,
    1.1,
    "b.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
axis.ticklabel_format(style="plain", axis="y", useOffset=False)

sns.boxenplot(
    cost_by_tech_frame(
        {
            key: value
            for key, value in data_to_variation_plot.items()
            if "rahimi" in key
        },
        cost_key="Total",
    ),
    ax=(axis := axes[2]),
    k_depth=K_DEPTH,
)
axis.set_title("Rahimi et al., 1694 tonnes / day")
axis.set_yscale(Y_SCALE)
axis.set_ylabel("Total lifetime system cost / MUSD")
axis.set_xlabel("Technology type specified")
axis.text(
    -0.08,
    1.1,
    "c.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
axis.ticklabel_format(style="plain", axis="y", useOffset=False)

# axis.set_title("La Paz, Mexico")
# axis.set_ylim(max(min(min(boxen_frame(la_paz_data)["Aux. heating"]), -0.05), -0.45), 1.05)


plt.savefig(
    "tijuana_technology_types.png", transparent=True, dpi=300, bbox_inches="tight"
)

plt.show()

# Plotting the variations by plant size across all four locations

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(hspace=0.25)

K_DEPTH: int = 3

# WEATHER_TYPE: str = "average_weather_conditions"
# WEATHER_TYPE: str = "upper_error_bar_weather_conditions"
WEATHER_TYPE: str = "lower_error_bar_weather_conditions"

TANK_INDEX: int | None = None
Y_SCALE = "linear"

sns.boxenplot(
    cost_by_tech_frame(abu_dhabi_joo, cost_key="Total"),
    ax=(axis := axes[0, 0]),
    k_depth=K_DEPTH,
)
axis.set_title("Abu Dhabi, UAE")
axis.set_yscale(Y_SCALE)
axis.set_ylabel("Total lifetime system cost / MUSD")
axis.set_xlabel("Technology type specified")
axis.text(
    -0.08,
    1.1,
    "a.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
# axis.set_ylim(0.18, math.ceil(10 * cost_by_tech_frame(la_paz_joo, cost_key="Total").max(axis=1).max(axis=0)) / 10)

sns.boxenplot(
    cost_by_tech_frame(gran_canaria_joo, cost_key="Total"),
    ax=(axis := axes[0, 1]),
    k_depth=K_DEPTH,
)
axis.set_title("Gando, Gran Canaria")
axis.set_yscale(Y_SCALE)
axis.set_ylabel("Total lifetime system cost / MUSD")
axis.set_xlabel("Technology type specified")
axis.text(
    -0.08,
    1.1,
    "b.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
axis.ticklabel_format(style="plain", axis="y", useOffset=False)
# axis.set_ylim(0.18, math.ceil(10 * cost_by_tech_frame(la_paz_joo, cost_key="Total").max(axis=1).max(axis=0)) / 10)

sns.boxenplot(
    cost_by_tech_frame(tijuana_joo, cost_key="Total"),
    ax=(axis := axes[1, 0]),
    k_depth=K_DEPTH,
)
axis.set_title("Tijuana, Mexico")
axis.set_yscale(Y_SCALE)
axis.set_ylabel("Total lifetime system cost / MUSD")
axis.set_xlabel("Technology type specified")
axis.text(
    -0.08,
    1.1,
    "c.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
axis.ticklabel_format(style="plain", axis="y", useOffset=False)
# axis.set_ylim(0.18, math.ceil(10 * cost_by_tech_frame(la_paz_joo, cost_key="Total").max(axis=1).max(axis=0)) / 10)

sns.boxenplot(
    cost_by_tech_frame(la_paz_joo, cost_key="Total"),
    ax=(axis := axes[1, 1]),
    k_depth=K_DEPTH,
)
axis.set_title("La Paz, Mexico")
axis.set_yscale(Y_SCALE)
axis.set_ylabel("Total lifetime system cost / MUSD")
axis.set_xlabel("Technology type specified")
axis.text(
    -0.08,
    1.1,
    "d.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
axis.ticklabel_format(style="plain", axis="y", useOffset=False)

# axis.set_title("La Paz, Mexico")
# axis.set_ylim(0.18, math.ceil(10 * cost_by_tech_frame(la_paz_joo, cost_key="Total").max(axis=1).max(axis=0)) / 10)


plt.savefig(
    "joo_technology_types_6.png", transparent=True, dpi=300, bbox_inches="tight"
)

plt.show()

# Specific cost technology types

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(hspace=0.25)

K_DEPTH: int = 3
PLANT: Plant = Plant.JOO
# WEATHER_TYPE: str = "average_weather_conditions"
# WEATHER_TYPE: str = "upper_error_bar_weather_conditions"
WEATHER_TYPE: str = "lower_error_bar_weather_conditions"

TANK_INDEX: int | None = None
Y_SCALE = "linear"

sns.boxenplot(
    specific_cost_by_tech_frame(abu_dhabi_joo, cost_key="Total", plant=PLANT),
    ax=(axis := axes[0, 0]),
    k_depth=K_DEPTH,
)
axis.set_title("Abu Dhabi, UAE")
axis.set_yscale(Y_SCALE)
axis.set_ylabel("Specific cost / USD/m$^3$")
axis.set_xlabel("Technology type specified")
axis.text(
    -0.08,
    1.1,
    "a.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
# axis.set_ylim(0.18, math.ceil(10 * cost_by_tech_frame(la_paz_joo, cost_key="Total").max(axis=1).max(axis=0)) / 10)

sns.boxenplot(
    specific_cost_by_tech_frame(gran_canaria_joo, cost_key="Total", plant=PLANT),
    ax=(axis := axes[0, 1]),
    k_depth=K_DEPTH,
)
axis.set_title("Gando, Gran Canaria")
axis.set_yscale(Y_SCALE)
axis.set_ylabel("Specific cost / USD/m$^3$")
axis.set_xlabel("Technology type specified")
axis.text(
    -0.08,
    1.1,
    "b.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
axis.ticklabel_format(style="plain", axis="y", useOffset=False)
# axis.set_ylim(0.18, math.ceil(10 * cost_by_tech_frame(la_paz_joo, cost_key="Total").max(axis=1).max(axis=0)) / 10)

sns.boxenplot(
    specific_cost_by_tech_frame(tijuana_joo, cost_key="Total", plant=PLANT),
    ax=(axis := axes[1, 0]),
    k_depth=K_DEPTH,
)
axis.set_title("Tijuana, Mexico")
axis.set_yscale(Y_SCALE)
axis.set_ylabel("Specific cost / USD/m$^3$")
axis.set_xlabel("Technology type specified")
axis.text(
    -0.08,
    1.1,
    "c.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
axis.ticklabel_format(style="plain", axis="y", useOffset=False)
# axis.set_ylim(0.18, math.ceil(10 * cost_by_tech_frame(la_paz_joo, cost_key="Total").max(axis=1).max(axis=0)) / 10)

sns.boxenplot(
    specific_cost_by_tech_frame(la_paz_joo, cost_key="Total", plant=PLANT),
    ax=(axis := axes[1, 1]),
    k_depth=K_DEPTH,
)
axis.set_title("La Paz, Mexico")
axis.set_yscale(Y_SCALE)
axis.set_ylabel("Specific cost / USD/m$^3$")
axis.set_xlabel("Technology type specified")
axis.text(
    -0.08,
    1.1,
    "d.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
axis.ticklabel_format(style="plain", axis="y", useOffset=False)

# axis.set_title("La Paz, Mexico")
# axis.set_ylim(0.18, math.ceil(10 * cost_by_tech_frame(la_paz_joo, cost_key="Total").max(axis=1).max(axis=0)) / 10)


plt.savefig(
    "joo_specific_technology_types_6.png",
    transparent=True,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# All-plants specific technology type plot

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(hspace=0.25)

K_DEPTH: int = 3
PLANT: Plant = Plant.JOO
# WEATHER_TYPE: str = "average_weather_conditions"
# WEATHER_TYPE: str = "upper_error_bar_weather_conditions"
WEATHER_TYPE: str = "lower_error_bar_weather_conditions"

TANK_INDEX: int | None = None
Y_SCALE = "linear"

sns.boxenplot(
    pd.concat(
        [
            specific_cost_by_tech_frame(
                abu_dhabi_joo, cost_key="Total", plant=Plant.JOO
            ),
            specific_cost_by_tech_frame(
                abu_dhabi_el, cost_key="Total", plant=Plant.EL_NASHAR
            ),
            specific_cost_by_tech_frame(
                abu_dhabi_rahimi, cost_key="Total", plant=Plant.RAHIMI
            ),
        ]
    ),
    ax=(axis := axes[0, 0]),
    k_depth=K_DEPTH,
)
axis.set_title("Abu Dhabi, UAE")
axis.set_yscale(Y_SCALE)
axis.set_ylabel("Specific cost / USD/m$^3$")
axis.set_xlabel("Technology type specified")
axis.text(
    -0.08,
    1.1,
    "a.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
# axis.set_ylim(0.18, math.ceil(10 * cost_by_tech_frame(la_paz_joo, cost_key="Total").max(axis=1).max(axis=0)) / 10)

sns.boxenplot(
    pd.concat(
        [
            specific_cost_by_tech_frame(
                gran_canaria_joo, cost_key="Total", plant=Plant.JOO
            ),
            specific_cost_by_tech_frame(
                gran_canaria_el, cost_key="Total", plant=Plant.EL_NASHAR
            ),
            specific_cost_by_tech_frame(
                gran_canaria_rahimi, cost_key="Total", plant=Plant.RAHIMI
            ),
        ]
    ),
    ax=(axis := axes[0, 1]),
    k_depth=K_DEPTH,
)
axis.set_title("Gando, Gran Canaria")
axis.set_yscale(Y_SCALE)
axis.set_ylabel("Specific cost / USD/m$^3$")
axis.set_xlabel("Technology type specified")
axis.text(
    -0.08,
    1.1,
    "b.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
axis.ticklabel_format(style="plain", axis="y", useOffset=False)
# axis.set_ylim(0.18, math.ceil(10 * cost_by_tech_frame(la_paz_joo, cost_key="Total").max(axis=1).max(axis=0)) / 10)

sns.boxenplot(
    pd.concat(
        [
            specific_cost_by_tech_frame(tijuana_joo, cost_key="Total", plant=Plant.JOO),
            specific_cost_by_tech_frame(
                tijuana_el, cost_key="Total", plant=Plant.EL_NASHAR
            ),
            specific_cost_by_tech_frame(
                tijuana_rahimi, cost_key="Total", plant=Plant.RAHIMI
            ),
        ]
    ),
    ax=(axis := axes[1, 0]),
    k_depth=K_DEPTH,
)
axis.set_title("Tijuana, Mexico")
axis.set_yscale(Y_SCALE)
axis.set_ylabel("Specific cost / USD/m$^3$")
axis.set_xlabel("Technology type specified")
axis.text(
    -0.08,
    1.1,
    "c.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
axis.ticklabel_format(style="plain", axis="y", useOffset=False)
# axis.set_ylim(0.18, math.ceil(10 * cost_by_tech_frame(la_paz_joo, cost_key="Total").max(axis=1).max(axis=0)) / 10)

sns.boxenplot(
    pd.concat(
        [
            specific_cost_by_tech_frame(la_paz_joo, cost_key="Total", plant=Plant.JOO),
            specific_cost_by_tech_frame(
                la_paz_el, cost_key="Total", plant=Plant.EL_NASHAR
            ),
            specific_cost_by_tech_frame(
                la_paz_rahimi, cost_key="Total", plant=Plant.RAHIMI
            ),
        ]
    ),
    ax=(axis := axes[1, 1]),
    k_depth=K_DEPTH,
)
axis.set_title("La Paz, Mexico")
axis.set_yscale(Y_SCALE)
axis.set_ylabel("Specific cost / USD/m$^3$")
axis.set_xlabel("Technology type specified")
axis.text(
    -0.08,
    1.1,
    "d.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
axis.ticklabel_format(style="plain", axis="y", useOffset=False)

# axis.set_title("La Paz, Mexico")
# axis.set_ylim(0.18, math.ceil(10 * cost_by_tech_frame(la_paz_joo, cost_key="Total").max(axis=1).max(axis=0)) / 10)


plt.savefig(
    "all_plant_specific_technology_types_6.png",
    transparent=True,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# Investigating Gran Canaria costs...
sns.boxenplot(
    cost_by_tech_frame(gran_canaria_joo, cost_key="Total"),
    k_depth=K_DEPTH,
)
plt.set_title("Gando, Gran Canaria")
plt.set_yscale(Y_SCALE)
plt.set_ylabel("Total lifetime system cost / MUSD")
plt.set_xlabel("Technology type specified")
plt.text(
    -0.08,
    1.1,
    "b.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
plt.ticklabel_format(style="plain", axis="y", useOffset=False)

plt.show()

# Investigating Tijuana data...
sns.boxenplot(
    cost_by_tech_frame(tijuana_joo, cost_key="Total"),
    k_depth=K_DEPTH,
)
plt.title("Tijuana, Mexico")
plt.yscale(Y_SCALE)
plt.ylabel("Total lifetime system cost / MUSD")
plt.xlabel("Technology type specified")
plt.text(
    -0.08,
    1.1,
    "c.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
plt.ticklabel_format(style="plain", axis="y", useOffset=False)
plt.show()


# Mean component numbers

mean_component_numbers = {
    "Abu Dhabi Joo Batt.": float(
        round(np.mean(components_boxen_frame(abu_dhabi_joo)["Batteries"]), 0)
    ),
    "Abu Dhabi Joo PV": round(
        float(np.mean(components_boxen_frame(abu_dhabi_joo)["PV"])), 0
    ),
    "Abu Dhabi Joo PV-T": round(
        float(np.mean(components_boxen_frame(abu_dhabi_joo)["PV-T"])), 0
    ),
    "Abu Dhabi Joo ST": float(
        round(np.mean(components_boxen_frame(abu_dhabi_joo)["Solar-thermal"]), 0)
    ),
    "Abu Dhabi El-Nashar Batt.": float(
        round(np.mean(components_boxen_frame(abu_dhabi_el)["Batteries"]), 0)
    ),
    "Abu Dhabi El-Nashar PV": float(
        round(np.mean(components_boxen_frame(abu_dhabi_el)["PV"]), 0)
    ),
    "Abu Dhabi El-Nashar PV-T": float(
        round(np.mean(components_boxen_frame(abu_dhabi_el)["PV-T"]), 0)
    ),
    "Abu Dhabi El-Nashar ST": float(
        round(np.mean(components_boxen_frame(abu_dhabi_el)["Solar-thermal"]), 0)
    ),
    "Abu Dhabi Raihimi Batt.": float(
        round(np.mean(components_boxen_frame(abu_dhabi_rahimi)["Batteries"]), 0)
    ),
    "Abu Dhabi Raihimi PV": float(
        round(np.mean(components_boxen_frame(abu_dhabi_rahimi)["PV"]), 0)
    ),
    "Abu Dhabi Raihimi PV-T": float(
        round(np.mean(components_boxen_frame(abu_dhabi_rahimi)["PV-T"]), 0)
    ),
    "Abu Dhabi Raihimi ST": float(
        round(np.mean(components_boxen_frame(abu_dhabi_rahimi)["Solar-thermal"]), 0)
    ),
    "Gran Canaria Joo Batt.": float(
        round(np.mean(components_boxen_frame(gran_canaria_joo)["Batteries"]), 0)
    ),
    "Gran Canaria Joo PV": float(
        round(np.mean(components_boxen_frame(gran_canaria_joo)["PV"]), 0)
    ),
    "Gran Canaria Joo PV-T": float(
        round(np.mean(components_boxen_frame(gran_canaria_joo)["PV-T"]), 0)
    ),
    "Gran Canaria Joo ST": float(
        round(np.mean(components_boxen_frame(gran_canaria_joo)["Solar-thermal"]), 0)
    ),
    "Gran Canaria El-Nashar Batt.": float(
        round(np.mean(components_boxen_frame(gran_canaria_el)["Batteries"]), 0)
    ),
    "Gran Canaria El-Nashar PV": float(
        round(np.mean(components_boxen_frame(gran_canaria_el)["PV"]), 0)
    ),
    "Gran Canaria El-Nashar PV-T": float(
        round(np.mean(components_boxen_frame(gran_canaria_el)["PV-T"]), 0)
    ),
    "Gran Canaria El-Nashar ST": float(
        round(np.mean(components_boxen_frame(gran_canaria_el)["Solar-thermal"]), 0)
    ),
    "Gran Canaria Raihimi Batt.": float(
        round(np.mean(components_boxen_frame(gran_canaria_rahimi)["Batteries"]), 0)
    ),
    "Gran Canaria Raihimi PV": float(
        round(np.mean(components_boxen_frame(gran_canaria_rahimi)["PV"]), 0)
    ),
    "Gran Canaria Raihimi PV-T": float(
        round(np.mean(components_boxen_frame(gran_canaria_rahimi)["PV-T"]), 0)
    ),
    "Gran Canaria Raihimi ST": float(
        round(np.mean(components_boxen_frame(gran_canaria_rahimi)["Solar-thermal"]), 0)
    ),
    "Tijuana Joo Batt.": float(
        round(np.mean(components_boxen_frame(tijuana_joo)["Batteries"]), 0)
    ),
    "Tijuana Joo PV": round(
        float(np.mean(components_boxen_frame(tijuana_joo)["PV"])), 0
    ),
    "Tijuana Joo PV-T": round(
        float(np.mean(components_boxen_frame(tijuana_joo)["PV-T"])), 0
    ),
    "Tijuana Joo ST": float(
        round(np.mean(components_boxen_frame(tijuana_joo)["Solar-thermal"]), 0)
    ),
    "Tijuana El-Nashar Batt.": float(
        round(np.mean(components_boxen_frame(tijuana_el)["Batteries"]), 0)
    ),
    "Tijuana El-Nashar PV": round(
        float(np.mean(components_boxen_frame(tijuana_el)["PV"])), 0
    ),
    "Tijuana El-Nashar PV-T": float(
        round(np.mean(components_boxen_frame(tijuana_el)["PV-T"]), 0)
    ),
    "Tijuana El-Nashar ST": float(
        round(np.mean(components_boxen_frame(tijuana_el)["Solar-thermal"]), 0)
    ),
    "Tijuana Raihimi Batt.": float(
        round(np.mean(components_boxen_frame(tijuana_rahimi)["Batteries"]), 0)
    ),
    "Tijuana Raihimi PV": round(
        float(np.mean(components_boxen_frame(tijuana_rahimi)["PV"])), 0
    ),
    "Tijuana Raihimi PV-T": float(
        round(np.mean(components_boxen_frame(tijuana_rahimi)["PV-T"]), 0)
    ),
    "Tijuana Raihimi ST": float(
        round(np.mean(components_boxen_frame(tijuana_rahimi)["Solar-thermal"]), 0)
    ),
    "La Paz Joo Batt.": round(
        float(np.mean(components_boxen_frame(la_paz_joo)["Batteries"])), 0
    ),
    "La Paz Joo PV": round(float(np.mean(components_boxen_frame(la_paz_joo)["PV"])), 0),
    "La Paz Joo PV-T": round(
        float(np.mean(components_boxen_frame(la_paz_joo)["PV-T"])), 0
    ),
    "La Paz Joo ST": float(
        round(np.mean(components_boxen_frame(la_paz_joo)["Solar-thermal"]), 0)
    ),
    "La Paz El-Nashar Batt.": float(
        round(np.mean(components_boxen_frame(la_paz_el)["Batteries"]), 0)
    ),
    "La Paz El-Nashar PV": round(
        float(np.mean(components_boxen_frame(la_paz_el)["PV"])), 0
    ),
    "La Paz El-Nashar PV-T": round(
        float(np.mean(components_boxen_frame(la_paz_el)["PV-T"])), 0
    ),
    "La Paz El-Nashar ST": float(
        round(np.mean(components_boxen_frame(la_paz_el)["Solar-thermal"]), 0)
    ),
    "La Paz Raihimi Batt.": float(
        round(np.mean(components_boxen_frame(la_paz_rahimi)["Batteries"]), 0)
    ),
    "La Paz Raihimi PV": round(
        float(np.mean(components_boxen_frame(la_paz_rahimi)["PV"])), 0
    ),
    "La Paz Raihimi PV-T": float(
        round(np.mean(components_boxen_frame(la_paz_rahimi)["PV-T"]), 0)
    ),
    "La Paz Raihimi ST": float(
        round(np.mean(components_boxen_frame(la_paz_rahimi)["Solar-thermal"]), 0)
    ),
}

print(json.dumps(mean_component_numbers, indent=4))

# Plot a bar plot of the component costs broken down for each plant.

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(hspace=0.25)

K_DEPTH: int = 3

# WEATHER_TYPE: str = "average_weather_conditions"
# WEATHER_TYPE: str = "upper_error_bar_weather_conditions"
WEATHER_TYPE: str = "lower_error_bar_weather_conditions"

TANK_INDEX: int | None = None
Y_SCALE = "linear"


data_to_plot = (
    pd.DataFrame(
        {
            "Small": (
                frame := specific_costs_boxen_frame(abu_dhabi_joo, plant=Plant.JOO)
            ).loc[frame["Total"].idxmin()],
            "Medium": (
                frame := specific_costs_boxen_frame(abu_dhabi_el, plant=Plant.EL_NASHAR)
            ).loc[frame["Total"].idxmin()],
            "Large": (
                frame := specific_costs_boxen_frame(
                    abu_dhabi_rahimi, plant=Plant.RAHIMI
                )
            ).loc[frame["Total"].idxmin()],
        }
    )
    .drop("Total")
    .transpose()
)
data_to_plot.plot.bar(ax=(axis := axes[0, 0]), rot=0, stacked=True)
axis.grid(axis="x")
axis.set_xlabel("MED Plant")
axis.set_ylabel("Specific cost / USD/m$^3$")
axis.set_title("Abu Dhabi, UAE")
axis.yaxis.set_minor_formatter(ticker.ScalarFormatter())
axis.yaxis.set_major_formatter(ticker.ScalarFormatter())
axis.set_ylim(0, 7)
axis.text(
    -0.08,
    1.1,
    "a.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)

data_to_plot = (
    pd.DataFrame(
        {
            "Small": (
                frame := specific_costs_boxen_frame(gran_canaria_joo, plant=Plant.JOO)
            ).loc[frame["Total"].idxmin()],
            "Medium": (
                frame := specific_costs_boxen_frame(
                    gran_canaria_el, plant=Plant.EL_NASHAR
                )
            ).loc[frame["Total"].idxmin()],
            "Large": (
                frame := specific_costs_boxen_frame(
                    gran_canaria_rahimi, plant=Plant.RAHIMI
                )
            ).loc[frame["Total"].idxmin()],
        }
    )
    .drop("Total")
    .transpose()
)
data_to_plot.plot.bar(ax=(axis := axes[0, 1]), rot=0, stacked=True)
axis.grid(axis="x")
axis.set_xlabel("MED Plant")
axis.set_ylabel("Specific cost / USD/m$^3$")
axis.set_title("Gando, Gran Canaria")
axis.set_ylim(0, 7)
axis.text(
    -0.08,
    1.1,
    "b.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
axis.ticklabel_format(style="plain", axis="y", useOffset=False)

data_to_plot = (
    pd.DataFrame(
        {
            "Small": (
                frame := specific_costs_boxen_frame(tijuana_joo, plant=Plant.JOO)
            ).loc[frame["Total"].idxmin()],
            "Medium": (
                frame := specific_costs_boxen_frame(tijuana_el, plant=Plant.EL_NASHAR)
            ).loc[frame["Total"].idxmin()],
            "Large": (
                frame := specific_costs_boxen_frame(tijuana_rahimi, plant=Plant.RAHIMI)
            ).loc[frame["Total"].idxmin()],
        }
    )
    .drop("Total")
    .transpose()
)
data_to_plot.plot.bar(ax=(axis := axes[1, 0]), rot=0, stacked=True)
axis.grid(axis="x")
axis.set_xlabel("MED Plant")
axis.set_ylabel("Specific cost / USD/m$^3$")
axis.set_ylim(0, 7)
axis.set_title("Tijuana, Mexico")
axis.text(
    -0.08,
    1.1,
    "c.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
axis.ticklabel_format(style="plain", axis="y", useOffset=False)

data_to_plot = (
    pd.DataFrame(
        {
            "Small": (
                frame := specific_costs_boxen_frame(la_paz_joo, plant=Plant.JOO)
            ).loc[frame["Total"].idxmin()],
            "Medium": (
                frame := specific_costs_boxen_frame(la_paz_el, plant=Plant.EL_NASHAR)
            ).loc[frame["Total"].idxmin()],
            "Large": (
                frame := specific_costs_boxen_frame(la_paz_rahimi, plant=Plant.RAHIMI)
            ).loc[frame["Total"].idxmin()],
        }
    )
    .drop("Total")
    .transpose()
)
data_to_plot.plot.bar(ax=(axis := axes[1, 1]), rot=0, stacked=True)
axis.grid(axis="x")
axis.set_xlabel("MED Plant")
axis.set_ylabel("Specific cost / USD/m$^3$")
axis.set_ylim(0, 7)
axis.set_title("La Paz, Mexico")
axis.text(
    -0.08,
    1.1,
    "d.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
axis.ticklabel_format(style="plain", axis="y", useOffset=False)

plt.savefig(
    "specific_costs_comparison_7.png", transparent=True, dpi=300, bbox_inches="tight"
)

plt.show()

# Plot a stacked-bar plot with a specific technology set.

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(hspace=0.25)

K_DEPTH: int = 3

# WEATHER_TYPE: str = "average_weather_conditions"
# WEATHER_TYPE: str = "upper_error_bar_weather_conditions"
WEATHER_TYPE: str = "lower_error_bar_weather_conditions"

TANK_INDEX: int | None = None
Y_SCALE = "linear"

data_to_plot = (
    pd.DataFrame(
        {
            "Small": specific_costs_boxen_frame(abu_dhabi_joo, plant=Plant.JOO).loc[
                "hpc_abu_dhabi_joo_sharp_300_eurotherm"
            ],
            "Medium": specific_costs_boxen_frame(
                abu_dhabi_el, plant=Plant.EL_NASHAR
            ).loc["hpc_abu_dhabi_el_sharp_300_eurotherm"],
            "Large": specific_costs_boxen_frame(
                abu_dhabi_rahimi, plant=Plant.RAHIMI
            ).loc["hpc_abu_dhabi_rahimi_sharp_300_eurotherm"],
        }
    )
    .drop("Total")
    .transpose()
)
data_to_plot.plot.bar(ax=(axis := axes[0, 0]), rot=0, stacked=True)
axis.grid(axis="x")
axis.set_xlabel("MED Plant")
axis.set_ylabel("Specific cost / USD/m$^3$")
axis.set_title("Abu Dhabi, UAE")
axis.yaxis.set_minor_formatter(ticker.ScalarFormatter())
axis.yaxis.set_major_formatter(ticker.ScalarFormatter())
axis.set_ylim(0, 40)
axis.text(
    -0.08,
    1.1,
    "a.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)

data_to_plot = (
    pd.DataFrame(
        {
            "Small": specific_costs_boxen_frame(gran_canaria_joo, plant=Plant.JOO).loc[
                "hpc_gran_canaria_joo_sharp_300_eurotherm"
            ],
            "Medium": specific_costs_boxen_frame(
                gran_canaria_el, plant=Plant.EL_NASHAR
            ).loc["hpc_gran_canaria_el_sharp_300_eurotherm"],
            "Large": specific_costs_boxen_frame(
                gran_canaria_rahimi, plant=Plant.RAHIMI
            ).loc["hpc_gran_canaria_rahimi_sharp_300_eurotherm"],
        }
    )
    .drop("Total")
    .transpose()
)
data_to_plot.plot.bar(ax=(axis := axes[0, 1]), rot=0, stacked=True)
axis.grid(axis="x")
axis.set_xlabel("MED Plant")
axis.set_ylabel("Specific cost / USD/m$^3$")
axis.set_title("Gando, Gran Canaria")
axis.set_ylim(0, 40)
axis.text(
    -0.08,
    1.1,
    "b.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
axis.ticklabel_format(style="plain", axis="y", useOffset=False)

data_to_plot = (
    pd.DataFrame(
        {
            "Small": specific_costs_boxen_frame(tijuana_joo, plant=Plant.JOO).loc[
                "hpc_tijuana_joo_sharp_300_eurotherm"
            ],
            "Medium": specific_costs_boxen_frame(tijuana_el, plant=Plant.EL_NASHAR).loc[
                "hpc_tijuana_el_sharp_300_eurotherm"
            ],
            "Large": specific_costs_boxen_frame(tijuana_rahimi, plant=Plant.JOO).loc[
                "hpc_tijuana_rahimi_sharp_300_eurotherm"
            ],
        }
    )
    .drop("Total")
    .transpose()
)
data_to_plot.plot.bar(ax=(axis := axes[1, 0]), rot=0, stacked=True)
axis.grid(axis="x")
axis.set_xlabel("MED Plant")
axis.set_ylabel("Specific cost / USD/m$^3$")
axis.set_ylim(0, 40)
axis.set_title("Tijuana, Mexico")
axis.text(
    -0.08,
    1.1,
    "c.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
axis.ticklabel_format(style="plain", axis="y", useOffset=False)

data_to_plot = (
    pd.DataFrame(
        {
            "Small": specific_costs_boxen_frame(la_paz_joo, plant=Plant.JOO).loc[
                "hpc_la_paz_joo_sharp_300_eurotherm"
            ],
            "Medium": specific_costs_boxen_frame(la_paz_el, plant=Plant.EL_NASHAR).loc[
                "hpc_la_paz_el_sharp_300_eurotherm"
            ],
            "Large": specific_costs_boxen_frame(la_paz_rahimi, plant=Plant.RAHIMI).loc[
                "hpc_la_paz_rahimi_sharp_300_eurotherm"
            ],
        }
    )
    .drop("Total")
    .transpose()
)
data_to_plot.plot.bar(ax=(axis := axes[1, 1]), rot=0, stacked=True)
axis.grid(axis="x")
axis.set_xlabel("MED Plant")
axis.set_ylabel("Specific cost / USD/m$^3$")
axis.set_ylim(0, 40)
axis.set_title("La Paz, Mexico")
axis.text(
    -0.08,
    1.1,
    "d.",
    transform=axis.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)
axis.ticklabel_format(style="plain", axis="y", useOffset=False)

# Plot a bar plot of the component costs broken down for each plant.

# Lowest-cost technology specification in each scenario

print("Abu Dhabi Joo")
print(
    (frame := specific_costs_boxen_frame(abu_dhabi_joo, plant=Plant.JOO)).loc[
        frame["Total"].idxmin()
    ]
)
print("Abu Dhabi El-Nashar")
print(
    (frame := specific_costs_boxen_frame(abu_dhabi_el, plant=Plant.EL_NASHAR)).loc[
        frame["Total"].idxmin()
    ]
)
print("Abu Dhabi Rahimi")
print(
    (frame := specific_costs_boxen_frame(abu_dhabi_rahimi, plant=Plant.RAHIMI)).loc[
        frame["Total"].idxmin()
    ]
)
print("Gran Canaria Joo")
print(
    (frame := specific_costs_boxen_frame(gran_canaria_joo, plant=Plant.JOO)).loc[
        frame["Total"].idxmin()
    ]
)
print("Gran Canaria El-Nashar")
print(
    (frame := specific_costs_boxen_frame(gran_canaria_el, plant=Plant.EL_NASHAR)).loc[
        frame["Total"].idxmin()
    ]
)
print("Gran Canaria Rahimi")
print(
    (frame := specific_costs_boxen_frame(gran_canaria_rahimi, plant=Plant.RAHIMI)).loc[
        frame["Total"].idxmin()
    ]
)
print("Tijuana Joo")
print(
    (frame := specific_costs_boxen_frame(tijuana_joo, plant=Plant.JOO)).loc[
        frame["Total"].idxmin()
    ]
)
print("Tijuana El-Nashar")
print(
    (frame := specific_costs_boxen_frame(tijuana_el, plant=Plant.EL_NASHAR)).loc[
        frame["Total"].idxmin()
    ]
)
print("Tijuana Rahimi")
print(
    (frame := specific_costs_boxen_frame(tijuana_rahimi, plant=Plant.RAHIMI)).loc[
        frame["Total"].idxmin()
    ]
)
print("La Paz Joo")
print(
    (frame := specific_costs_boxen_frame(la_paz_joo, plant=Plant.JOO)).loc[
        frame["Total"].idxmin()
    ]
)
print("La Paz El-Nashar")
print(
    (frame := specific_costs_boxen_frame(la_paz_el, plant=Plant.EL_NASHAR)).loc[
        frame["Total"].idxmin()
    ]
)
print("La Paz Rahimi")
print(
    (frame := specific_costs_boxen_frame(la_paz_rahimi, plant=Plant.RAHIMI)).loc[
        frame["Total"].idxmin()
    ]
)

print(json.dumps(lowest_cost, indent=4))

########################
# Sensitivity analysis #
########################

# Compile a regex for going through the data.
regex = re.compile(
    r"hpc_(?P<location>abu_dhabi|gran_canaria|tijuana|la_paz)_(?P<plant>joo|el|rahimi)_(?P<pv>[^_]*)_(?P<pvt>[^_]*)_(?P<st>[^_]*)_frac_(?P<frac_var>[^_]*)_(?P<percent>.*)\.json"
)

# Lowest cost for JOO:
#   Sharp       - "sharp"
#   D.S. 400    - "400"
#   STI         - "sti"

joo_p_si_sensitivity_data = {
    key: value
    for key, value in full_data.items()
    if regex.match(key) is not None
    and regex.match(key).group("plant") == "joo"
    and regex.match(key).group("pv") == "sharp"
    and regex.match(key).group("pvt") == "400"
    and regex.match(key).group("st") == "augusta"
}

joo_m_si_sensitivity_data = {
    key: value
    for key, value in full_data.items()
    if regex.match(key) is not None
    and regex.match(key).group("plant") == "joo"
    and regex.match(key).group("pv") == "rec"
    and regex.match(key).group("pvt") == "400"
    and regex.match(key).group("st") == "augusta"
}

el_nashar_p_si_sensitivity_data = {
    key: value
    for key, value in full_data.items()
    if regex.match(key) is not None
    and regex.match(key).group("plant") == "el"
    and regex.match(key).group("pv") == "sharp"
    and regex.match(key).group("pvt") == "400"
    and regex.match(key).group("st") == "augusta"
}

el_nashar_m_si_sensitivity_data = {
    key: value
    for key, value in full_data.items()
    if regex.match(key) is not None
    and regex.match(key).group("plant") == "el"
    and regex.match(key).group("pv") == "rec"
    and regex.match(key).group("pvt") == "400"
    and regex.match(key).group("st") == "augusta"
}

rahimi_p_si_sensitivity_data = {
    key: value
    for key, value in full_data.items()
    if regex.match(key) is not None
    and regex.match(key).group("plant") == "rahimi"
    and regex.match(key).group("pv") == "sharp"
    and regex.match(key).group("pvt") == "400"
    and regex.match(key).group("st") == "augusta"
}

rahimi_m_si_sensitivity_data = {
    key: value
    for key, value in full_data.items()
    if regex.match(key) is not None
    and regex.match(key).group("plant") == "rahimi"
    and regex.match(key).group("pv") == "rec"
    and regex.match(key).group("pvt") == "400"
    and regex.match(key).group("st") == "augusta"
}

# Assemble grid-cost data
# joo_abu_dhabi_p_si_grid_sensitivity_data = {
#     key: value
#     for key, value in joo_p_si_sensitivity_data.items()
#     if regex.match(key).group("frac_var") == "grid"
#     and regex.match(key).group("location") == "abu_dhabi"
# }
# joo_gando_p_si_grid_sensitivity_data = {
#     key: value
#     for key, value in joo_p_si_sensitivity_data.items()
#     if regex.match(key).group("frac_var") == "grid"
#     and regex.match(key).group("location") == "gran_canaria"
# }
# joo_tijuana_p_si_grid_sensitivity_data = {
#     key: value
#     for key, value in joo_p_si_sensitivity_data.items()
#     if regex.match(key).group("frac_var") == "grid"
#     and regex.match(key).group("location") == "tijuana"
# }
# joo_la_paz_p_si_grid_sensitivity_data = {
#     key: value
#     for key, value in joo_p_si_sensitivity_data.items()
#     if regex.match(key).group("frac_var") == "grid"
#     and regex.match(key).group("location") == "la_paz"
# }
# joo_m_si_grid_sensitivity_data = {
#     key: value
#     for key, value in joo_m_si_sensitivity_data.items()
#     if regex.match(key).group("frac_var") == "grid"
# }


def _sensitivity_fraction(
    data_to_fraction: dict[str, Any],
    variable: str,
    weather_conditions: str = "average_weather_conditions",
):
    results: dict[float, float] = {}
    for key, entry in data_to_fraction.items():
        results.update(
            {
                float(regex.match(key).group("percent")): float(
                    entry[tank_index][1][weather_conditions][0][variable]
                )
                for tank_index in range(3)
                if _scenario_match(key, tank_index)
            }
        )
    return results


def plot_fraction_sensitivity(data_to_plot, fraction_variable_name: str = "grid"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.25)
    WEATHER_TYPE: str = "average_weather_conditions"
    # WEATHER_TYPE: str = "upper_error_bar_weather_conditions"
    # WEATHER_TYPE: str = "lower_error_bar_weather_conditions"
    # Plot the fractions of power from storage, pv, and the grid.
    x, grid_fraction = zip(
        *sorted(
            zip(
                (
                    sensitivity_results := _sensitivity_fraction(
                        (
                            sensitivity_data := {
                                key: value
                                for key, value in data_to_plot.items()
                                if regex.match(key).group("frac_var")
                                == fraction_variable_name
                                and regex.match(key).group("location") == "abu_dhabi"
                            }
                        ),
                        "grid_electricity_fraction",
                        WEATHER_TYPE,
                    )
                ).keys(),
                sensitivity_results.values(),
            )
        )
    )
    _, solar_fraction = zip(
        *sorted(
            zip(
                (
                    sensitivity_results := _sensitivity_fraction(
                        sensitivity_data, "solar_electricity_fraction", WEATHER_TYPE
                    )
                ).keys(),
                sensitivity_results.values(),
            )
        )
    )
    _, storage_fraction = zip(
        *sorted(
            zip(
                (
                    sensitivity_results := _sensitivity_fraction(
                        sensitivity_data, "storage_electricity_fraction", WEATHER_TYPE
                    )
                ).keys(),
                sensitivity_results.values(),
            )
        )
    )
    (axis := axes[0, 0]).plot(
        x,
        (
            grid_line := [
                storage_fraction[index] + grid_fraction[index] + solar_fraction[index]
                for index in range(len(storage_fraction))
            ]
        ),
        color=f"C2",
        label="grid fraction",
    )
    axis.plot(
        x,
        (
            solar_line := [
                storage_fraction[index] + solar_fraction[index]
                for index in range(len(storage_fraction))
            ]
        ),
        color=f"C1",
        label="solar fraction",
    )
    axis.plot(x, storage_fraction, color=f"C0", label="storage fraction")
    axis.fill_between(
        x, [0] * len(storage_fraction), storage_fraction, color="C0", alpha=0.7
    )
    axis.fill_between(x, storage_fraction, solar_line, color="C1", alpha=0.7)
    axis.fill_between(x, solar_line, grid_line, color="C2", alpha=0.7)
    axis.set_xlabel("Fractional cost change")
    axis.set_ylabel("Fraction")
    axis.set_title("Abu Dhabi, UAE")
    axis.text(
        -0.08,
        1.1,
        "a.",
        transform=axis.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    axis.legend()
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlim(-0.2, 0.2)
    # Gando data
    x, grid_fraction = zip(
        *sorted(
            zip(
                (
                    sensitivity_results := _sensitivity_fraction(
                        (
                            sensitivity_data := {
                                key: value
                                for key, value in data_to_plot.items()
                                if regex.match(key).group("frac_var")
                                == fraction_variable_name
                                and regex.match(key).group("location") == "gran_canaria"
                            }
                        ),
                        "grid_electricity_fraction",
                        WEATHER_TYPE,
                    )
                ).keys(),
                sensitivity_results.values(),
            )
        )
    )
    _, solar_fraction = zip(
        *sorted(
            zip(
                (
                    sensitivity_results := _sensitivity_fraction(
                        sensitivity_data, "solar_electricity_fraction", WEATHER_TYPE
                    )
                ).keys(),
                sensitivity_results.values(),
            )
        )
    )
    _, storage_fraction = zip(
        *sorted(
            zip(
                (
                    sensitivity_results := _sensitivity_fraction(
                        sensitivity_data, "storage_electricity_fraction", WEATHER_TYPE
                    )
                ).keys(),
                sensitivity_results.values(),
            )
        )
    )
    (axis := axes[0, 1]).plot(
        x,
        (
            grid_line := [
                storage_fraction[index] + grid_fraction[index] + solar_fraction[index]
                for index in range(len(storage_fraction))
            ]
        ),
        color=f"C2",
        label="grid fraction",
    )
    axis.plot(
        x,
        (
            solar_line := [
                storage_fraction[index] + solar_fraction[index]
                for index in range(len(storage_fraction))
            ]
        ),
        color=f"C1",
        label="solar fraction",
    )
    axis.plot(x, storage_fraction, color=f"C0", label="storage fraction")
    axis.fill_between(
        x, [0] * len(storage_fraction), storage_fraction, color="C0", alpha=0.7
    )
    axis.fill_between(x, storage_fraction, solar_line, color="C1", alpha=0.7)
    axis.fill_between(x, solar_line, grid_line, color="C2", alpha=0.7)
    axis.set_xlabel("Fractional cost change")
    axis.set_ylabel("Fraction")
    axis.set_title("Gando, Gran Canaria")
    axis.text(
        -0.08,
        1.1,
        "b.",
        transform=axis.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    axis.legend()
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlim(-0.2, 0.2)
    # Gando data
    x, grid_fraction = zip(
        *sorted(
            zip(
                (
                    sensitivity_results := _sensitivity_fraction(
                        (
                            sensitivity_data := {
                                key: value
                                for key, value in data_to_plot.items()
                                if regex.match(key).group("frac_var")
                                == fraction_variable_name
                                and regex.match(key).group("location") == "tijuana"
                            }
                        ),
                        "grid_electricity_fraction",
                        WEATHER_TYPE,
                    )
                ).keys(),
                sensitivity_results.values(),
            )
        )
    )
    _, solar_fraction = zip(
        *sorted(
            zip(
                (
                    sensitivity_results := _sensitivity_fraction(
                        sensitivity_data, "solar_electricity_fraction", WEATHER_TYPE
                    )
                ).keys(),
                sensitivity_results.values(),
            )
        )
    )
    _, storage_fraction = zip(
        *sorted(
            zip(
                (
                    sensitivity_results := _sensitivity_fraction(
                        sensitivity_data, "storage_electricity_fraction", WEATHER_TYPE
                    )
                ).keys(),
                sensitivity_results.values(),
            )
        )
    )
    (axis := axes[1, 0]).plot(
        x,
        (
            grid_line := [
                storage_fraction[index] + grid_fraction[index] + solar_fraction[index]
                for index in range(len(storage_fraction))
            ]
        ),
        color=f"C2",
        label="grid fraction",
    )
    axis.plot(
        x,
        (
            solar_line := [
                storage_fraction[index] + solar_fraction[index]
                for index in range(len(storage_fraction))
            ]
        ),
        color=f"C1",
        label="solar fraction",
    )
    axis.plot(x, storage_fraction, color=f"C0", label="storage fraction")
    axis.fill_between(
        x, [0] * len(storage_fraction), storage_fraction, color="C0", alpha=0.7
    )
    axis.fill_between(x, storage_fraction, solar_line, color="C1", alpha=0.7)
    axis.fill_between(x, solar_line, grid_line, color="C2", alpha=0.7)
    axis.set_xlabel("Fractional cost change")
    axis.set_ylabel("Fraction")
    axis.set_title("Tijuana, Mexico")
    axis.text(
        -0.08,
        1.1,
        "b.",
        transform=axis.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    axis.legend()
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlim(-0.2, 0.2)
    # La Paz data
    x, grid_fraction = zip(
        *sorted(
            zip(
                (
                    sensitivity_results := _sensitivity_fraction(
                        (
                            sensitivity_data := {
                                key: value
                                for key, value in data_to_plot.items()
                                if regex.match(key).group("frac_var")
                                == fraction_variable_name
                                and regex.match(key).group("location") == "la_paz"
                            }
                        ),
                        "grid_electricity_fraction",
                        WEATHER_TYPE,
                    )
                ).keys(),
                sensitivity_results.values(),
            )
        )
    )
    _, solar_fraction = zip(
        *sorted(
            zip(
                (
                    sensitivity_results := _sensitivity_fraction(
                        sensitivity_data, "solar_electricity_fraction", WEATHER_TYPE
                    )
                ).keys(),
                sensitivity_results.values(),
            )
        )
    )
    _, storage_fraction = zip(
        *sorted(
            zip(
                (
                    sensitivity_results := _sensitivity_fraction(
                        sensitivity_data, "storage_electricity_fraction", WEATHER_TYPE
                    )
                ).keys(),
                sensitivity_results.values(),
            )
        )
    )
    (axis := axes[1, 1]).plot(
        x,
        (
            grid_line := [
                storage_fraction[index] + grid_fraction[index] + solar_fraction[index]
                for index in range(len(storage_fraction))
            ]
        ),
        color=f"C2",
        label="grid fraction",
    )
    axis.plot(
        x,
        (
            solar_line := [
                storage_fraction[index] + solar_fraction[index]
                for index in range(len(storage_fraction))
            ]
        ),
        color=f"C1",
        label="solar fraction",
    )
    axis.plot(x, storage_fraction, color=f"C0", label="storage fraction")
    axis.fill_between(
        x, [0] * len(storage_fraction), storage_fraction, color="C0", alpha=0.7
    )
    axis.fill_between(x, storage_fraction, solar_line, color="C1", alpha=0.7)
    axis.fill_between(x, solar_line, grid_line, color="C2", alpha=0.7)
    axis.set_xlabel("Fractional cost change")
    axis.set_ylabel("Fraction")
    axis.set_title("La Paz, Mexico")
    axis.text(
        -0.08,
        1.1,
        "d.",
        transform=axis.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    axis.legend()
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlim(-0.2, 0.2)


def plot_auxiliary_heating_sensitivity(
    data_to_plot, fraction_variable_name: str = "grid"
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.25)
    WEATHER_TYPE: str = "average_weather_conditions"
    # WEATHER_TYPE: str = "upper_error_bar_weather_conditions"
    # WEATHER_TYPE: str = "lower_error_bar_weather_conditions"
    # Plot the fractions of power from storage, pv, and the grid.
    x, auxiliary_heating_fraction = zip(
        *sorted(
            zip(
                (
                    sensitivity_results := _sensitivity_fraction(
                        (
                            sensitivity_data := {
                                key: value
                                for key, value in data_to_plot.items()
                                if regex.match(key).group("frac_var")
                                == fraction_variable_name
                                and regex.match(key).group("location") == "abu_dhabi"
                            }
                        ),
                        "auxiliary_heating_fraction",
                        WEATHER_TYPE,
                    )
                ).keys(),
                sensitivity_results.values(),
            )
        )
    )
    (axis := axes[0, 0]).plot(
        x,
        (
            solar_line := [
                1 - auxiliary_heating_fraction[index]
                for index in range(len(auxiliary_heating_fraction))
            ]
        ),
        color=f"C1",
        label="solar-thermal fraction",
    )
    axis.plot(
        x,
        (
            auxiliary_line := [
                auxiliary_heating_fraction[index] + solar_line[index]
                for index in range(len(auxiliary_heating_fraction))
            ]
        ),
        color=f"C3",
        label="auxiliary-heating fraction",
    )
    axis.fill_between(x, [0] * len(solar_line), solar_line, color="C1", alpha=0.7)
    axis.fill_between(x, solar_line, auxiliary_line, color="C3", alpha=0.7)
    axis.set_xlabel("Fractional cost change")
    axis.set_ylabel("Fraction")
    axis.set_title("Abu Dhabi, UAE")
    axis.text(
        -0.08,
        1.1,
        "a.",
        transform=axis.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    axis.legend()
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlim(-0.2, 0.2)
    # Gando data
    x, auxiliary_heating_fraction = zip(
        *sorted(
            zip(
                (
                    sensitivity_results := _sensitivity_fraction(
                        (
                            sensitivity_data := {
                                key: value
                                for key, value in data_to_plot.items()
                                if regex.match(key).group("frac_var")
                                == fraction_variable_name
                                and regex.match(key).group("location") == "gran_canaria"
                            }
                        ),
                        "auxiliary_heating_fraction",
                        WEATHER_TYPE,
                    )
                ).keys(),
                sensitivity_results.values(),
            )
        )
    )
    (axis := axes[0, 1]).plot(
        x,
        (
            solar_line := [
                1 - auxiliary_heating_fraction[index]
                for index in range(len(auxiliary_heating_fraction))
            ]
        ),
        color=f"C1",
        label="solar-thermal fraction",
    )
    axis.plot(
        x,
        (
            auxiliary_line := [
                auxiliary_heating_fraction[index] + solar_line[index]
                for index in range(len(auxiliary_heating_fraction))
            ]
        ),
        color=f"C3",
        label="auxiliary-heating fraction",
    )
    axis.fill_between(x, [0] * len(solar_line), solar_line, color="C1", alpha=0.7)
    axis.fill_between(x, solar_line, auxiliary_line, color="C3", alpha=0.7)
    axis.set_xlabel("Fractional cost change")
    axis.set_ylabel("Fraction")
    axis.set_title("Gando, Gran Canaria")
    axis.text(
        -0.08,
        1.1,
        "b.",
        transform=axis.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    axis.legend()
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlim(-0.2, 0.2)
    # Tijuana data
    x, auxiliary_heating_fraction = zip(
        *sorted(
            zip(
                (
                    sensitivity_results := _sensitivity_fraction(
                        (
                            sensitivity_data := {
                                key: value
                                for key, value in data_to_plot.items()
                                if regex.match(key).group("frac_var")
                                == fraction_variable_name
                                and regex.match(key).group("location") == "tijuana"
                            }
                        ),
                        "auxiliary_heating_fraction",
                        WEATHER_TYPE,
                    )
                ).keys(),
                sensitivity_results.values(),
            )
        )
    )
    (axis := axes[1, 0]).plot(
        x,
        (
            solar_line := [
                1 - auxiliary_heating_fraction[index]
                for index in range(len(auxiliary_heating_fraction))
            ]
        ),
        color=f"C1",
        label="solar-thermal fraction",
    )
    axis.plot(
        x,
        (
            auxiliary_line := [
                auxiliary_heating_fraction[index] + solar_line[index]
                for index in range(len(auxiliary_heating_fraction))
            ]
        ),
        color=f"C3",
        label="auxiliary-heating fraction",
    )
    axis.fill_between(x, [0] * len(solar_line), solar_line, color="C1", alpha=0.7)
    axis.fill_between(x, solar_line, auxiliary_line, color="C3", alpha=0.7)
    axis.set_xlabel("Fractional cost change")
    axis.set_ylabel("Fraction")
    axis.set_title("Tijuana, Mexico")
    axis.text(
        -0.08,
        1.1,
        "b.",
        transform=axis.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    axis.legend()
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlim(-0.2, 0.2)
    # La Paz data
    x, auxiliary_heating_fraction = zip(
        *sorted(
            zip(
                (
                    sensitivity_results := _sensitivity_fraction(
                        (
                            sensitivity_data := {
                                key: value
                                for key, value in data_to_plot.items()
                                if regex.match(key).group("frac_var")
                                == fraction_variable_name
                                and regex.match(key).group("location") == "la_paz"
                            }
                        ),
                        "auxiliary_heating_fraction",
                        WEATHER_TYPE,
                    )
                ).keys(),
                sensitivity_results.values(),
            )
        )
    )
    (axis := axes[1, 1]).plot(
        x,
        (
            solar_line := [
                1 - auxiliary_heating_fraction[index]
                for index in range(len(auxiliary_heating_fraction))
            ]
        ),
        color=f"C1",
        label="solar-thermal fraction",
    )
    axis.plot(
        x,
        (
            auxiliary_line := [
                auxiliary_heating_fraction[index] + solar_line[index]
                for index in range(len(auxiliary_heating_fraction))
            ]
        ),
        color=f"C3",
        label="auxiliary-heating fraction",
    )
    axis.fill_between(x, [0] * len(solar_line), solar_line, color="C1", alpha=0.7)
    axis.fill_between(x, solar_line, auxiliary_line, color="C3", alpha=0.7)
    axis.set_xlabel("Fractional cost change")
    axis.set_ylabel("Fraction")
    axis.set_title("La Paz, Mexico")
    axis.text(
        -0.08,
        1.1,
        "d.",
        transform=axis.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    axis.legend()
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlim(-0.2, 0.2)


["dumped_electricity"]

# Abu Dhabi HIST
hist_plot(abu_dhabi_data)

# Gran Canaria HIST
hist_plot(gran_canaria_data)

# La Paz HIST
hist_plot(la_paz_data)

# Tijuana HIST
hist_plot(tijuana_data)

hist_plot(abu_dhabi_joo)

hist_plot(abu_dhabi_rahimi)

hist_plot(abu_dhabi_el)


hist_plot(tijuana_joo)

hist_plot(tijuana_rahimi)

hist_plot(tijuana_el)

#########################################
# Inverter-battery sensitivity analysis #
#########################################

# Define levels in z-axis where we want lines to appear
levels = np.array(
    [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.45,
        1.5,
        1.55,
        1.6,
        1.65,
        1.7,
        1.75,
        1.8,
        1.85,
        1.9,
        2,
    ]
)

# Generate the regex
regex = re.compile(
    r"hpc_(?P<location>abu_dhabi|gran_canaria|tijuana|la_paz)_(?P<plant>joo|el_nashar|rahimi|el)_(?P<pv>[^_]*)_(?P<pvt>[^_]*)_(?P<st>[^_]*)_(?P<batt_cycles>\d*)_batt_cycles_(?P<inverter_lifetime>\d*)_inverter_years"
)

# Filter out the battery-inverter data points
battery_inverter_data = {
    key: value for key, value in data.items() if regex.match(key) is not None
}
rahimi_battery_inverter_data = {
    key: value
    for key, value in data.items()
    if regex.match(key).group("plant") == "rahimi"
}
el_battery_inverter_data = {
    key: value for key, value in data.items() if "el" in regex.match(key).group("plant")
}
joo_battery_inverter_data = {
    key: value
    for key, value in data.items()
    if regex.match(key).group("plant") == "joo"
}

# Generate a color-map palette in-keeping with the colorblind colour scheme
colorblind_cmap = colorblind_palette = sns.color_palette(
    [
        "#E04606",  # Orange
        "#F09F52",  # Pale orange
        "#52C0AD",  # Pale green
        "#006264",  # Green
        "#D8247C",  # Pink
        "#EDEDED",  # Pale pink
        "#E7DFBE",  # Pale yellow
        "#FBBB2C",  # Yellow
    ],
    as_cmap=True,
)

colorblind_colours = [
    "#E04606",  # Orange
    "#F09F52",  # Pale orange
    "#52C0AD",  # Pale green
    "#006264",  # Green
    "#D8247C",  # Pink
    "#EDEDED",  # Pale pink
    "#E7DFBE",  # Pale yellow
    "#FBBB2C",  # Yellow
]

COLORBLIND_COLOURS: dict[str, str] = {
    "storage_electricity_fraction": "#E04606",  # Orange
    "solar_electricity_fraction": "#F09F52",  # Pale orange
    "grid_electricity_fraction": "#52C0AD",  # Pale green
    "auxiliary_heating_fraction": "#006264",  # Green
    "total_cost": "#D8247C",  # Pink
    "components_cost": "#E04606",  # Orange
    "grid_costs": "#F09F52",  # Pale orange
    "inverters_cost": "#52C0AD",  # Pale green
    "heat_pump_cost": "#006264",  # Green
    # "#EDEDED",  # Pale pink
    # "#E7DFBE",  # Pale yellow
    # "#FBBB2C",  # Yellow
}


def _subfigure_plot(
    data_to_plot,
    axis,
    figure,
    label: str,
    location: str,
    variable: str,
    weather_conditions: str = "average_weather_conditions",
):
    battery_lifetimes = [
        float(regex.match(key).group("batt_cycles"))
        for key in data_to_plot
        if regex.match(key).group("location") == location
    ]
    inverter_lifetimes = [
        float(regex.match(key).group("inverter_lifetime"))
        for key in data_to_plot
        if regex.match(key).group("location") == location
    ]
    plotting_variable = _results_value(
        {
            key: value
            for key, value in data_to_plot.items()
            if regex.match(key).group("location") == location
        },
        None,
        (plotting_variable_name := variable),
        weather_conditions,
    )
    plotting_variable = [
        (entry / 10**6) if "cost" in label else entry for entry in plotting_variable
    ]
    # Generate the frame
    frame = pd.DataFrame(
        {
            (x_key := "Storage lifetime / cycles"): battery_lifetimes,
            # "Number of PV-T collectors": pv_t_sizes,
            (y_key := "Inverter lifetime / years"): inverter_lifetimes,
            (
                z_key := plotting_variable_name.replace("_", " ").capitalize()
            ): plotting_variable,
        }
    )
    pivotted_frame = frame.pivot(
        index=y_key,
        columns=x_key,
        values=z_key
        # index="Number of PV panels", columns="Number of PV-T collectors", values="Cost / MUSD"
    )
    # Extract the arrays.
    Z = pivotted_frame.values
    X_unique = np.sort(frame[x_key].unique())
    Y_unique = np.sort(frame[y_key].unique())
    X, Y = np.meshgrid(X_unique, Y_unique)
    # Interpolate the z-axis results if necessary
    triang = tri.Triangulation(battery_lifetimes, inverter_lifetimes)
    interpolator = tri.LinearTriInterpolator(triang, plotting_variable)
    Z_int = interpolator(X, Y)
    # Generate the plot
    cmap = sns.light_palette(COLORBLIND_COLOURS[variable], len(levels), as_cmap=True)
    cpf = axis.contourf(X, Y, Z_int, len(levels), cmap=cmap)
    # Set all level lines to black
    line_colors = ["black" for l in cpf.levels]
    # Make plot and customize axes
    contours = axis.contour(X, Y, Z_int, levels=levels, colors=line_colors)
    axis.clabel(contours, fontsize=10, colors=line_colors)
    axis.set_xlabel(x_key)
    axis.set_ylabel(y_key)
    figure.colorbar(cpf, ax=axis, label=label)


def plot_battery_inverter_contour_map(
    data_to_plot,
    label: str,
    variable: str,
    weather_conditions: str = "average_weather_conditions",
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.25)
    # Abu dhabi
    _subfigure_plot(
        data_to_plot,
        (axis := axes[0, 0]),
        fig,
        label,
        "abu_dhabi",
        variable,
        weather_conditions,
    )
    axis.set_title("Abu Dhabi, UAE")
    axis.text(
        -0.08,
        1.1,
        "a.",
        transform=axis.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    # Gando
    _subfigure_plot(
        data_to_plot,
        (axis := axes[0, 1]),
        fig,
        label,
        "gran_canaria",
        variable,
        weather_conditions,
    )
    axis.set_title("Gando, Gran Canaria")
    axis.text(
        -0.08,
        1.1,
        "b.",
        transform=axis.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    # Tijuana
    _subfigure_plot(
        data_to_plot,
        (axis := axes[1, 0]),
        fig,
        label,
        "tijuana",
        variable,
        weather_conditions,
    )
    axis.set_title("Tijuana, Mexico")
    axis.text(
        -0.08,
        1.1,
        "c.",
        transform=axis.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    # La Paz
    _subfigure_plot(
        data_to_plot,
        (axis := axes[1, 1]),
        fig,
        label,
        "la_paz",
        variable,
        weather_conditions,
    )
    axis.set_title("La Paz, Mexico")
    axis.text(
        -0.08,
        1.1,
        "d.",
        transform=axis.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )


plt.show()

batt_uniq, batt_index = np.unique(battery_lifetimes, return_inverse=True)
inverter_uniq, inverter_index = np.unique(inverter_lifetimes, return_inverse=True)

batt_mesh, inverter_mesh = np.meshgrid(batt_uniq, inverter_uniq)
plotting_interpolated = interpolator(batt_mesh, inverter_mesh)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
contours = (ax := axes[0, 0]).contourf(
    batt_mesh, inverter_mesh, plotting_interpolated, levels=100, cmap=cmap
)
fig.colorbar(contours, ax=ax)

##########################
# HPC Combination script #
##########################

import os
import json

from typing import Any

from tqdm import tqdm

data: dict[str, Any] = {}

for filename in tqdm(os.listdir("."), desc="files", unit="file"):
    if "feb" in filename:
        continue
    if "mar" in filename:
        continue
    with open(filename, "r", encoding="UTF-8") as f:
        data[filename] = json.load(f)

with open("28_mar_23.json", "w", encoding="UTF-8") as f:
    json.dump(data, f)
