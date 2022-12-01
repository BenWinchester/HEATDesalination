interact
import matplotlib.pyplot as plt

ALPHA = 1.0
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

ax.legend()
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

ax.legend()
plt.show()


######################################################
# Plotting weather profiles with standard deviations #
######################################################

import json
import matplotlib.pyplot as plt
import seaborn as sns
from src.heatdesalination.__utils__ import ProfileType

sns.set_palette("colorblind")

with open("auto_generated/fujairah_united_arab_emirates.json", "r") as f:
    data = json.load(f)

keywords_to_plot = ["irradiance", "ambient_temperature", "wind_speed"]
ylabels = [
    "Solar irradiance / W/m^2",
    "Ambient temperature / degrees Celcius",
    "Wind speed / m/s",
]
map_colours = ["C3", "C1", "C0"]

for index, keyword in enumerate(keywords_to_plot):
    mapping = {
        int(key): value
        for key, value in data[ProfileType.AVERAGE.value][keyword].items()
    }
    average_profile = {key: mapping[key] for key in sorted(mapping)}
    mapping = {
        int(key): value
        for key, value in data[ProfileType.LOWER_STANDARD_DEVIATION.value][
            keyword
        ].items()
    }
    lower_profile = {key: mapping[key] for key in sorted(mapping)}
    mapping = {
        int(key): value
        for key, value in data[ProfileType.UPPER_STANDARD_DEVIATION.value][
            keyword
        ].items()
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
    plt.legend()
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

with open("pv_t_1262_st_318_tank_49_output.json", "r") as f:
    data = json.load(f)

costs = [
    (entry["results"][ProfileType.AVERAGE.value][1][TotalCost.name] / 10**6)
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
    index="Number of PV panels", columns="Storage capacity / kWh", values="Cost / MUSD"
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
    index="Number of PV panels", columns="Storage capacity / kWh", values="Cost / MUSD"
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


with open("200_x_200_pv_batt_square.json", "r") as f:
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
from src.heatdesalination.optimiser import TotalCost

with open("pv_t_1262_st_318_tank_49_output.json", "r") as f:
    data = json.load(f)

costs = [
    (entry["results"][ProfileType.AVERAGE.value][1][TotalCost.name] / 10**6)
    for entry in data
]
# palette = sns.color_palette("blend:#0173B2,#64B5CD", as_cmap=True)
# palette = sns.color_palette("rocket", as_cmap=True)

pv_sizes = [entry["simulation"]["pv_system_size"] for entry in data]
battery_capacities = [entry["simulation"]["battery_capacity"] for entry in data]

# Generate the frame
frame = pd.DataFrame(
    {
        "Storage capacity / kWh": battery_capacities,
        "Number of PV panels": pv_sizes,
        "Cost / MUSD": costs,
    }
)
pivotted_frame = frame.pivot(
    index="Number of PV panels", columns="Storage capacity / kWh", values="Cost / MUSD"
)

# Extract the arrays.
Z = pivotted_frame.values
X_unique = np.sort(frame["Storage capacity / kWh"].unique())
Y_unique = np.sort(frame["Number of PV panels"].unique())
X, Y = np.meshgrid(X_unique, Y_unique)

# Define levels in z-axis where we want lines to appear
levels = np.array(
    [
        round(min(costs), 2) + 0.01,
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

min_cost_index = {cost: index for index, cost in enumerate(costs)}[min(costs)]

scatter_palette = sns.color_palette("colorblind", n=7)

# Open all vec files and scatter their journeys
# plt.scatter(
#     battery_capacities[min_cost_index],
#     pv_sizes[min_cost_index],
#     marker="x",
#     color="#A40000",
#     label="optimum point",
#     linewidths=2.5,
#     s=150,
#     zorder=1,
# )

# Nelder-Mead
with open("pv_t_1262_st_318_tank_49_nelder_mead_vecs.json", "r") as f:
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
with open("pv_t_1262_st_318_tank_49_powell.json", "r") as f:
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
with open("pv_t_1262_st_318_tank_49_cg.json", "r") as f:
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
with open("pv_t_1262_st_318_tank_49_bfgs.json", "r") as f:
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
with open("pv_t_1262_st_318_tank_49_l_bfgs_g.json", "r") as f:
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
with open("pv_t_1262_st_318_tank_49_tnc.json", "r") as f:
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

plt.legend()

plt.xlabel("Storage capacity / kWh")
plt.ylabel("Number of PV panels")
plt.xlim(min(battery_capacities), max(battery_capacities))
plt.ylim(min(pv_sizes), max(pv_sizes))

plt.show()

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
    (batt_key := "battery_capacity"): 20,
    (tank_key := "buffer_tank_capacity"): 100000,
    "mass_flow_rate": 20,
    (pv_key := "pv_system_size"): 5000,
    (pv_t_key := "pv_t_system_size"): 300,
    (st_key := "solar_thermal_system_size"): 300,
    "scenario": "default",
    "start_hour": 8,
    "system_lifetime": 25,
    "output": None,
    "profile_types": ["avr", "usd", "lsd", "max", "min"],
}

battery_capacities = range(0, 1001, 20)
pv_sizes = range(0, 10001, 200)
pv_t_sizes = range(72, 3600, 70)
solar_thermal_sizes = range(218, 1220, 20)
tank_capacities = range(15, 101, 34)

runs = []
for batt in battery_capacities:
    for pv in pv_sizes:
        entry = default_entry.copy()
        entry[batt_key] = batt
        entry[pv_key] = pv
        runs.append(entry)

with open(os.path.join("inputs", "pv_batt_square_200_x_200.json"), "w") as f:
    json.dump(runs, f)

runs = []
for batt in battery_capacities:
    for pv_t in pv_t_sizes:
        entry = default_entry.copy()
        entry[batt_key] = batt
        entry[pv_t_key] = pv_t
        runs.append(entry)

with open(os.path.join("inputs", "pv_t_batt_square_simulations.json"), "w") as f:
    json.dump(runs, f)

runs = []
for batt in battery_capacities:
    for st in solar_thermal_sizes:
        entry = default_entry.copy()
        entry[batt_key] = batt
        entry[st_key] = st
        runs.append(entry)


with open(os.path.join("inputs", "st_batt_square_simulations.json"), "w") as f:
    json.dump(runs, f)

runs = []
for pv in pv_sizes:
    for pv_t in pv_t_sizes:
        entry = default_entry.copy()
        entry[pv_key] = pv
        entry[pv_t_key] = pv_t
        runs.append(entry)

with open(os.path.join("inputs", "pv_pv_t_square_simulations.json"), "w") as f:
    json.dump(runs, f)

runs = []
for pv in pv_sizes:
    for st in solar_thermal_sizes:
        entry = default_entry.copy()
        entry[pv_key] = pv
        entry[st_key] = st
        runs.append(entry)

with open(os.path.join("inputs", "pv_st_square_simulations.json"), "w") as f:
    json.dump(runs, f)

runs = []
for pv_t in pv_t_sizes:
    for st in solar_thermal_sizes:
        entry = default_entry.copy()
        entry[pv_t_key] = pv_t
        entry[st_key] = st
        runs.append(entry)

with open(os.path.join("inputs", "pv_t_st_square_simulations.json"), "w") as f:
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

with open(os.path.join("inputs", "ten_by_ten_simulations.json"), "w") as f:
    json.dump(runs, f)

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
                os.path.join(basename.format(pv_t=pv_t, st=st, tank=tank)), "w"
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

from typing import Dict

import json

from tqdm import tqdm

from src.heatdesalination.__utils__ import ProfileType
from src.heatdesalination.optimiser import TotalCost

regex = re.compile(r"pv_t_(?P<pv_t>\d*)_st_(?P<st>\d*)_tank_(?P<tank>\d*)_runs_output")
output_filenames = [
    entry for entry in os.listdir(".") if regex.match(entry) is not None
]

# Cycle through the file names, compute the costs, and, if the file has a lower cost,
# save it as the lowest-cost filename.

min_cost: float = 10**10
min_cost_filename: str | None = None
min_cost_overflow: Dict[str, float] = {}

for filename in tqdm(output_filenames, desc="files", unit="file"):
    with open(filename, "r") as f:
        data = json.load(f)
    # Calculate the costs
    costs = [
        (entry["results"][ProfileType.AVERAGE.value][1][TotalCost.name] / 10**6)
        for entry in data
    ]
    # If the lowest cost is lower than the lowest value encountered so far, use this.
    if (current_minimum_cost := min(costs)) < min_cost:
        min_cost_filename = filename
        min_cost = current_minimum_cost
        continue
    # If the lowest cost is equal to the lowest value encountered so far, save this.
    if current_minimum_cost == min_cost:
        min_cost_overflow[filename] = current_minimum_cost

####################
# Dump all vectors #
####################

interact

import json

algorithm = "cobyla"
vecs = [list(entry) for entry in result.allvecs]

with open(f"pv_t_1262_st_318_tank_49_{algorithm}.json", "w") as f:
    json.dump(vecs, f)

##########################
# Creating optimisations #
##########################

default_optimisation = {
    "location": "fujairah_united_arab_emirates",
    (output_key:="output"): "parallel_optimisation_output_1",
    "profile_types": [
        "avr", "usd", "lsd", "max", "min"
    ],
    (scenario_key:="scenario"): "default",
    (system_lifetime_key:="system_lifetime"): 25,
}

output = "fujairah_uae_{}"
optimisations = []

for discount_rate in range(-20, 21, 1):
    scenario = "uae_dr_{}".format(f"{'m_' if discount_rate < 0 else ''}{f'{round(discount_rate/10, 2)}'.replace('.', '').replace('-','')}")
    optimisation = default_optimisation.copy()
    optimisation[output_key] = output.format(scenario)
    optimisation[scenario_key] = scenario
    optimisations.append(optimisation)

with open(os.path.join("inputs","optimisations.json"), "w") as f:
    json.dump(optimisations, f)
