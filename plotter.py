

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

with open("pv_t_72_st_218_tank_32_runs_output.json", "r") as f:
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

with open("pv_t_72_st_218_tank_32_runs_output.json", "r") as f:
    data = json.load(f)

costs = [
    (entry["results"][ProfileType.AVERAGE.value][TotalCost.name] / 10**6)
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

min_cost_index = {cost: index for index, cost in enumerate(costs)}[min(costs)]

scatter_palette = sns.color_palette("colorblind", n=7)

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

with open("min_cost_data.json", "r") as f:
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
    index="Number of PV panels", columns="Storage capacity / kWh", values="Cost / MUSD"
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
    (batt_key := "battery_capacity"): 170,
    (tank_key := "buffer_tank_capacity"): 100000,
    "mass_flow_rate": 20,
    (pv_key := "pv_system_size"): 6400,
    (pv_t_key := "pv_t_system_size"): 300,
    (st_key := "solar_thermal_system_size"): 300,
    "scenario": "default_uae",
    "start_hour": 8,
    "system_lifetime": 25,
    "output": None,
    "profile_types": ["avr", "usd", "lsd", "max", "min"],
}

battery_capacities = range(0, 1001, 100)
pv_sizes = range(0, 10001, 1000)
pv_t_sizes = range(72, 3600, 145)
solar_thermal_sizes = range(218, 1283, 44)
tank_capacities = range(15, 100, 80)

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

with open(os.path.join("inputs", "fifty_by_fifth_simulations.json"), "w") as f:
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

from typing import List, Dict

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
min_cost_overflow: Dict[str, float] = {}
output_to_display: List[str] = []

for filename in tqdm(output_filenames, desc="files", unit="file"):
    with open(filename, "r") as f:
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

with open("../min_cost_analysis.txt", "w") as f:
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
    with open(filename, "r") as f:
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

with open(f"pv_t_1262_st_318_tank_49_{algorithm}.json", "w") as f:
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

with open(os.path.join("inputs", "optimisations.json"), "w") as f:
    json.dump(optimisations, f)

##########################################
# Post-processing parallel optimisations #
##########################################

import json
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.signal import lfilter

# Noise-filtering parameters
n = 5  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1

sns.set_palette("colorblind")

with open("hpc_grid_optimisations_probe.json", "r") as f:
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
x = [
    entry_2.replace("m_", "-")
    for entry_2 in [
        entry_1.split("_dr_")[-1] for entry_1 in [entry.split("uae")[1] for entry in x]
    ]
]
x = [entry.replace("--", "-") for entry in x]
x = [float(entry) for entry in x]
x = [(-(200 + entry) if entry < 0 else entry) for entry in x]

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
        plt.plot(x, y, color=f"C{index}")
        plt.fill_between(x, y_ler, y, color=f"C{index}", alpha=0.5)
        plt.fill_between(x, y, y_uer, color=f"C{index}", alpha=0.5)
        # plt.plot(x, y_min, "--", color=f"C{index}")
        # plt.plot(x, y_max, "--", color=f"C{index}")
        plt.plot(x, y_uer, "--", color=f"C{index}")
        plt.plot(x, y_ler, "--", color=f"C{index}")
        plt.ylabel(key.replace("_", " ").capitalize())
        plt.xlabel("Mean grid discount rate / %/year")
        plt.xlim(-25, 25)
        plt.ylim(0, 1.5 * max(y[75:125]))
        plt.title(f"{key.replace('_', ' ').capitalize()} (unsmoothed) for {title.capitalize()}")
        plt.show()
        plt.plot(x, lfilter(b, a, y), color=f"C{index}")
        plt.fill_between(x, lfilter(b, a, y_ler), lfilter(b, a, y), color=f"C{index}", alpha=0.5)
        plt.fill_between(x, lfilter(b, a, y), lfilter(b, a, y_uer), color=f"C{index}", alpha=0.5)
        # plt.plot(x, y_min, "--", color=f"C{index}")
        # plt.plot(x, y_max, "--", color=f"C{index}")
        plt.plot(x, lfilter(b, a, y_uer), "--", color=f"C{index}")
        plt.plot(x, lfilter(b, a, y_ler), "--", color=f"C{index}")
        plt.ylabel(key.replace("_", " ").capitalize())
        plt.xlabel("Mean grid discount rate / %/year")
        plt.xlim(-25, 25)
        plt.ylim(0, 1.5 * max(y[75:125]))
        plt.title(f"{key.replace('_', ' ').capitalize()} (smoothed) for {title.capitalize()}")
        plt.show()
        with open(f"grid_high_res_weather_error_{key}.json", "w") as f:
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
    plt.legend()
    plt.show()
    # Plot without raised minima
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
    plt.legend()
    plt.show()

################################
# PV degradation probe results #
################################

import json
import matplotlib.pyplot as plt
import os
import seaborn as sns

sns.set_palette("colorblind")

with open("hpc_pv_degradation_optimisations_probe.json", "r") as f:
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
        with open(f"pv_degradation_rate_{key}.json", "w") as f:
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
    plt.legend()
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
    plt.legend()
    plt.show()


#######################
# Scenario-generation #
#######################

import os
import yaml

with open((scenarios_filepath := os.path.join("inputs", "scenarios.yaml")), "r") as f:
    scenarios = yaml.safe_load(f)[(scenarios_key := "scenarios")]

default_scenario = scenarios[0]

default_optimisation = {
    "location": "fujairah_united_arab_emirates",
    "output": "parallel_optimisation_output_1",
    "profile_types": ["avr", "ler", "uer"],
    "scenario": "default",
    "system_lifetime": 25,
}


new_scenarios = [default_scenario]
grid_optimisations = []

for discount_rate in range(-100, 100, 1):
    scenario = default_scenario.copy()
    scenario["discount_rate"] = float(discount_rate / 100)
    scenario[
        "name"
    ] = f"uae_dr_{'m_' if discount_rate < 0 else ''}{abs(discount_rate) // 100}{(abs(discount_rate) % 100) // 10}{(abs(discount_rate) % 100) % 10}"
    new_scenarios.append(scenario)
    optimisation = default_optimisation.copy()
    optimisation["scenario"] = scenario["name"]
    grid_optimisations.append(optimisation)

import numpy as np

pv_degradation_optimisations = []

for pv_degradation in np.linspace(0.007, 0.039, 100):
    scenario = default_scenario.copy()
    scenario["pv_degradation_rate"] = float(pv_degradation)
    scenario[
        "name"
    ] = f"uae_pv_deg_{int(round(100*pv_degradation//1,0))}.{int(round(100*round(100*pv_degradation%1,3), 0))}%"
    new_scenarios.append(scenario)
    optimisation = default_optimisation.copy()
    optimisation["scenario"] = scenario["name"]
    pv_degradation_optimisations.append(optimisation)


heat_exchanger_efficiency_optimisations = []

for heat_exchanger_efficiency in np.linspace(0, 1, 100):
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

for heat_pump_efficiency in np.linspace(0, 1, 100):
    scenario = default_scenario.copy()
    scenario["heat_pump_efficiency"] = float(heat_pump_efficiency)
    scenario[
        "name"
    ] = f"uae_heat_pump_eff_{int(round(100*heat_pump_efficiency//1,0))}.{int(round(100*round(100*heat_pump_efficiency%1,3), 0))}%"
    new_scenarios.append(scenario)
    optimisation = default_optimisation.copy()
    optimisation["scenario"] = scenario["name"]
    heat_pump_efficiency_optimisations.append(optimisation)


with open(scenarios_filepath, "w") as f:
    yaml.dump({"scenarios": new_scenarios}, f)

with open(os.path.join("inputs", "optimisations_pv_degradation.json"), "w") as f:
    json.dump(pv_degradation_optimisations, f)

with open(
    os.path.join("inputs", "optimisations_heat_exchanger_efficiency.json"), "w"
) as f:
    json.dump(heat_exchanger_efficiency_optimisations, f)

with open(os.path.join("inputs", "optimisations_heat_pump_efficiency.json"), "w") as f:
    json.dump(heat_pump_efficiency_optimisations, f)
