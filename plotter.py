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
pv_t_system_size = result.x[2]
solar_thermal_system_size = result.x[3]

buffer_tank.cacpacity = 100
buffer_tank_capacity = 100

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
    pv_t_system_size,
    scenario,
    solar_irradiances[profile_type],
    solar_thermal_collector,
    solar_thermal_system_size,
    system_lifetime,
    disable_tqdm=disable_tqdm,
)

print("Total cost = {} MUSD".format(TotalCost.calculate_value(
    {
        battery: battery_capacity,
        buffer_tank: buffer_tank_capacity,
        pv_panel: pv_system_size,
        hybrid_pv_t_panel: pv_t_system_size,
        solar_thermal_collector: solar_thermal_system_size,
    },
    scenario,
    solution,
    system_lifetime,
) / 10 ** 6))

###########################################
# Plotting costs of the surrounding areas #
###########################################

import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.heatdesalination.__utils__ import ProfileType
from src.heatdesalination.optimiser import TotalCost

with open("200_x_200_pv_batt_square.json", "r") as f:
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
cmap1 = sns.color_palette("blend:#00548F,#CA6630", as_cmap=True)
cmap1.set_over("none")
cmap2 = sns.color_palette("blend:#84BB4E,#00548F", as_cmap=True)
cmap2.set_over("none")

ax1 = sns.heatmap(frame, vmin=2, vmax=max(costs), cmap=cmap1, cbar_kws={'pad': 0.02})
sns.heatmap(frame, vmin=min(costs), vmax=2, cmap=cmap2, ax=ax1)

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

contours = ax.contourf(batt_mesh, pv_mesh, cost_mesh, 100, cmap="rocket")
fig.colorbar(contours, ax=ax)
plt.show()

cmap1 = sns.color_palette("blend:#00548F,#84BB4E", as_cmap=True)
cmap1.set_over("none")
cmap2 = sns.color_palette("blend:#84BB4E,#00548F", as_cmap=True)
cmap2.set_over("none")

contours = plt.contour(batt_mesh, pv_mesh, cost_mesh, 300, cmap=cmap1)
plt.xlabel("Battery capacity / kWh")
plt.ylabel("PV system size / collectors")
plt.colorbar(contours)
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
pv_t_sizes = range(72, 3600, 350)
solar_thermal_sizes = range(218, 1283, 100)
tank_capacities = range(15, 100, 8)

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
