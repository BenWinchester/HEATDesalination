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

solution = result

TotalCost.calculate_value(
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

with open("initial_square.json", "r") as f:
    data = json.load(f)

costs = [
    entry["results"][ProfileType.AVERAGE.value][1][TotalCost.name] / 10 ** 6 for entry in data
]
# palette = sns.color_palette("blend:#0173B2,#64B5CD", as_cmap=True)
palette = sns.color_palette("rocket", as_cmap=True)

pv_sizes = [entry["simulation"]["pv_system_size"] for entry in data]
battery_capacities = [entry["simulation"]["battery_capacity"] for entry in data]

frame = pd.DataFrame({"Number of PV panels": pv_sizes, "Storage capacity / kWh": battery_capacities, "Cost / MUSD": costs}).pivot(index="Number of PV panels", columns="Storage capacity / kWh", values="Cost / MUSD")
sns.heatmap(frame, cmap=palette, annot=True)
plt.show()

##############################
# Setting up simulation runs #
##############################

default_entry = {(batt_key:='battery_capacity'): 20, 'buffer_tank_capacity': 15000, 'mass_flow_rate': 20, (pv_key:='pv_system_size)': 5000, (pv_t_key:='pv_t_system_size'): 185, 'solar_thermal_system_size': 218, 'scenario': 'default', 'start_hour': 8, 'system_lifetime': 25, 'output': None, 'profile_types': ['avr', 'usd', 'lsd', 'max', 'min']}

battery_capacities = range(400, 650, 10)
pv_sizes = range(5000, 6000, 40)
pv_t_sizes = range(75, 630, 23)
solar_thermal_sizes = range(218, 700, 20)

for 
