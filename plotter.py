interact
import matplotlib.pyplot as plt

ALPHA = 1.0
fig, ax = plt.subplots()

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
    battery_storage_profile.keys(),
    battery_storage_profile.values(),
    "--",
    label="battery state of charge",
    color="C0",
)

###################
# Plot solar bars #
###################
ax.bar(
    solar_power_supplied.keys(),
    solar_power_supplied.values(),
    alpha=ALPHA,
    label="solar power supplied",
    color="C1",
)
bottom = list(solar_power_supplied.values())

ax.bar(
    battery_power_input_profile.keys(),
    battery_power_input_profile.values(),
    alpha=0.6,
    label="power to batteries",
    bottom=bottom,
    color="C1",
)
bottom = [
    entry + battery_power_input_profile[index] for index, entry in enumerate(bottom)
]

ax.bar(
    dumped_solar.keys(),
    dumped_solar.values(),
    alpha=0.3,
    label="dumped solar",
    bottom=bottom,
    color="C1",
)
bottom = [entry + dumped_solar[index] for index, entry in enumerate(bottom)]

#####################
# Plot battery bars #
#####################
ax.bar(
    battery_power_supplied_profile.keys(),
    battery_power_supplied_profile.values(),
    alpha=ALPHA,
    label="storage power supplied",
    bottom=bottom,
    color="C0",
)
bottom = [
    entry + battery_power_supplied_profile[index] for index, entry in enumerate(bottom)
]

##################
# Plot grid bars #
##################

ax.bar(
    grid_profile.keys(),
    grid_profile.values(),
    alpha=ALPHA,
    label="grid power supplied",
    bottom=bottom,
    color="C2",
)

plt.xlabel("Hour of day")
plt.ylabel("Average hourly power flow / kWh")

ax.legend()
plt.show()
