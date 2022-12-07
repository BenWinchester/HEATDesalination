#PBS -lwalltime=06:00:00
#PBS -lselect=1:ncpus=8:mem=11800Mb

echo -e "HPC script executed"

# Load the anaconda environment
module load anaconda3/personal
source activate py310

cd $PBS_O_WORKDIR

# OPTIMISATIONS_FILE="optimisations_pv_degradation"
# OPTIMISATIONS_FILE="grid_optimisations"
# OPTIMISATIONS_FILE="inputs/optimisations_heat_pump_efficiency.json"
OPTIMISATIONS_FILE="inputs/optimisations_heat_exchanger_efficiency.json"

OUTPUT_NAME="hpc_heat_exchanger_efficiency_optimisations_probe"
# OUTPUT_NAME="hpc_heat_pump_efficiency_optimisations_probe"
# OUTPUT_NAME="hpc_grid_optimisations_probe"
# OUTPUT_NAME="hpc_pv_degradation_optimisations_probe"

# Sending runs to the HPC
echo -e "Running parallel simulation module..."
python3.10 -u -m src.heatdesalination.parallel_optimiser -o $OPTIMISATIONS_FILE -out $OUTPUT_NAME
echo -e "Complete"

cd $CURRENT_DIR

exit 0
