#PBS -lwalltime=01:00:00
#PBS -lselect=1:ncpus=8:mem=11800Mb

echo -e "HPC script executed"

# Load the anaconda environment
module load anaconda3/personal
source activate py310

cd $PBS_O_WORKDIR

# Sending runs to the HPC
echo -e "Running parallel simulation module..."
python3.10 -u -m src.heatdesalination.parallel_simulator -l {LOCATION} -o {OUTPUT} -s {SIMULATIONS}
echo -e "Complete"

cd $CURRENT_DIR

exit 0
