########################################################################################
# hpc_template.sh - Script for executing HEATDesalination as an array job on the HPC.  #
#                                                                                      #
# Author: Ben Winchester                                                               #
# Copyright: Ben Winchester, 2022                                                      #
# Date created: 29/11/2022                                                             #
#                                                                                      #
# For more information, please email:                                                  #
#   benedict.winchester@gmail.com                                                      #
########################################################################################
#PBS -J 1-{NUM_RUNS}
#PBS -lwalltime={WALLTIME}:00:00
#PBS -lselect=1:ncpus=1:mem=11800Mb

echo -e "HPC array script executed"

# Load the anaconda environment
module load anaconda3/personal
source activate py310

cd $PBS_O_WORKDIR

# Sending runs to the HPC
echo -e "Running parallel simulation module..."
python3.10 -u -m src.heatdesalination.hpc_wrapper -r {RUNS_FILE} -w {WALLTIME}
echo -e "Complete"

cd $CURRENT_DIR

exit 0
