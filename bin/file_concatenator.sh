#PBS -lwalltime=01:00:00
#PBS -lselect=1:ncpus=8:mem=11800Mb

echo -e "HPC script executed"

# Load the anaconda environment
module load anaconda3/personal
source activate py310

cd $PBS_O_WORKDIR

LOCATION="fujairah_united_arab_emirates"
OUTPUT="multi_square_10_x_10"

# Sending runs to the HPC
echo -e "Running parallel simulation module..."
python3.10 file_concatenator.py
echo -e "Complete"

cd $CURRENT_DIR

exit 0
