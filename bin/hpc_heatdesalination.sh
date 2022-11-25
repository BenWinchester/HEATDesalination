#PBS -lwalltime=01:00:00
#PBS -lselect=1:ncpus=8:mem=11800Mb

echo -e "HPC script executed"

# Load the anaconda environment
module load anaconda3/personal
source activate py310

cd $PBS_O_WORKDIR


POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -l|--location)
      LOCATION="$2"
      shift # past argument
      shift # past value
      ;;
    -o|--output)
      OUTPUT="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done


# Sending runs to the HPC
echo -e "Running parallel simulation module..."
python3.10 -u -m src.heatdesalination.parallel_simulator --location $LOCATION --output $OUTPUT
echo -e "Complete"

cd $CURRENT_DIR

exit 0
