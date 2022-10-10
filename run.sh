#!/usr/bin/env bash

# user defined items
DATA_DIR="MISSING" # directory to data (<DATASET_ROOT>)
WANDB_KEY="MISSING" # weights & biases api key
GPU='"device=0"' # default gpu


# define parameters mode, config
POSITIONAL=()

# default values
USE_WANDB="FALSE"
WITH_GPU=0
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --with_gpu)
      WITH_GPU="$2"
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

# setup internal variables
#USER_DOCKER=" -u $(id -u ${USER}):$(id -g ${USER})"
USER_DOCKER=""
if [[ "${POSITIONAL[@]}" == *"--wandb"* ]]; then
  USE_WANDB="TRUE"
  USER_DOCKER=""
fi

# setup with gpus
PREFIX=""
if [ "$WITH_GPU" -gt "0" ]; then
    PREFIX="with_gpu -n ${WITH_GPU} "
    GPU="device=\$CUDA_VISIBLE_DEVICES"
fi

set -- "${POSITIONAL[@]}" # restore positional parameters

echo "Start Benchmark with ${POSITIONAL[@]}"
echo "Using Weights&Biases = ${USE_WANDB}"

DEFAULT_CMD="docker run -it --rm -v $(pwd):/src -w /src -v ${DATA_DIR}:/data ${USER_DOCKER} --shm-size=42gb --env WANDB_API_KEY=${WANDB_KEY} --gpus ${GPU} dcic:latest python -m"

echo "RUN COMMAND: ${PREFIX} ${DEFAULT_CMD} ${POSITIONAL[@]}"

${PREFIX} docker run -it --rm -v $(pwd):/src -w /src -v ${DATA_DIR}:/data ${USER_DOCKER} --shm-size=42gb --env WANDB_API_KEY=${WANDB_KEY} --gpus ${GPU} dcic:latest python -m ${POSITIONAL[@]}
