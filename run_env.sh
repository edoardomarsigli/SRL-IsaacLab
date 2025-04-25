#!/bin/bash

ISAAC_SIM_PATH="/home/edomrl/isaac-sim"
PROJECT_PATH="$(pwd)"

# Per CUDA/cupti
export LD_LIBRARY_PATH="$ISAAC_SIM_PATH/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/cuda_cupti/lib:$LD_LIBRARY_PATH"

# Per i moduli locali
export PYTHONPATH="$PROJECT_PATH/source/isaaclab:$PROJECT_PATH/source/isaaclab_tasks"

# Avvia Isaac Sim con la app Isaac Lab ufficiale
$ISAAC_SIM_PATH/isaac-sim.sh --no-window=false --app "omni.isaac.lab.app" \
    --/app/lab/load_script="${PROJECT_PATH}/prova_env.py"
