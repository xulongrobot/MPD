#!/bin/bash
echo "$HOME"
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/mpd-splines-public/lib:$LD_LIBRARY_PATH
export CPATH=$HOME/miniconda3/envs/mpd-splines-public/include:$CPATH
export PYTHONPATH=${THIS_DIR}:$PYTHONPATH
