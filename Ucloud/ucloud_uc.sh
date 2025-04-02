#!/bin/bash
# Print starting directory
echo "Starting script in directory: $(pwd)"
# List contents and install package
echo "Now in directory: $(pwd)"
ls -l /work
ls -l /work/rl_project/EconDLSolvers
# Install editable package
pip install -e /work/rl_project/EconDLSolvers
pip install -r /work/rl_project/EconDLSolvers/requirements.txt
# Other packages
pip install EconModel Consav
pip install torch torchvision torchaudio
pip install nvidia-ml-py
pip install line_profiler
pip install papermill
pip install scipy
pip install seaborn
# Keep shell open
exec bash