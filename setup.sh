#!/bin/sh
export PYTHONPATH=/home/cthota/machine_learning/workspaces/gpu_benchmark_tests:$PYTHONPATH
nohup python3 /home/cthota/machine_learning/workspaces/gpu_benchmark_tests/lstm_model/benchmark_model.py > output.log &
echo "script started"
