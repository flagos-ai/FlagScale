#!/bin/bash

set -x

ulimit -n 65535 && source /root/miniconda3/bin/activate flagscale-inference

if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=/root/miniconda3/envs/flagscale-inference/lib/python3.12/site-packages:/root/FlagScale
else
    export PYTHONPATH="$PYTHONPATH:/root/miniconda3/envs/flagscale-inference/lib/python3.12/site-packages:/root/FlagScale"
fi

export CUDA_VISIBLE_DEVICES=0 && export CUDA_DEVICE_MAX_CONNECTIONS=1 && export no_proxy=127.0.0.1,localhost
ulimit -n 65535 && source /root/miniconda3/bin/activate flagscale-inference
mkdir -p /root/FlagScale/tests/functional_tests/serve/qwen2_5/test_results/0.5b/serve_logs
mkdir -p /root/FlagScale/tests/functional_tests/serve/qwen2_5/test_results/0.5b/serve_logs/pids

cd /root/FlagScale

cmd="CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 no_proxy=127.0.0.1,localhost python flagscale/serve/run_inference_engine.py --config-path=/root/FlagScale/tests/functional_tests/serve/qwen2_5/test_results/0.5b/serve_logs/scripts/serve.yaml --log-dir=/root/FlagScale/tests/functional_tests/serve/qwen2_5/test_results/0.5b/serve_logs"

echo '=========== launch task ==========='
nohup bash -c "$cmd; sync" >> /root/FlagScale/tests/functional_tests/serve/qwen2_5/test_results/0.5b/serve_logs/host_0_localhost.output 2>&1 & echo $! > /root/FlagScale/tests/functional_tests/serve/qwen2_5/test_results/0.5b/serve_logs/pids/host_0_localhost.pid

