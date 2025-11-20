# Environment
Begin at the root path of `FlagScale` repository:
1. Install backend
```
cd flagscale/runner/auto_tuner/simulator/custom_backend/
python setup.py develop
```

# Setup
2. Set necessary parameters in `config_gen.py`. For example:
```
device_type_list = ["A", "B"]
device_num_list = [4, 4]
global_batch_size = 32
num_micro_batches = 8
num_layers = 4
```
# Run a Task
3. Start the auto-tuning: 
  a. set PYTHONPATH 
   ```
   export PYTHONPATH=/***/FlagScale:$PYTHONPATH
   export PYTHONPATH=$PYTHONPATH:/***/FlagScale/third_party/Megatron-LM
    
   vim /***/FlagScale/flagscale/runner/auto_tuner/simulator/analylize_pipeline_time.py
   os.environ["PYTHONPATH"] = (
        "/***/FlagScale:"
        "/***/FlagScale/third_party/Megatron-LM"
    )
   ```
  b. run 
  
  vim flagscale/runner/auto_tuner/simulator/config_gen.py
  
  set scheme = vpp or scheme = 1F1B 
  
  python flagscale/runner/auto_tuner/simulator/config_gen.py

  c. result
  ```
  {'mesh': [2, 1, 1, 1, 2, 1, 1, 1, 1, 4], 'device_types': ['A800', 'A800'], 'pp_layer_split': [8, 8, 5, 5, 5, 1], 'recompute_granularity': None, 'recompute_method': 'uniform', 'recompute_num_layers': 1, 'simulated_time': 57.52105478485333, 'theory_peak_memory': [110.487650304, 118.80914944, 158.35625472, 158.35625472, 158.35625472, 42.519842816], 'oom_error': True}
  {'mesh': [2, 1, 1, 1, 2, 1, 1, 1, 1, 4], 'device_types': ['A800', 'A800'], 'pp_layer_split': [8, 7, 5, 5, 5, 2], 'recompute_granularity': None, 'recompute_method': 'uniform', 'recompute_num_layers': 1, 'simulated_time': 61.20105478485332, 'theory_peak_memory': [110.487650304, 109.345202176, 158.35625472, 158.35625472, 158.35625472, 61.447737344], 'oom_error': True}
  {'mesh': [2, 1, 1, 1, 2, 1, 1, 1, 1, 4], 'device_types': ['A800', 'A800'], 'pp_layer_split': [8, 8, 5, 5, 4, 2], 'recompute_granularity': None, 'recompute_method': 'uniform', 'recompute_num_layers': 1, 'simulated_time': 54.73105478485331, 'theory_peak_memory': [110.487650304, 118.80914944, 158.35625472, 158.35625472, 119.365943296, 61.447737344], 'oom_error': True}
...
```
