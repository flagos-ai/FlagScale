# After completing the environment setup based on the README.md file in the upper directory

# Function HighLight:
# 1)Support multiple instances,
# 2)Support cross node operation,
# 3)Support automatic expansion and contraction of capacity

## Program Run Command
python run.py --config-path examples/emu3.5/conf/ --config-name serve_emu3p5 action=run

## Program Stop Command
python run.py --config-path examples/emu3.5/conf/ --config-name serve_emu3p5 action=stop

## output result
Path of output image: examples/emu3.5/conf/serve/outputs

# parameter adjustment
You should adjust the parameters of DEFAULT_CONFIG in the emu3p5.py file to suit your need, including model paths, task types, etc

You should specify the working_dir path in the upper level configuration file serve_emu3p5.yaml, which contains the src folder in the local Emu3.5 project: https://github.com/baaivision/Emu3.5

note:  if you want to run command `python emu_vllm.py`, you should set the env : export PATHONPATH=/path/to/FlagScale:${PYTHONPATH}

# Client request
```
import requests
import uuid

# Service address
url = "http://127.0.0.1:9710/emu3p5"

# Prepare request data
data = {
    "prompt": "As shown in the second figure: The ripe strawberry rests on a green leaf in the garden. Replace the chocolate truffle in first image with ripe strawberry from 2nd image",
    "reference_image": ["./assets/ref_0.png", "./assets/ref_1.png"],  # Change to local image path
}

try:
    # Send POST request, assuming the service accepts JSON data
    response = requests.post(url, json=data)

    # Check response status code
    if response.status_code == 200:
        print("Request successful!")
        print("Response content:", response.text)
    else:
        print(f"Request failed, status code: {response.status_code}")
        print("Response content:", response.text)
except Exception as e:
    print("Fail", e)
```
