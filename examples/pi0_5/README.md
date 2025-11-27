# Pi0.5 Support in FlagScale

Pi0.5 extends Pi0 with enhanced architecture support for discrete state input and 32-dimensional actions.

## Pi0.5 vs Pi0 Key Differences

| Feature | Pi0 | Pi0.5 |
|---------|-----|-------|
| Action Dimension | 7 | **32** |
| Action Horizon | - | **16** |
| Max Token Length | 48 | **200** |
| State Input | Continuous | **Discrete** |
| Normalization | RMSNorm | **AdaRMSNorm** |
| Flow Matching | No | **Yes** |

# Setup Environment

Follow the same setup instructions as Pi0:

```bash
git clone https://github.com/FlagOpen/FlagScale.git
cd FlagScale/
```

Install dependencies according to main [README](https://github.com/FlagOpen/FlagScale/blob/main/README.md).

# Download Pi0.5 Model

```bash
git lfs install

mkdir -p /share/pi0_5
cd /share/pi0_5

# Download Pi0.5 models (when available)
git clone https://huggingface.co/lerobot/pi05_base
git clone https://huggingface.co/lerobot/pi05_droid
git clone https://huggingface.co/lerobot/pi05_libero
```

If no internet access:
```bash
modelscope download --model lerobot/pi05_base --local_dir /share/pi0_5/pi05_base
modelscope download --model lerobot/pi05_droid --local_dir /share/pi0_5/pi05_droid
modelscope download --model lerobot/pi05_libero --local_dir /share/pi0_5/pi05_libero
```

# Download Pi0.5 Dataset

```bash
mkdir -p /share/pi05_dataset
cd /share/pi05_dataset

# Download Pi0.5 datasets (when available)
git clone https://huggingface.co/datasets/lerobot/droid
mv droid/* ./droid/
```

# Download Tokenizer (reuse Pi0's)

```bash
mkdir -p /share/paligemma-3b-pt-224
cd /share/paligemma-3b-pt-224
git clone https://huggingface.co/google/paligemma-3b-pt-224
```

# Training

## Edit Config

```bash
cd FlagScale/
vim examples/pi0_5/conf/train/pi0_5.yaml
```

Change fields to match your paths:
- `model.checkpoint_dir` -> `/share/pi0_5/pi05_droid`
- `model.stat_path` -> `/share/pi05_dataset/droid/meta/stats.json`
- `data.tokenizer_path` -> `/share/paligemma-3b-pt-224`
- `data.data_path` -> `/share/pi05_dataset/droid/wds-2`

## Start Training

```bash
cd FlagScale/
python run.py --config-path ./examples/pi0_5/conf --config-name train action=run
```

# Inference

## Edit Config

```bash
cd FlagScale/
vim examples/pi0_5/conf/inference/pi0_5.yaml
```

Change fields:
- `engine.model` -> `/share/pi0_5/pi05_droid`
- `engine.stat_path` -> `/share/pi05_dataset/droid/meta/stats.json`
- `engine.tokenizer` -> `/share/paligemma-3b-pt-224`

## Start Inference

```bash
cd FlagScale/
python run.py --config-path ./examples/pi0_5/conf --config-name inference action=run
```

# Serving

## Quick Start (Mock Mode)

```bash
# Start Pi0.5 server (automatically uses Mock model if real model not found)
cd FlagScale/
python flagscale/serve/run_serve_pi0_5.py

# Test with client
python examples/pi0_5/client_pi0_5.py
```

## Production Mode

```bash
# Edit serving config
vim examples/pi0_5/conf/serve/pi0_5.yaml

# Run production server
gunicorn --config flagscale/serve/gunicorn_config.py \
         flagscale.serve.run_serve_pi0_5_production:create_wsgi_app()
```

# Test Server with Client

Download test images:
```bash
cd FlagScale/
wget https://gitee.com/hchnr/flag-scale/blob/robotics_dataset/orbbec_0_latest.jpg
wget https://gitee.com/hchnr/flag-scale/blob/robotics_dataset/orbbec_1_latest.jpg
wget https://gitee.com/hchnr/flag-scale/blob/robotics_dataset/orbbec_2_latest.jpg
```

Run client:
```bash
cd FlagScale/
python examples/pi0_5/client_pi0_5.py \
    --host 127.0.0.1 \
    --port 5000 \
    --base-img orbbec_0_latest.jpg \
    --left-wrist-img orbbec_1_latest.jpg \
    --right-wrist-img orbbec_2_latest.jpg \
    --num-steps 16
```

# Testing

Run Pi0.5 test suite:
```bash
# Component tests
python test_pi05_inference.py

# Integration tests
python test_pi05_server.py

# End-to-end tests
python test_pi05_end_to_end.py
```

# Expected Directory Structure

See [EXPECTED_SETUP.md](EXPECTED_SETUP.md) for detailed expected directory structure.

# Current Status

✅ **Ready for Testing**: Mock model works, API endpoints functional
⏳ **Waiting for Models**: Real Pi0.5 model weights not yet available
✅ **Data Ready**: Sample WebDataset data available
✅ **Documentation**: Complete setup and usage guides

# Key Files

- `flagscale/models/pi0/modeling_pi0_5.py` - Pi0.5 model implementation
- `flagscale/serve/run_serve_pi0_5.py` - Inference server
- `examples/pi0_5/client_pi0_5.py` - Python client
- `examples/pi0_5/conf/` - Configuration files
- `test_pi05_*.py` - Test suites

## **Directory Structure**

```
examples/pi0_5/
├── README.md                          # This file
├── conf/
│   └── inference/
│       └── pi0_5.yaml               # Pi0.5 inference configuration
└── client_pi0_5.py                   # Pi0.5 client for testing

flagscale/serve/
└── run_serve_pi0_5.py               # Pi0.5 inference server

flagscale/models/pi0/
└── modeling_pi0_5.py                 # Pi0.5 model implementation
```

## **Getting Started**

### **1. Model Preparation**

You need to download a Pi0.5 model checkpoint:

```bash
# Example path - replace with actual Pi0.5 model location
mkdir -p /share/pi0_5
# Download Pi0.5 model to /share/pi0_5/
```

### **2. Configuration**

Edit the inference configuration file:

```yaml
# examples/pi0_5/conf/inference/pi0_5.yaml
engine:
  model: /share/pi0_5                    # Path to Pi0.5 model
  tokenizer: /share/paligemma-3b-pt-224   # Tokenizer path
  stat_path: /share/lerobot/aloha_mobile_cabinet/meta/stats.json

generate:
  action_dim: 32                        # Pi0.5 uses 32-dim actions
  action_horizon: 16                    # Pi0.5 typical action horizon
  discrete_state_input: true             # Enable discrete state input
  max_token_len: 200                    # Pi0.5 supports longer sequences
  pi05_mode: true                       # Enable Pi0.5 mode
```

### **3. Start Inference Server**

```bash
# Start Pi0.5 inference server
python flagscale/serve/run_serve_pi0_5.py \
    --config examples/pi0_5/conf/inference/pi0_5.yaml

# The server will start on http://127.0.0.1:5000
```

### **4. Test with Client**

```bash
# Test Pi0.5 inference with client
python examples/pi0_5/client_pi0_5.py \
    --host 127.0.0.1 \
    --port 5000 \
    --images /path/to/camera0.jpg /path/to/camera1.jpg /path/to/camera2.jpg \
    --instruction "Pick up the red block and place it in the box"
```

## **API Reference**

### **Inference Server Endpoints**

#### **Health Check**
```bash
GET /health
```

Returns:
```json
{
  "status": "healthy",
  "model": "Pi0.5",
  "pi05_mode": true,
  "device": "cuda",
  "max_token_len": 200,
  "action_dim": 32,
  "action_horizon": 16
}
```

#### **Prediction**
```bash
POST /predict
```

Request body:
```json
{
  "instruction": "Pick up the red object",
  "states": [0.0, 0.1, 0.2, ...],  // 32-dim state vector
  "observation.images.camera0": "data:image/jpeg;base64,...",
  "observation.images.camera1": "data:image/jpeg;base64,...",
  "observation.images.camera2": "data:image/jpeg;base64,..."
}
```

Returns:
```json
{
  "success": true,
  "predictions": [...],  // Action sequence
  "pi05_mode": true,
  "action_horizon": 16
}
```

### **Key Parameters**

- **`pi05_mode`**: Enable Pi0.5 specific processing (default: true)
- **`discrete_state_input`**: Use discrete state tokens (default: true)
- **`max_token_len`**: Maximum token sequence length (default: 200)
- **`action_dim`**: Action dimensionality (default: 32)
- **`action_horizon`**: Action prediction horizon (default: 16)

## **Implementation Details**

### **Discrete State Processing**

Pi0.5 converts continuous states to discrete tokens:

```python
# Continuous state: [0.1, -0.2, 0.5, ...]
# Discrete tokens: [245, 512, 780, ...]  # Vocabulary indices
discrete_states = state_processor(continuous_states)
```

### **AdaRMSNorm**

Adaptive normalization based on flow matching timestep:

```python
# Normalize features with adaptive weighting
normalized_features = ada_norm(features, timestep)
```

### **Flow Matching Generation**

Pi0.5 uses flow matching for action generation:

```python
# Sample timestep and generate actions
timestep = torch.rand(batch_size) * flow_matching_timesteps
actions = policy.generate_flow_matching(images, states, instruction, timestep)
```

## **Migration from Pi0**

### **Configuration Changes**

1. Update `engine.model` to point to Pi0.5 checkpoint
2. Add Pi0.5 specific parameters in `generate` section
3. Set `pi05_mode: true`

### **Code Changes**

1. Import Pi0.5 classes: `from flagscale.models.pi0.modeling_pi0_5 import PI05Policy`
2. Use `PI05PolicyConfig` instead of `PI0PolicyConfig`
3. Enable Pi0.5 mode in inference calls

### **Model Differences**

- Pi0.5 requires larger action horizon (16 vs 10)
- Pi0.5 expects 32-dim state vectors
- Pi0.5 supports longer instruction sequences

## **Troubleshooting**

### **Common Issues**

1. **Model Loading Error**
   - Ensure Pi0.5 checkpoint is downloaded and accessible
   - Check model path in configuration file

2. **State Dimension Mismatch**
   - Pi0.5 expects 32-dim state vectors
   - Verify state input format

3. **Token Length Exceeded**
   - Pi0.5 supports max 200 tokens
   - Check instruction length and discretization

4. **Memory Issues**
   - Pi0.5 requires more memory due to longer sequences
   - Consider reducing batch size or action horizon

### **Debugging**

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check server logs for detailed error messages.

## **Contributing**

To extend Pi0.5 support:

1. Modify `flagscale/models/pi0/modeling_pi0_5.py` for model changes
2. Update `flagscale/serve/run_serve_pi0_5.py` for inference changes
3. Add tests in `examples/pi0_5/`

## **References**

- [OpenPI Repository](https://github.com/Physical-Intelligence/openpi)
- [Pi0.5 Blog Post](https://www.physicalintelligence.company/blog/pi05)
- [Knowledge Insulation Paper](https://www.physicalintelligence.company/research/knowledge_insulation)

---

**Note**: Pi0.5 support is currently in beta. Please report issues and feedback.