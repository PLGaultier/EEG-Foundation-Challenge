# Submission Directory

This directory contains files ready for submission to the EEG Foundation Challenge.

The challenge has **two separate competitions** requiring **two separate submissions**.

## Challenge Structure

### Challenge 1: Cross-Task Transfer Learning
- **Objective**: Predict behavioral performance from CCD using SuS EEG data
- **Tasks**: Response time (regression) + Success rate (classification)
- **Directory**: `challenge1/`

### Challenge 2: Psychopathology Factor Prediction
- **Objective**: Predict 4 CBCL psychopathology scores from multi-paradigm EEG
- **Tasks**: p-factor, internalizing, externalizing, attention (all regression)
- **Directory**: `challenge2/`

## Submission Requirements

Each challenge submission must be a **separate** `.zip` archive containing:

### Required Files
- `model.py` - Contains the `Model` class (PyTorch nn.Module)
- `weights.pt` - Pre-trained model weights (optional)

### Constraints
- **No folders allowed** in each zip archive
- The `Model` class must be instantiable without arguments
- Model must be in evaluation mode after instantiation

## Directory Structure
```
submission/
├── challenge1/
│   ├── model.py              # Copy of model_challenge1.py
│   ├── weights_challenge1.pt # Challenge 1 weights
│   └── challenge1_submission.zip
├── challenge2/
│   ├── model.py              # Copy of model_challenge2.py
│   ├── weights_challenge2.pt # Challenge 2 weights
│   └── challenge2_submission.zip
└── README.md
```

## Preparation Steps

### For Challenge 1:
1. Copy `model_challenge1.py` to `challenge1/model.py`
2. Copy trained weights as `challenge1/weights_challenge1.pt`
3. Test the model works with CCD behavioral prediction
4. Create `challenge1_submission.zip`

### For Challenge 2:
1. Copy `model_challenge2.py` to `challenge2/model.py`
2. Copy trained weights as `challenge2/weights_challenge2.pt`
3. Test the model works with psychopathology prediction
4. Create `challenge2_submission.zip`

## Testing Your Submissions

### Test Challenge 1:
```python
# Test Challenge 1 model
import sys
sys.path.append('challenge1')
from model import Model

model = Model()
import torch
dummy_input = torch.randn(1, 64, 1000)  # SuS EEG data
output = model(dummy_input)
print(f"Response time shape: {output['response_time'].shape}")
print(f"Success rate shape: {output['success_rate'].shape}")
```

### Test Challenge 2:
```python
# Test Challenge 2 model
import sys
sys.path.append('challenge2')
from model import Model

model = Model()
import torch
dummy_input = torch.randn(1, 64, 1000)  # Multi-paradigm EEG data
output = model(dummy_input)
print(f"Psychopathology scores shape: {output.shape}")  # Should be (1, 4)
```

## Creating Submission Archives

```bash
# Create Challenge 1 submission
cd challenge1
zip challenge1_submission.zip model.py weights_challenge1.pt
cd ..

# Create Challenge 2 submission  
cd challenge2
zip challenge2_submission.zip model.py weights_challenge2.pt
cd ..
```