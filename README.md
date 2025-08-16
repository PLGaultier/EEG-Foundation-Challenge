# EEG Foundation Challenge - From Cross-Task to Cross-Subject EEG Decoding

The 2025 EEG Foundation Challenge: From Cross-Task to Cross-Subject EEG Decoding is a biosignal challenge accepted to the NeurIPS 2025 Competition Track. This competition aims to advance the field of EEG decoding by addressing two critical challenges:

1. **Cross-Task Transfer Learning**: Developing models that can effectively transfer knowledge from passive EEG tasks to active tasks
2. **Subject Invariant Representation**: Creating robust representations that generalize across different subjects while predicting clinical factors

📄 **Challenge Paper**: [arXiv:10.48550/arXiv.2506.19141](https://arxiv.org/abs/2506.19141)

## Two Competition Challenges

### Challenge 1: Cross-Task Transfer Learning
**Objective**: Predict behavioral performance metrics from active tasks using passive task EEG data
- **Input**: EEG from Surround Suppression (SuS) - passive paradigm
- **Output**: Behavioral metrics from Contrast Change Detection (CCD) - active paradigm
  - Response time (regression)
  - Success rate (classification)
- **Focus**: Cross-paradigm generalization from passive to active tasks

### Challenge 2: Psychopathology Factor Prediction (Subject Invariant)
**Objective**: Predict psychopathology scores with subject-invariant representations
- **Input**: EEG from multiple paradigms (RS, SuS, MW, CCD, SL, SyS)
- **Output**: 4 CBCL psychopathology dimensions (all regression)
  - p-factor, internalizing, externalizing, attention
- **Focus**: Cross-subject generalization and robust clinical prediction

## Dataset: HBN-EEG

The competition uses the HBN-EEG dataset with EEG recordings from over 3,000 participants across six distinct cognitive tasks:

### Passive Tasks
- **Resting State (RS)**: Eyes open/closed conditions with fixation cross
- **Surround Suppression (SuS)**: Four flashing peripheral disks with contrasting background
- **Movie Watching (MW)**: Four short films with different themes

### Active Tasks
- **Contrast Change Detection (CCD)**: Identifying dominant contrast in co-centric flickering grated disks
- **Sequence Learning (SL)**: Memorizing and reproducing sequences of flashed circles
- **Symbol Search (SyS)**: Computerized version of WISC-IV subtest

### Target Variables
- 4 psychopathology dimensions from Child Behavior Checklist (CBCL)
- Demographic information: age, sex, handedness

## Project Structure

```
EEG-Foundation-Challenge/
├── model_challenge1.py   # Challenge 1: Cross-task transfer model
├── model_challenge2.py   # Challenge 2: Psychopathology prediction model
├── model.py              # Template/development model
├── requirements.txt      # Python dependencies
├── src/                  # Source code
│   ├── models/          # Model architectures
│   ├── data/            # Data processing utilities
│   ├── training/        # Training loops and utilities
│   └── evaluation/      # Evaluation metrics
├── data/                # Dataset storage
│   ├── raw/             # Raw HBN-EEG data (BIDS format)
│   ├── processed/       # Preprocessed data
│   └── splits/          # Train/val/test splits
├── experiments/         # Experiment configs and results
├── notebooks/           # Jupyter notebooks for exploration
├── submission/          # Final submission files
│   ├── challenge1/      # Challenge 1 submission
│   └── challenge2/      # Challenge 2 submission
└── utils/               # Utility functions
```

## Quick Start

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Data**: Place HBN-EEG dataset in `data/raw/`

3. **Explore**: Start with notebooks in `notebooks/`

4. **Develop**: Build models in `src/models/`

5. **Submit**: Prepare final submission in `submission/`

## Submission Requirements

- Submit a `.zip` archive containing `model.py` (required) and `weights.pt` (optional)
- No folders allowed in the zip
- `Model` class must be instantiable without arguments
- Model automatically set to evaluation mode

## Development Workflow

### For Both Challenges:
1. **Data exploration and preprocessing** - Use notebooks for EDA
2. **Baseline model development** - Start with simple architectures
3. **Pretraining strategies** - Unsupervised/self-supervised learning

### Challenge 1 Specific:
4. **Cross-task transfer experiments** - SuS → CCD transfer
5. **Multi-objective optimization** - Balance regression + classification
6. **Cross-paradigm validation** - Test generalization

### Challenge 2 Specific:
4. **Subject invariant learning** - Domain adaptation techniques
5. **Multi-paradigm fusion** - Combine all task data
6. **Cross-subject validation** - Test robustness

### Final Steps:
7. **Model evaluation and analysis**
8. **Prepare two separate submissions**

## Submission Strategy

You can participate in **one or both challenges**:
- **Challenge 1 only**: Focus on cross-task transfer learning
- **Challenge 2 only**: Focus on subject-invariant psychopathology prediction  
- **Both challenges**: Leverage shared EEG representations and preprocessing

Each challenge requires a separate submission with its own `model.py` and optional weights file.
