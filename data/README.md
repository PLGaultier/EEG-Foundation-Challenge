# Data Directory

This directory contains the EEG datasets and related files.

## Structure

- `raw/` - Raw HBN-EEG dataset files (BIDS format)
- `processed/` - Preprocessed EEG data ready for training
- `splits/` - Train/validation/test splits
- `metadata/` - CBCL psychopathology dimensions and demographic info

## Dataset Information

The HBN-EEG dataset includes:

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

## Usage

Place your downloaded HBN-EEG dataset files in the `raw/` subdirectory following the BIDS format.