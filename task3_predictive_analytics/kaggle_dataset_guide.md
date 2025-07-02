# Kaggle Dataset Integration Guide

## Overview

The notebook has been updated to support the Kaggle Breast Cancer Competition dataset as the primary data source, with automatic fallback to the scikit-learn breast cancer dataset if the Kaggle data is not available.

## Dataset Configuration

### Primary: Kaggle Competition Dataset
- **Competition**: iuss-23-24-automatic-diagnosis-breast-cancer
- **Location**: `/kaggle/input/iuss-23-24-automatic-diagnosis-breast-cancer/training_set`
- **Command**: `kaggle competitions download -c iuss-23-24-automatic-diagnosis-breast-cancer`

### Fallback: Scikit-learn Dataset
- **Source**: `sklearn.datasets.load_breast_cancer()`
- **Features**: 30 tumor characteristics
- **Samples**: 569 cases

## Implementation Details

### Smart Dataset Loading
The notebook includes intelligent dataset detection:

```python
root_dir = '/kaggle/input/iuss-23-24-automatic-diagnosis-breast-cancer/training_set'

def load_kaggle_dataset(root_dir):
    """Load Kaggle breast cancer dataset if available"""
    if os.path.exists(root_dir):
        # Look for CSV files and load the dataset
        csv_files = glob.glob(os.path.join(root_dir, "*.csv"))
        if csv_files:
            df_kaggle = pd.read_csv(csv_files[0])
            return df_kaggle, True
    return None, False
```

### Automatic Target Detection
The system automatically detects target columns using common naming conventions:
- `diagnosis`
- `target`
- `label`
- `class`
- `y`

### Data Preprocessing Adaptations
- **Kaggle Dataset**: Uses all numeric features, handles string labels
- **Scikit-learn Dataset**: Uses original 30 tumor characteristics
- **Priority Scoring**: Adapts to available features for resource allocation

## Usage Instructions

### Option 1: Use Kaggle Dataset (Recommended)

1. **Install Kaggle API**:
   ```bash
   pip install kaggle
   ```

2. **Set up Kaggle credentials**:
   - Go to https://www.kaggle.com/account
   - Create new API token
   - Download `kaggle.json`
   - Place in `~/.kaggle/kaggle.json`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. **Download the dataset**:
   ```bash
   kaggle competitions download -c iuss-23-24-automatic-diagnosis-breast-cancer
   ```

4. **Extract to correct location**:
   ```bash
   mkdir -p /kaggle/input/iuss-23-24-automatic-diagnosis-breast-cancer/training_set
   unzip iuss-23-24-automatic-diagnosis-breast-cancer.zip -d /kaggle/input/iuss-23-24-automatic-diagnosis-breast-cancer/
   ```

5. **Run the notebook** - it will automatically detect and use the Kaggle dataset

### Option 2: Use Automatic Fallback

1. **Simply run the notebook** without downloading Kaggle data
2. The system will automatically detect the missing Kaggle dataset
3. It will fallback to the scikit-learn breast cancer dataset
4. All functionality remains intact

## Benefits of This Approach

### Flexibility
- Works with either dataset source
- No manual configuration required
- Automatic detection and adaptation

### Robustness
- Graceful fallback if Kaggle data unavailable
- Handles different data formats automatically
- Consistent API regardless of data source

### Educational Value
- Demonstrates real-world data loading scenarios
- Shows how to handle multiple data sources
- Teaches dataset preprocessing techniques

## Output Examples

### With Kaggle Dataset
```
USING KAGGLE BREAST CANCER COMPETITION DATASET
===============================================
Kaggle Dataset Information:
- Total samples: [varies by competition]
- Total features: [varies by dataset]
- Target column: 'diagnosis'
- Target distribution: [competition-specific]
```

### With Fallback Dataset
```
FALLBACK: USING SCIKIT-LEARN BREAST CANCER DATASET
===================================================
Scikit-learn Dataset Information:
- Total samples: 569
- Total features: 30
- Target classes: ['malignant' 'benign']
```

## File Structure

```
task3_predictive_analytics/
├── ai_software_engineering_assignment.ipynb  # Main notebook with dataset integration
├── download_kaggle_dataset.sh               # Automated download script
├── kaggle_dataset_guide.md                  # This documentation
└── predictive_analytics.py                  # Standalone Python version
```

## Technical Implementation

### Modified Sections
1. **Section 1**: Added file handling imports (`glob`, `pathlib`)
2. **Section 9**: Complete dataset loading rewrite with smart detection
3. **Section 10**: Preprocessing adapted for both dataset types

### Key Features
- **Automatic Detection**: Checks for Kaggle data presence
- **Dynamic Feature Selection**: Adapts to available columns
- **Robust Error Handling**: Graceful fallback mechanisms
- **Consistent Interface**: Same API regardless of data source

This implementation ensures the assignment works perfectly whether you have access to the Kaggle competition data or not, while demonstrating professional-grade data handling practices.
