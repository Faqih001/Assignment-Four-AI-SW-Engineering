#!/bin/bash

# Kaggle Dataset Download Script for Breast Cancer Competition
# This script demonstrates how to download the Kaggle dataset

echo "Kaggle Breast Cancer Dataset Download Instructions"
echo "=================================================="
echo ""

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "Installing Kaggle API..."
    pip install kaggle
else
    echo "✓ Kaggle API is already installed"
fi

echo ""
echo "Steps to download the dataset:"
echo "1. Go to https://www.kaggle.com/account and create API credentials"
echo "2. Download kaggle.json and place it in ~/.kaggle/kaggle.json"
echo "3. Set permissions: chmod 600 ~/.kaggle/kaggle.json"
echo "4. Run the download command:"
echo ""
echo "   kaggle competitions download -c iuss-23-24-automatic-diagnosis-breast-cancer"
echo ""
echo "5. Extract the dataset:"
echo "   unzip iuss-23-24-automatic-diagnosis-breast-cancer.zip -d /kaggle/input/iuss-23-24-automatic-diagnosis-breast-cancer/"
echo ""

# Create directory structure for local testing
echo "Creating local directory structure for testing..."
mkdir -p "/kaggle/input/iuss-23-24-automatic-diagnosis-breast-cancer/training_set"

echo ""
echo "Note: The notebook will automatically fallback to scikit-learn dataset"
echo "      if the Kaggle dataset is not available."
echo ""
echo "Directory created: /kaggle/input/iuss-23-24-automatic-diagnosis-breast-cancer/training_set"
echo "Place your downloaded CSV files in this directory."
