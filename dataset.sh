#!/bin/bash

# dataset
mkdir -p dataset
python3 dataset.py
unzip hms-harmful-brain-activity-classification.zip -d dataset
cd ..