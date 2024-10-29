#!/bin/bash

# dataset
source dataset.sh


# CNN
cd cnn
source prepare_data_train.sh
cd ..


echo "All process has completed!"