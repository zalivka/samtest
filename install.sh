#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Run checkpoints.sh
echo "Running checkpoints.sh..."
./checkpoints.sh

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "checkpoints.sh executed successfully."
else
    echo "Error: checkpoints.sh failed to execute properly."
    exit 1
fi

pip install -r requirements.txt
echo "All dependencies are installed successfully."
