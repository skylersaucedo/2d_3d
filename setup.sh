#!/bin/bash

# Install OpenSCAD
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update
    sudo apt-get install -y openscad
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install openscad
elif [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "win32" ]]; then
    echo "Please install OpenSCAD from: https://openscad.org/downloads.html"
    echo "Make sure it's added to your system PATH"
fi

# Install Python dependencies
pip install -r requirements.txt 