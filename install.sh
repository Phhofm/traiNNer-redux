#!/bin/bash

python3.12 -m venv venv
source venv/bin/activate
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -U pillow-avif-plugin
pip install .

echo "traiNNer-redux dependencies installed successfully!"
