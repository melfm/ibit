#!/bin/bash

# dmc2gym
cd third_party/dmc2gym
pip install -e .
cd ../../
# dm_control
cd third_party/dm_control
pip install .
cd ../../
# metaworld
cd third_party/metaworld
pip install -e .
cd ../../