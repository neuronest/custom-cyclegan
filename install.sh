#!/usr/bin/env bash

conda remove --name gan --all
conda create --name gan python=3.8
conda activate gan

pip install --upgrade pip
pip install -Ur requirements.txt
pip install -Ur requirements_dev.txt

pre-commit install