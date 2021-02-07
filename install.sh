#!/usr/bin/env bash

conda remove --name gan --all
conda create --name gan python=3.8
conda activate gan

pip install --upgrade pip
pip install gdown
gdown https://drive.google.com/uc?id=1vgKReklsHDXRzNokWjUVkd-EIox6_f9_ -O /tmp/
unzip /tmp/summer2winter_yosemite.zip -d data
rm summer2winter_yosemite.zip
pip install -Ur requirements.txt
pip install -Ur requirements_dev.txt

pre-commit install