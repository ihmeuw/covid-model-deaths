#!/bin/bash
eval "$(/ihme/covid-19/miniconda/bin/conda shell.bash hook)" &&
dt=$(date '+%Y-%m-%d_%H-%M-%S') &&
echo "Creating environment covid-deaths-$dt" &&
umask 002
conda create -y --name=covid-deaths-"$dt" python=3.6 &&
conda activate covid-deaths-"$dt" &&
pip install --extra-index-url https://artifactory.ihme.washington.edu/artifactory/api/pypi/pypi-shared/simple/ -e .
