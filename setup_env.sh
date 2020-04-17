#!/bin/bash
if [ "$HOSTNAME" == "gen-uge-submit-p01" ] || [ "$HOSTNAME" == "gen-uge-submit-p02" ]; then
  echo "This script cannot be run from a submit host.  Pleas qlogin and try again."
  exit 1
fi

if hash conda 2>/dev/null; then
  echo "Using conda package manager found at $(command -v conda)"
else
  echo "Using shared conda package manager."
  eval "$(/ihme/covid-19/miniconda/bin/conda shell.bash hook)"
fi

dt=$(date '+%Y-%m-%d_%H-%M-%S') &&
echo "Creating environment covid-deaths-$dt" &&
umask 002
conda create -y --name=covid-deaths-"$dt" python=3.6 &&
conda activate covid-deaths-"$dt" &&
pip install --extra-index-url https://artifactory.ihme.washington.edu/artifactory/api/pypi/pypi-shared/simple/ -e .
